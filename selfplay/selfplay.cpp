// selfplay/selfplay.cpp —— 自对弈样本生成器
//
// 双方都用 MCTS 驱动（先验/价值来自评估函数，将来换成网络），逐步记录
//   (观测, 根节点访问分布 π, 终局结果 z)
// 这是 AlphaZero 式策略迭代的数据来源。
//
// 【为什么必须有噪声和采样】本游戏完全确定性：双方跑同一个确定性策略会
// 每局下出**完全相同的一盘棋**，没有多样性就没有梯度。所以自对弈必须
// ① 根部先验加 Dirichlet 噪声 ② 按访问分布带温度采样落子。
// 只有部署时才用 argmax（确定性）。
//
// 【信息隔离】生成器持有世界真实状态，但交给策略的必须是**遮罩后的视图**：
// 看不见对手时只能用信念锚点，绝不能把真实位置喂进去——否则是对弈数据
// 在作弊，学出来的策略上真机就废了。
//
// 用法:
//   ./selfplay --games 1000 --out data.bin [--threads N] [--budget-ms 20]

#include "brain/actions.h"
#include "brain/belief_state.h"
#include "brain/mcts.h"
#include "obs/encode.h"
#include "brain/net.h"
#include "sim/rules.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    int games = 100;
    int threads = 8;
    int budget_ms = 20;
    int temp_moves = 12; // 前 N 步按温度采样，之后取访问次数最多
    double temp = 1.0;
    double dirichlet_alpha = 0.3;
    double dirichlet_eps = 0.25;
    unsigned seed = 1;
    std::string out;
} g_opt;

struct Sample {
    float obs[obs::kObsDim];
    float pi[brain::kActionDim];
    float z = 0.0f;  // 终局结果（从 side 视角：胜 +1 / 平 0 / 负 -1）
    char side = 'R'; // 这条样本是站在哪一方记录的
};

struct SideState {
    char color = 'R';
    brain::SideBelief belief;
    int used = 0;
    bool free_turn = false;
    Pos last_end_pos{-1, -1};
};

struct LocalView {
    sim::State state;
    bool enemy_visible = false;
};

// 构造某方的局部视图：看得见才写真实位置，否则只给信念锚点
LocalView make_view(const sim::State& world, char color, const brain::SideBelief& b) {
    LocalView v;
    v.state = sim::to_local(world, color);
    const Pos true_enemy = v.state.blue.last_known_pos;
    v.enemy_visible = sim::can_see(v.state.red, true_enemy, v.state.obstacles);
    if (!v.enemy_visible) {
        v.state.blue.last_known_pos = b.anchor;
        v.state.blue.last_known_facing = b.anchor_facing;
    }
    v.state.blue.visible = v.enemy_visible;
    // 对手的 CD 在真机观测里恒为 -1（引擎不暴露），这里必须保持一致
    v.state.blue.fire_cd = -1;
    v.state.blue.scan_cd = -1;
    return v;
}

brain::MctsConfig mcts_config(unsigned seed, int move_index) {
    brain::MctsConfig cfg;
    cfg.seed = seed;
    cfg.add_root_noise = true; // 确定性游戏自对弈的必要条件
    cfg.dirichlet_alpha = g_opt.dirichlet_alpha;
    cfg.dirichlet_eps = g_opt.dirichlet_eps;
    cfg.sample_temperature = (move_index < g_opt.temp_moves) ? g_opt.temp : 0.0;
    return cfg;
}

// 跑一局，把样本追加到 out
void play_game(unsigned seed, std::vector<Sample>& out) {
    const brain::Weights weights; // 目前用手工评估；将来换成网络
    sim::State world = sim::make_initial_state();

    SideState sides[2];
    sides[0].color = 'R';
    sides[1].color = 'B';
    for (SideState& sd : sides) {
        sd.free_turn = true; // 开局双方各有一次免费转向
        sd.last_end_pos = sim::spawn_of(sd.color, world.size);
    }

    out.clear();
    int move_index = 0;
    const int max_rounds = 40; // 兜底，防止逻辑漏洞导致死循环

    for (int round = 0; round < max_rounds; ++round) {
        bool finished = false;
        for (int si = 0; si < 2 && !finished; ++si) {
            SideState& sd = sides[si];

            LocalView v = make_view(world, sd.color, sd.belief);
            sd.belief.begin_turn(v.state, world.turn, v.enemy_visible);
            v = make_view(world, sd.color, sd.belief);

            // 免费转向资格：开局，或刚从别处被击回出生点
            const Pos my_spawn = sim::spawn_of('R', world.size); // 局部坐标恒为 (0,0)
            const bool at_spawn = sim::is_at_spawn(v.state, 'R');
            sd.free_turn = (world.turn == 0) ||
                           (at_spawn && !(sd.last_end_pos.x == my_spawn.x &&
                                          sd.last_end_pos.y == my_spawn.y));
            sd.used = 0;

            for (int step = 0; step < brain::kMaxActionsPerTurn; ++step) {
                brain::TurnInput in;
                in.state = v.state;
                in.belief = sd.belief.set;
                in.used_by_now = sd.used;
                in.free_turn_available = sd.free_turn;
                in.enemy_visible = v.enemy_visible;

                const auto deadline =
                    Clock::now() + std::chrono::milliseconds(g_opt.budget_ms);
                const brain::MctsResult r = brain::mcts_search(
                    in, weights, mcts_config(seed + static_cast<unsigned>(move_index * 7919),
                                             move_index),
                    deadline);
                ++move_index;

                // π 全零 = 搜索被时间预算截断、根节点一次都没扩展。
                // 这种样本没有策略信号，留着只会污染训练集，直接丢弃。
                float pi_sum = 0.0f;
                for (int k = 0; k < brain::kActionDim; ++k) pi_sum += r.pi[k];
                if (pi_sum > 1e-6f) {
                    Sample s;
                    obs::encode(v.state, sd.belief.set, v.enemy_visible, sd.used,
                                sd.free_turn, s.obs);
                    std::memcpy(s.pi, r.pi, sizeof(s.pi));
                    s.side = sd.color;
                    out.push_back(s);
                }

                if (!r.plan.valid || r.plan.count == 0) break;
                const brain::Cand local_c{r.plan.actions[0], r.plan.args[0]};
                if (local_c.action == brain::kStopAction) break; // 主动收手

                const brain::Cand world_c = brain::local_to_world(local_c, sd.color);
                bool hit = false;
                if (!brain::apply_step(world, sd.color, world_c, sd.used, sd.free_turn,
                                       &hit)) {
                    break; // 搜索保证可行；真出现就结束本阶段
                }
                if (hit) sd.belief.on_our_hit(world.size);

                v = make_view(world, sd.color, sd.belief);
                sd.belief.after_action(v.state);
                v = make_view(world, sd.color, sd.belief);
            }

            brain::end_phase(world, sd.color);
            sd.last_end_pos = sim::spawn_of('R', world.size); // 局部坐标下的出生点
            {
                const LocalView after = make_view(world, sd.color, sd.belief);
                sd.last_end_pos = after.state.red.last_known_pos;
            }
            if (brain::is_terminal(world)) finished = true;
        }
        if (finished) break;
    }

    // 填终局结果 z（从每条样本自己的视角）
    for (Sample& s : out) {
        const Sentry& mine = world.sentry_for(s.side);
        const Sentry& theirs = world.sentry_for(s.side == 'R' ? 'B' : 'R');
        s.z = mine.score > theirs.score ? 1.0f
              : (mine.score < theirs.score ? -1.0f : 0.0f);
    }
}

void write_header(FILE* f) {
    const char magic[4] = {'S', 'D', 'S', 'P'};
    std::uint32_t version = 1;
    std::uint32_t obs_dim = obs::kObsDim;
    std::uint32_t act_dim = brain::kActionDim;
    std::fwrite(magic, 1, 4, f);
    std::fwrite(&version, sizeof(version), 1, f);
    std::fwrite(&obs_dim, sizeof(obs_dim), 1, f);
    std::fwrite(&act_dim, sizeof(act_dim), 1, f);
}

} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if (a == "--games") { const char* v = next(); if (v) g_opt.games = std::atoi(v); }
        else if (a == "--threads") { const char* v = next(); if (v) g_opt.threads = std::atoi(v); }
        else if (a == "--budget-ms") { const char* v = next(); if (v) g_opt.budget_ms = std::atoi(v); }
        else if (a == "--temp-moves") { const char* v = next(); if (v) g_opt.temp_moves = std::atoi(v); }
        else if (a == "--temp") { const char* v = next(); if (v) g_opt.temp = std::atof(v); }
        else if (a == "--seed") { const char* v = next(); if (v) g_opt.seed = static_cast<unsigned>(std::atoi(v)); }
        else if (a == "--out") { const char* v = next(); if (v) g_opt.out = v; }
        else { std::fprintf(stderr, "未知参数: %s\n", a.c_str()); return 1; }
    }
    if (g_opt.out.empty()) {
        std::fprintf(stderr,
                     "用法: %s --games N --out FILE [--threads N] [--budget-ms MS]\n",
                     argv[0]);
        return 1;
    }

    std::printf("自对弈生成: %d 局, %d 线程, 每步 %dms 预算\n", g_opt.games,
                g_opt.threads, g_opt.budget_ms);
    std::printf("网络: %s\n", brain::net_available()
                                  ? "已加载（根部先验来自策略网络）"
                                  : "未加载（回退手工评估；先跑 tools/train_mlp.py）");

    const auto t0 = Clock::now();
    std::atomic<int> next_game{0};
    std::atomic<long long> total_samples{0};
    std::vector<std::thread> pool;
    std::vector<std::string> parts(g_opt.threads);

    for (int t = 0; t < g_opt.threads; ++t) {
        parts[t] = g_opt.out + ".part" + std::to_string(t);
        pool.emplace_back([t, &parts, &next_game, &total_samples]() {
            FILE* f = std::fopen(parts[t].c_str(), "wb");
            if (f == nullptr) return;
            write_header(f);
            std::vector<Sample> samples;
            for (;;) {
                const int idx = next_game.fetch_add(1);
                if (idx >= g_opt.games) break;
                play_game(g_opt.seed + static_cast<unsigned>(idx) * 104729u, samples);
                if (!samples.empty()) {
                    std::fwrite(samples.data(), sizeof(Sample), samples.size(), f);
                    total_samples += static_cast<long long>(samples.size());
                }
                const int done = idx + 1;
                if (done % 50 == 0 || done == g_opt.games) {
                    std::printf("  进度 %d/%d  样本 %lld\n", done, g_opt.games,
                                total_samples.load());
                    std::fflush(stdout);
                }
            }
            std::fclose(f);
        });
    }
    for (std::thread& th : pool) th.join();

    // 合并分片为一个文件
    FILE* out = std::fopen(g_opt.out.c_str(), "wb");
    if (out == nullptr) { std::fprintf(stderr, "无法写 %s\n", g_opt.out.c_str()); return 1; }
    write_header(out);
    for (const std::string& p : parts) {
        FILE* in = std::fopen(p.c_str(), "rb");
        if (in == nullptr) continue;
        std::fseek(in, 16, SEEK_SET); // 跳过分片头
        char buf[1 << 16];
        std::size_t n;
        while ((n = std::fread(buf, 1, sizeof(buf), in)) > 0) std::fwrite(buf, 1, n, out);
        std::fclose(in);
        std::remove(p.c_str());
    }
    std::fclose(out);

    const double secs =
        std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("完成: %lld 条样本, 用时 %.1fs (%.1f 样本/秒)\n", total_samples.load(),
                secs, secs > 0 ? total_samples.load() / secs : 0.0);
    std::printf("写入 %s\n", g_opt.out.c_str());
    return 0;
}

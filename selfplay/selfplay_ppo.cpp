// selfplay/selfplay_ppo.cpp —— 阶段③ 的 PPO 轨迹生成器
//
// 与 selfplay.cpp（阶段②，产出 MCTS 的 (obs, π, z) 样本）的区别：
//   · 动作由**策略网络直接采样**，不是 MCTS 搜索
//   · 记录的是 (obs, action, reward) 序列，**不是 π**，也**不记 logprob**
//   · 观测用 obs v3（无手工信念），视图走 sim::ViewMirror（引擎口径的 Intel）
//
// 【为什么只存动作、不存 logprob】
// PPO 的重要性比率要用 old_logprob。如果那个值由 C++ 手写的 GRU 算出来，
// 它和 torch 之间的任何数值差异都会直接污染比率——而两份实现的漂移是**不会
// 报错**的，只会让训练悄悄变质。所以这里只存动作，old_logprob 由 Python 侧
// 在更新前用自己的前向重算。手写 GRU 于是只需要"行为足够接近"，
// 这一点由 tools/difftest_policy.py 卡住。
//
// 【红蓝必须各成一条序列】
// 双方共用同一个策略网络（引擎对蓝方镜像 + 我们 to_local，所以"我"恒在 red
// 槽位），但 hidden 状态**绝不能跨方共享**——那等于把对手的思考过程灌进
// 自己的记忆。所以每局写出两条独立序列，各自从零初始化 hidden。
//
// 格式（SDPP）：
//   magic "SDPP" 4B | version u32 | obs_dim u32 | act_dim u32 | games u32
//   每局：for side in {R,B}: steps u32; 然后 steps 条
//           { obs f32[obs_dim], action u8, reward f32 }
//   每条序列都以真实终局结束，所以不需要 done 标志，bootstrap 值恒为 0。
//
// 用法：
//   ./selfplay_ppo --games 2000 --threads 160 --out data.bin
//                   [--temperature 1.0] [--seed 1]

#include "brain/actions.h"
#include "brain/policy_net.h"
#include "obs/encode_v3.h"
#include "sim/rules.h"
#include "sim/view_mirror.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    int games = 100;
    int threads = 8;
    double temperature = 1.0;
    unsigned seed = 1;
    std::string out;
} g_opt;

constexpr char kMagic[4] = {'S', 'D', 'P', 'P'};
constexpr uint32_t kVersion = 1;

// 击杀得分。与引擎 board.cpp 的 `me.score += 2` 一致。
constexpr float kKillReward = 2.0f;
// ── 奖励设计（方案 A：纯边际回报）──
//
// 只保留**真实计分**：击杀 +2（引擎加分处）、占点 +1（阶段末结算处）。
// 于是每条序列的回报**恰好等于该方的最终净胜分**——不多不少。
//
// 删掉的两项及原因（都实测过）：
//   · 势能塑形：轨迹求和望远镜抵消（Σ F ≈ γᵀΦ(s_T) − Φ(s₀)），与路径无关
//     ——这正是它"不改变最优策略"的原因，也意味着它几乎不改变优势函数。
//     实测密度 0.6%→72.8% 而强度零变化。
//   · 终局 ±3：净胜分本身已编码符号与大小，阶跃项冗余。
//
// 配套：训练侧 γ=1.0、λ=1.0（MC 优势）。回报 = 净胜分，无偏且不依赖弱 critic。

struct Step {
    float obs[obs::kObsDimV3];
    uint8_t action = 0;
    float reward = 0.0f;
};

struct SideTraj {
    std::vector<Step> steps;
    float hidden[brain::kPolicyHidden];
};

// 按温度采样。**确定性游戏的自对弈必须带随机性**，否则每一局都完全相同，
// 数据里没有第二个样本。
int sample_action(const float* logits, double temperature, std::mt19937& rng) {
    const int n = brain::kPolicyActDim;
    float mx = logits[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, logits[i]);

    const double inv = temperature > 1e-6 ? 1.0 / temperature : 0.0;
    double sum = 0.0;
    double p[brain::kPolicyActDim];
    for (int i = 0; i < n; ++i) {
        // temperature -> 0 时退化为取最大，避免除零
        p[i] = std::exp((logits[i] - mx) * (inv > 0.0 ? inv : 1e6));
        sum += p[i];
    }
    std::uniform_real_distribution<double> uni(0.0, sum);
    double r = uni(rng);
    for (int i = 0; i < n; ++i) {
        r -= p[i];
        if (r <= 0.0) return i;
    }
    return n - 1;
}

void append_step(SideTraj& t, const float* obs, int action, float reward) {
    Step s;
    std::memcpy(s.obs, obs, sizeof(s.obs));
    s.action = static_cast<uint8_t>(action);
    s.reward = reward;
    t.steps.push_back(s);
}

// ── 对手池（方案 B）──
// 蓝方可以用一份**运行时加载**的历史权重，打破"永远和当前自己打"的循环退化
// （实测 r4→r5 回退）。红方永远用当前权重。hidden 双方本来就各自一份。
brain::PolicyWeights g_opp{};
bool g_has_opp = false;

// 跑一局，把红蓝两条序列写进 out_red / out_blue。
void play_game(unsigned seed, SideTraj& traj_r, SideTraj& traj_b) {
    std::mt19937 rng(seed);
    sim::State world = sim::make_initial_state();
    sim::ViewMirror mirror;
    mirror.reset();

    SideTraj* traj[2] = {&traj_r, &traj_b};
    traj_r.steps.clear();
    traj_b.steps.clear();
    for (SideTraj* t : traj) brain::policy_reset_hidden(t->hidden);

    // 对应引擎的 respawn_turn_pending_：开局双方各有一次免费转向，
    // 之后每当被击中就重新获得一次。
    bool respawn_pending[2] = {true, true};

    const int max_rounds = 40; // 兜底，防逻辑漏洞导致死循环
    bool finished = false;

    for (int round = 0; round < max_rounds && !finished; ++round) {
        for (int si = 0; si < 2; ++si) {
            const char side = (si == 0) ? 'R' : 'B';
            SideTraj& t = *traj[si];

            mirror.begin_phase(world, side);
            bool free_turn = respawn_pending[si];
            respawn_pending[si] = false;
            int used = 0;

            for (int k = 0; k < brain::kMaxActionsPerTurn; ++k) {
                const sim::State view = mirror.local_view(world, side);
                float obs[obs::kObsDimV3];
                obs::encode_v3(view, mirror.enemy_visible(side), used, free_turn, obs);

                float logits[brain::kPolicyActDim];
                float value = 0.0f;
                bool ok;
                if (si == 1 && g_has_opp) {
                    ok = brain::policy_forward_w(g_opp, obs, t.hidden, logits, &value);
                } else {
                    ok = brain::policy_forward(obs, t.hidden, logits, &value);
                }
                if (!ok) {
                    // 没有权重就不该跑到这里——上游必须先导出 policy_weights.h
                    std::fprintf(stderr, "selfplay_ppo: 策略网络不可用\n");
                    std::exit(1);
                }
                const int a = sample_action(logits, g_opt.temperature, rng);

                append_step(t, obs, a, 0.0f);

                if (a == brain::kActionDim - 1) break; // 收手，结束本阶段

                const brain::Cand world_c =
                    brain::local_to_world(brain::index_to_cand(a), side);
                bool hit = false;
                if (!brain::apply_step(world, side, world_c, used, free_turn, &hit)) {
                    // 策略选了当前不可行的动作。**这是策略的问题，不是规则的**，
                    // 但我们不能就此中止——继续本阶段剩下的额度，让它从后果里学。
                    continue;
                }
                if (hit) {
                    // 击杀：即时奖励记在**开火那一步**（引擎在这里 +2 分）
                    t.steps.back().reward += kKillReward;
                    // 对手回出生点并重新获得一次免费转向
                    respawn_pending[1 - si] = true;
                }
                mirror.after_action(world, side, world_c.action);
                if (used >= brain::kMaxActionsPerTurn) break;
            }

            // 阶段结束结算占点分（引擎 end_side_turn：在得分区则 +1）。
            // 这一分记在本方**行动阶段的最后一步**上——它是在此刻才确定的。
            const int before = world.sentry_for(side).score;
            sim::end_side_turn(world, side);
            const int gained = world.sentry_for(side).score - before;
            if (gained != 0 && !t.steps.empty()) {
                t.steps.back().reward += static_cast<float>(gained);
            }

            if (brain::is_terminal(world)) {
                finished = true;
                break;
            }
        }
        if (finished) break;
        sim::end_round(world);
        if (brain::is_terminal(world)) finished = true;
    }

    // 终局不再追加奖励：各步真实计分的累计已经**就是**净胜分。
}

// —— 输出：一个全局互斥锁保护的文件追加 ——
std::mutex g_write_mutex;
FILE* g_out = nullptr;

void write_game(const SideTraj& a, const SideTraj& b) {
    std::lock_guard<std::mutex> lock(g_write_mutex);
    for (const SideTraj* t : {&a, &b}) {
        const uint32_t n = static_cast<uint32_t>(t->steps.size());
        std::fwrite(&n, sizeof(n), 1, g_out);
        for (const Step& s : t->steps) {
            // 逐字段写，**不要 memcpy 整个 struct**——struct 有填充字节，
            // 会让 Python 侧的结构体布局与实际不符。
            std::fwrite(s.obs, sizeof(float), obs::kObsDimV3, g_out);
            std::fwrite(&s.action, 1, 1, g_out);
            std::fwrite(&s.reward, sizeof(float), 1, g_out);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    auto next = [&](int& i) -> const char* { return (i + 1 < argc) ? argv[++i] : nullptr; };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--games") { const char* v = next(i); if (v) g_opt.games = std::atoi(v); }
        else if (a == "--threads") { const char* v = next(i); if (v) g_opt.threads = std::atoi(v); }
        else if (a == "--temperature") { const char* v = next(i); if (v) g_opt.temperature = std::atof(v); }
        else if (a == "--seed") { const char* v = next(i); if (v) g_opt.seed = static_cast<unsigned>(std::atoi(v)); }
        else if (a == "--out") { const char* v = next(i); if (v) g_opt.out = v; }
        else if (a == "--opp-weights") {
            const char* v = next(i);
            if (v != nullptr) {
                if (!brain::policy_load_bin(v, g_opp)) {
                    std::fprintf(stderr, "对手权重加载失败: %s\n", v);
                    return 1;
                }
                g_has_opp = true;
            }
        }
        else if (a == "-h" || a == "--help") {
            std::printf("用法: %s --games N --out FILE [--threads N] [--temperature T] [--seed S]\n",
                        argv[0]);
            return 0;
        }
    }
    if (g_opt.out.empty() || g_opt.games <= 0) {
        std::fprintf(stderr, "用法: %s --games N --out FILE [--threads N]\n", argv[0]);
        return 2;
    }
    if (!brain::policy_available()) {
        std::fprintf(stderr,
                     "没有编译进策略权重。先跑 tools/difftest_policy.py 或 "
                     "tools/train_ppo.py 导出 brain/policy_weights.h，再重新 make。\n");
        return 1;
    }
    if (g_opt.threads < 1) g_opt.threads = 1;

    g_out = std::fopen(g_opt.out.c_str(), "wb");
    if (g_out == nullptr) {
        std::fprintf(stderr, "打不开输出文件 %s\n", g_opt.out.c_str());
        return 1;
    }
    {
        const uint32_t games = static_cast<uint32_t>(g_opt.games);
        const uint32_t obs_dim = obs::kObsDimV3;
        const uint32_t act_dim = brain::kPolicyActDim;
        std::fwrite(kMagic, 1, 4, g_out);
        std::fwrite(&kVersion, sizeof(kVersion), 1, g_out);
        std::fwrite(&obs_dim, sizeof(obs_dim), 1, g_out);
        std::fwrite(&act_dim, sizeof(act_dim), 1, g_out);
        std::fwrite(&games, sizeof(games), 1, g_out);
    }

    std::printf("策略自对弈: %d 局 / %d 线程 / 温度 %.2f%s\n", g_opt.games,
                g_opt.threads, g_opt.temperature,
                g_has_opp ? "  [蓝方=池中对手]" : "");

    const auto t0 = Clock::now();
    std::atomic<int> done{0};
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(g_opt.threads));
    for (int th = 0; th < g_opt.threads; ++th) {
        pool.emplace_back([th, &done] {
            SideTraj a, b;
            for (int g = th; g < g_opt.games; g += g_opt.threads) {
                play_game(g_opt.seed + static_cast<unsigned>(g) * 7919u, a, b);
                write_game(a, b);
                const int n = ++done;
                if (n % 200 == 0) {
                    std::printf("  进度 %d/%d\n", n, g_opt.games);
                    std::fflush(stdout);
                }
            }
        });
    }
    for (std::thread& t : pool) t.join();

    std::fclose(g_out);
    const double secs =
        std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("完成: %d 局, 用时 %.1fs (%.1f 局/秒)\n", g_opt.games, secs,
                g_opt.games / (secs > 0 ? secs : 1));
    return 0;
}

// tools/solve_ab.cpp —— 用 αβ + 置换表 + 分数上界截断求开局最优走法
//
// 为什么不走"完整状态 DP"：实测可达状态的增长速度是超线性的，turn 10 就到
// 13.3 亿，外推到 turn 20 是 10¹² 量级。并行化只解决常数因子，解决不了
// 数量级。所以改用只求**根节点值**的搜索。
//
// 三个手段，各自的理由：
//
// 1. **αβ + 置换表**。树 10^87，但跨回合的重复局面极多（置换因子 ~10^78），
//    置换表把树压成图。这是主力。
//
// 2. **分数上界截断**（本游戏特有，威力大）。turn t、分差 d 时，落后方能追回的
//    分数有硬上界：
//        剩余占点 ≤ (25−t)              （每回合最多 +1）
//        剩余击杀 ≤ 2·⌈(25−t)/2⌉        （fire_cd=2，两回合最多一次击杀）
//    若 d 超过这个上界，胜负已定 → **整棵子树直接返回定值**，不用展开。
//    一局才 20–25 回合，中后期 3 分领先基本就锁死了。
//
// 3. **终局只有三种取值**（红胜/平/红负）。值域只有 3 个元素，αβ 实际上退化成
//    "红能否强胜？否则红能否强平？"两次布尔搜索，剪枝比连续值域有效得多。
//
// 动作集：枚举全部合法动作，只剪掉"转向到当前朝向"这一种严格劣势动作
// （花掉额度却不改变任何状态；想结束阶段有收手）。见 tools/solve_game.cpp 的说明。

#include "brain/actions.h"
#include "obs/encode_v3.h"
#include "sim/rules.h"
#include "sim/view_mirror.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdlib>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kCells = sim::kBoardSize * sim::kBoardSize;
constexpr int kPoses = kCells * 4 * 3 * 4; // 2352
constexpr int kTurnN = 26;
constexpr int kDiffMax = 51;
constexpr int kDiffN = 2 * kDiffMax + 1;

// 值：-1 红负 / 0 平 / +1 红胜
constexpr int kLose = -1, kDraw = 0, kWin = 1;

long long g_nodes = 0;        // 展开的状态数
long long g_tt_hits = 0;      // 置换表命中
long long g_futile_cuts = 0;  // 分数上界截断次数
int g_max_turn = 25;

int encode_pose(const Sentry& s) {
    if (!(s.last_known_pos.x >= 0 && s.last_known_pos.x < sim::kBoardSize &&
          s.last_known_pos.y >= 0 && s.last_known_pos.y < sim::kBoardSize)) {
        return -1;
    }
    int fi = 0;
    for (int i = 0; i < 4; ++i) {
        if (s.last_known_facing == "NESW"[i]) fi = i;
    }
    const int cell = s.last_known_pos.y * sim::kBoardSize + s.last_known_pos.x;
    const int fcd = s.fire_cd < 0 ? 0 : (s.fire_cd > 2 ? 2 : s.fire_cd);
    const int scd = s.scan_cd < 0 ? 0 : (s.scan_cd > 3 ? 3 : s.scan_cd);
    return ((cell * 4) + fi) * 12 + (fcd * 4 + scd);
}

void apply_pose(int p, Sentry& s) {
    const int cd = p % 12;
    const int t = p / 12;
    const int cell = t / 4;
    s.last_known_pos = {cell % sim::kBoardSize, cell / sim::kBoardSize};
    s.last_known_facing = "NESW"[t % 4];
    s.fire_cd = cd / 4;
    s.scan_cd = cd % 4;
}

// ── 局面键 ──
// (红姿态, 蓝姿态, 回合, 行动方, 已用额度, 免费转向×2, 分差)
inline std::uint64_t make_key(int pr, int pb, int turn, int side, int ac, int freef,
                              int diff) {
    std::uint64_t v = static_cast<std::uint64_t>(pr) * kPoses + pb;
    v = v * kTurnN + static_cast<std::uint64_t>(turn);
    v = v * 2 + static_cast<std::uint64_t>(side);
    v = v * 4 + static_cast<std::uint64_t>(ac);
    v = v * 4 + static_cast<std::uint64_t>(freef);
    v = v * kDiffN + static_cast<std::uint64_t>(diff + kDiffMax);
    return v;
}

enum Flag : std::uint8_t { kExact = 0, kLower = 1, kUpper = 2 };

struct Entry {
    std::int8_t value = 0;
    std::uint8_t flag = kExact;
    std::uint8_t depth = 0;
    std::uint8_t best = 255; // 最佳动作下标
};

std::unordered_map<std::uint64_t, Entry> g_tt;

inline bool terminal_of(int turn, int diff) {
    if (turn < 20) return false;
    if (turn >= 25) return true;
    return diff != 0;
}

inline int terminal_value(int diff) { return diff > 0 ? kWin : (diff < 0 ? kLose : kDraw); }

// 落后方在剩余回合里最多还能追回的分数
inline int max_remaining(int turn) {
    const int left = kTurnN - 1 - turn; // 25 - turn
    if (left <= 0) return 0;
    return left + 2 * ((left + 1) / 2);
}

// 分数上界截断：分差已超出追回上界 → 胜负已定
inline bool decided_by_score(int turn, int diff, int& out) {
    const int mr = max_remaining(turn);
    if (diff > mr) { out = kWin; return true; }
    if (-diff > mr) { out = kLose; return true; }
    return false;
}

struct Pos2 {
    sim::State world;
};

// 动作清单：全部合法动作，只剪"转向到当前朝向"
struct Move {
    int action;
    char arg;
};

void gen_moves(const sim::State& s, char side, std::vector<Move>& out) {
    out.clear();
    const char f = s.sentry_for(side).last_known_facing;
    out.push_back({sim::kMove, 0});
    for (char c : {'N', 'E', 'S', 'W'}) {
        if (c != f) out.push_back({sim::kTurn, c});
    }
    out.push_back({sim::kFire, 0});
    out.push_back({sim::kScan, 0});
}

// ── αβ 搜索 ──
// depth 用剩余回合数近似（只用于置换表的深度校验）
int search(const sim::State& s0, char side, int ac, bool free_r, bool free_b, int diff,
           int alpha, int beta, int depth) {
    const int turn = s0.turn;

    if (terminal_of(turn, diff)) return terminal_value(diff);
    {
        int v;
        if (decided_by_score(turn, diff, v)) { ++g_futile_cuts; return v; }
    }

    const int pr = encode_pose(s0.red), pb = encode_pose(s0.blue);
    if (pr < 0 || pb < 0) return kDraw;
    const int side_idx = (side == 'R') ? 0 : 1;
    const int freef = (free_r ? 2 : 0) | (free_b ? 1 : 0);
    const std::uint64_t key = make_key(pr, pb, turn, side_idx, ac, freef, diff);

    // 置换表查表
    auto it = g_tt.find(key);
    if (it != g_tt.end()) {
        const Entry& e = it->second;
        if (e.depth >= depth) {
            if (e.flag == kExact) { ++g_tt_hits; return e.value; }
            if (e.flag == kLower) { ++g_tt_hits; if (e.value >= beta) return e.value; if (e.value > alpha) alpha = e.value; }
            if (e.flag == kUpper) { ++g_tt_hits; if (e.value <= alpha) return e.value; if (e.value < beta) beta = e.value; }
        }
    }

    ++g_nodes;
    if (g_nodes % 20000000 == 0) {
        std::fprintf(stderr, "[search] nodes=%lld tt=%zu\n", g_nodes, g_tt.size());
        // 解码抽样：看新键都落在哪些字段上
        int hist_turn[26] = {}, hist_diff_out = 0, hist_ac[4] = {}, hist_ff[4] = {};
        for (const auto& kv : g_tt) {
            std::uint64_t t = kv.first;
            const int dd = static_cast<int>(t % kDiffN) - kDiffMax; t /= kDiffN;
            const int ff = static_cast<int>(t % 4); t /= 4;
            const int acx = static_cast<int>(t % 4); t /= 4;
            const int sidx = static_cast<int>(t % 2); t /= 2;
            const int tn = static_cast<int>(t % kTurnN);
            if (tn < 26) ++hist_turn[tn];
            if (dd < -kDiffMax || dd > kDiffMax) ++hist_diff_out;
            ++hist_ac[acx & 3]; ++hist_ff[ff & 3];
        }
        std::fprintf(stderr, "  turn 分布: ");
        for (int i = 0; i < 26; ++i) if (hist_turn[i]) std::fprintf(stderr, "%d:%d ", i, hist_turn[i]);
        std::fprintf(stderr, "\n  ac: %d/%d/%d/%d  ff: %d/%d/%d/%d  diff越界: %d\n",
                     hist_ac[0], hist_ac[1], hist_ac[2], hist_ac[3],
                     hist_ff[0], hist_ff[1], hist_ff[2], hist_ff[3], hist_diff_out);
        std::fflush(stderr);
    }

    const int alpha0 = alpha, beta0 = beta;
    int best = (side == 'R') ? kLose : kWin; // 红方取最大，蓝方取最小
    int best_move = 255;

    // next_side 必须显式传入：**同一阶段内的动作由同一方继续**（一方在 act()
    // 里最多连续做 3 个动作），只有"收手"才把行动方交给对方。
    // 早先这里一律翻转，阶段结构被整个破坏，一个回合都搜不完。
    auto evaluate_child = [&](const sim::State& ns, char next_side, int nac, bool nfr,
                              bool nfb, int nv, int mi) {
        const int v = search(ns, next_side, nac, nfr, nfb, nv, alpha, beta, depth - 1);
        if (side == 'R') {
            if (v > best) { best = v; best_move = mi; }
            if (best > alpha) alpha = best;
        } else {
            if (v < best) { best = v; best_move = mi; }
            if (best < beta) beta = best;
        }
    };

    // 收手（永远可用）
    {
        sim::State es = s0;
        const int before = es.sentry_for(side).score;
        sim::end_side_turn(es, side);
        const int ediff =
            diff + (es.sentry_for(side).score - before) * (side == 'R' ? 1 : -1);
        if (side == 'R') {
            evaluate_child(es, (side == 'R') ? 'B' : 'R', 0, free_r, free_b, ediff, 254);
        } else {
            sim::State rs = es;
            if (turn + 1 < kTurnN) {
                sim::end_round(rs);
                evaluate_child(rs, 'R', 0, free_r, free_b, ediff, 254);
            } else {
                const int v = terminal_value(ediff);
                if (v < best) { best = v; best_move = 254; }
                if (best < beta) beta = best;
            }
        }
    }

    if (ac < brain::kMaxActionsPerTurn) {
        std::vector<Move> mv;
        gen_moves(s0, side, mv);
        for (std::size_t mi = 0; mi < mv.size(); ++mi) {
            sim::State ns = s0;
            int used = ac;
            bool ft = (side == 'R') ? free_r : free_b;
            bool hit = false;
            const brain::Cand wc = brain::local_to_world({mv[mi].action, mv[mi].arg}, side);
            if (!brain::apply_step(ns, side, wc, used, ft, &hit)) continue;
            const int nv = diff + (hit ? (side == 'R' ? 2 : -2) : 0);
            bool nfr = (side == 'R') ? ft : free_r;
            bool nfb = (side == 'B') ? ft : free_b;
            if (hit) {
                if (side == 'R') nfb = true; else nfr = true;
            }
            evaluate_child(ns, side, used, nfr, nfb, nv, static_cast<int>(mi));
            if (side == 'R' ? (alpha >= beta) : (beta <= alpha)) break;
        }
    }

    Entry e;
    e.value = static_cast<std::int8_t>(best);
    e.flag = (best <= alpha0) ? kUpper : (best >= beta0 ? kLower : kExact);
    e.depth = static_cast<std::uint8_t>(depth < 255 ? depth : 255);
    e.best = static_cast<std::uint8_t>(best_move);
    g_tt[key] = e;
    return best;
}

} // namespace

// ═══════════════════ teacher 模式：求解器自对弈，产出蒸馏数据 ═══════════════════
//
// 完整求解后，置换表已覆盖绝大部分可达空间。此时让求解器自己下棋：
// 每个决策点枚举全部候选（收手 + gen_moves），逐个查表取后继值，
// 红 max / 蓝 min 选出最优手——这就是"全知教师"。
//
// 学生侧的观测是**部分信息**的（ViewMirror + obs v3），教师标签是全知最优。
// 学生要学会的正是：用看得见的东西 + 历史，逼近全知者会做的选择。
//
// ε 探索只用于**执行**（走出多样局面），标签始终写教师最优——BC 需要的是
// 覆盖广且标签正确的数据，不是模仿次优动作。
//
// 数据格式沿用 SDPP（reward 恒 0，训练侧忽略）：每局红蓝各一条序列，
// 每条以真实终局结束。tools/train_bc.py 是消费端。

namespace teacher {

struct Seq {
    std::vector<float> obs;
    std::vector<std::uint8_t> action;
};

// 查一个后继的值：优先直读置换表（命中则 O(1)），缺失则用全窗口搜一遍
// （搜完会入库，下次即命中）。
int child_value(const sim::State& ns, char next_side, int nac, bool nfr, bool nfb,
                int nv, int turn_next) {
    const int pr = encode_pose(ns.red), pb = encode_pose(ns.blue);
    if (pr < 0 || pb < 0) return (next_side == 'R') ? kLose : kWin;
    const int side_idx = (next_side == 'R') ? 0 : 1;
    const int freef = (nfr ? 2 : 0) | (nfb ? 1 : 0);
    // make_key 内部已经 +kDiffMax，这里传原始 diff——上一版多加了一次，
    // 键永远对不上，快速查表路径形同虚设，每个后继都退化成全窗搜索。
    const std::uint64_t key =
        make_key(pr, pb, turn_next, side_idx, nac, freef, nv);
    const auto it = g_tt.find(key);
    if (it != g_tt.end()) return it->second.value;
    return search(ns, next_side, nac, nfr, nfb, nv, kLose, kWin, 0);
}

// 教师在当前局面选动作，返回 (动作下标 0..7, 是否收手)
int teacher_action(const sim::State& world, char side, int ac, bool fr, bool fb,
                   int diff, std::mt19937& rng, double eps) {
    std::vector<Move> mv;
    gen_moves(world, side, mv);

    // ε 探索：在合法动作里均匀抽一个（收手也算）
    if (std::uniform_real_distribution<double>(0, 1)(rng) < eps) {
        const int n = static_cast<int>(mv.size()) + 1;
        const int k = std::uniform_int_distribution<int>(0, n - 1)(rng);
        return k == static_cast<int>(mv.size()) ? 7 : brain::cand_to_index({mv[k].action, mv[k].arg});
    }

    int best_idx = 7;
    int best_val = (side == 'R') ? 2 : -2;
    auto consider = [&](const sim::State& ns, char next_side, int nac, bool nfr2,
                        bool nfb2, int nv, int turn_next, int idx) {
        const int v = child_value(ns, next_side, nac, nfr2, nfb2, nv, turn_next);
        if (side == 'R' ? (v > best_val) : (v < best_val)) {
            best_val = v;
            best_idx = idx;
        }
    };

    // 收手
    {
        sim::State es = world;
        const int before = es.sentry_for(side).score;
        sim::end_side_turn(es, side);
        const int ediff = diff + (es.sentry_for(side).score - before) * (side == 'R' ? 1 : -1);
        if (side == 'R') {
            consider(es, 'B', 0, fr, fb, ediff, es.turn, 7);
        } else if (world.turn + 1 < kTurnN) {
            sim::State rs = es;
            sim::end_round(rs);
            consider(rs, 'R', 0, fr, fb, ediff, rs.turn, 7);
        } else {
            const int v = terminal_value(ediff);
            if (side == 'R' ? (v > best_val) : (v < best_val)) best_idx = 7;
        }
    }
    for (std::size_t i = 0; i < mv.size(); ++i) {
        sim::State ns = world;
        int used = ac;
        bool ft = (side == 'R') ? fr : fb;
        bool hit = false;
        const brain::Cand wc = brain::local_to_world({mv[i].action, mv[i].arg}, side);
        if (!brain::apply_step(ns, side, wc, used, ft, &hit)) continue;
        const int nv = diff + (hit ? (side == 'R' ? 2 : -2) : 0);
        bool nfr = (side == 'R') ? ft : fr;
        bool nfb = (side == 'B') ? ft : fb;
        if (hit) {
            if (side == 'R') nfb = true; else nfr = true;
        }
        consider(ns, side, used, nfr, nfb, nv, ns.turn, brain::cand_to_index({mv[i].action, mv[i].arg}));
    }
    return best_idx;
}

// 跑一局教师自对弈，产出两条（部分信息观测，教师标签）序列
void play_teacher_game(std::mt19937& rng, double eps, int from_turn,
                       Seq& seq_r, Seq& seq_b) {
    sim::State world = sim::make_initial_state();
    world.turn = from_turn;  // 【关键】与主求解同一起点。上一版漏了这行，
                             // 教师从 turn 0 开始下，第一步查询就在搜整棵
                             // 完整博弈——15 分钟一局都跑不完的真正原因。
    sim::ViewMirror mirror;
    mirror.reset();
    seq_r.obs.clear(); seq_r.action.clear();
    seq_b.obs.clear(); seq_b.action.clear();

    bool respawn_pending[2] = {true, true};
    // 双方"当前是否还有免费转向"——教师查表的键里包含这两个旗标，
    // 必须按引擎语义维护（apply_step 会消耗它），否则键错、命中率崩。
    bool side_free[2] = {true, true};
    Seq* seq[2] = {&seq_r, &seq_b};

    const int max_rounds = 40;
    for (int round = 0; round < max_rounds; ++round) {
        for (int si = 0; si < 2; ++si) {
            const char side = (si == 0) ? 'R' : 'B';
            bool ft = respawn_pending[si];
            respawn_pending[si] = false;
            side_free[si] = ft;
            int used = 0;

            for (int k = 0; k < brain::kMaxActionsPerTurn; ++k) {
                // 学生观测：部分信息
                const sim::State view = mirror.local_view(world, side);
                float obs[obs::kObsDimV3];
                obs::encode_v3(view, mirror.enemy_visible(side), used, ft, obs);

                const int diff = world.red.score - world.blue.score;
                int a = teacher_action(world, side, used,
                                       side_free[0], side_free[1], diff, rng, eps);

                Seq& out = *seq[si];
                out.obs.insert(out.obs.end(), obs, obs + obs::kObsDimV3);
                out.action.push_back(static_cast<std::uint8_t>(a));
                if (getenv("TEACHER_TRACE") != nullptr) {
                    std::fprintf(stderr, "[step] g=0 side=%c k=%d used=%d turn=%d "
                                 "nodes=%lld tt=%zu\n", side, k, used, world.turn,
                                 g_nodes, g_tt.size());
                }
                // reward 恒 0：BC 不用（SDPP 布局保留字段）
                // 执行动作（ε 时执行的可能不是教师最优——标签仍是最优）
                if (a == 7) break; // 收手
                const brain::Cand wc = brain::local_to_world(brain::index_to_cand(a), side);
                bool hit = false;
                if (!brain::apply_step(world, side, wc, used, ft, &hit)) continue;
                side_free[si] = ft;
                if (hit) {
                    respawn_pending[1 - si] = true;
                    side_free[1 - si] = true;
                }
                mirror.after_action(world, side, wc.action);
                if (used >= brain::kMaxActionsPerTurn) break;
            }

            sim::end_side_turn(world, side);
            if (side == 'B') sim::end_round(world);
            if (brain::is_terminal(world)) return;
        }
    }
}

void run(const std::string& prefix, int games, double eps, int from_turn) {
    std::FILE* f = std::fopen(prefix.c_str(), "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "teacher: 打不开输出 %s\n", prefix.c_str());
        std::exit(1);
    }
    const char magic[4] = {'S', 'D', 'P', 'P'};
    const std::uint32_t version = 1, obs_dim = obs::kObsDimV3, act_dim = static_cast<std::uint32_t>(brain::kActionDim);
    const std::uint32_t g32 = static_cast<std::uint32_t>(games);
    std::fwrite(magic, 1, 4, f);
    std::fwrite(&version, sizeof(version), 1, f);
    std::fwrite(&obs_dim, sizeof(obs_dim), 1, f);
    std::fwrite(&act_dim, sizeof(act_dim), 1, f);
    std::fwrite(&g32, sizeof(g32), 1, f);

    std::mt19937 rng(20260916u);
    const auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < games; ++g) {
        Seq r, b;
        play_teacher_game(rng, eps, from_turn, r, b);
        for (const Seq* sq : {&r, &b}) {
            const std::uint32_t n = static_cast<std::uint32_t>(sq->action.size());
            std::fwrite(&n, sizeof(n), 1, f);
            for (std::size_t i = 0; i < sq->action.size(); ++i) {
                std::fwrite(&sq->obs[i * obs::kObsDimV3], sizeof(float), obs::kObsDimV3, f);
                const float zero = 0.0f;
                std::fwrite(&sq->action[i], 1, 1, f);
                std::fwrite(&zero, sizeof(zero), 1, f);
            }
        }
        if ((g + 1) % 10 == 0 || games <= 20) {
            std::printf("  teacher %d/%d 局  (TT %zu, 节点 %lld)\n",
                        g + 1, games, g_tt.size(), g_nodes);
            std::fflush(stdout);
        }
    }
    std::fclose(f);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  teacher 完成: %d 局, %.1f 秒\n", games, secs);
}

} // namespace teacher

int main(int argc, char** argv) {
    int from_turn = 0;
    std::size_t tt_reserve = 1u << 24;
    const char* teacher_prefix = nullptr;
    int teacher_games = 4000;
    double teacher_eps = 0.3;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--from-turn") == 0 && i + 1 < argc) {
            from_turn = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--tt") == 0 && i + 1 < argc) {
            tt_reserve = static_cast<std::size_t>(std::atoll(argv[++i]));
        } else if (std::strcmp(argv[i], "--teacher") == 0 && i + 1 < argc) {
            teacher_prefix = argv[++i];
        } else if (std::strcmp(argv[i], "--games") == 0 && i + 1 < argc) {
            teacher_games = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--eps") == 0 && i + 1 < argc) {
            teacher_eps = std::atof(argv[++i]);
        }
    }
    g_max_turn = 25;

    std::printf("αβ + 置换表 + 分数上界截断\n");
    std::printf("起始回合 %d   置换表预留 %zu 条\n", from_turn, tt_reserve);
    std::fflush(stdout);

    g_tt.reserve(tt_reserve);

    sim::State init = sim::make_initial_state();
    init.turn = from_turn;
    // 从 turn=N 开始时比分未知 —— 这里取 0:0。
    // 所以 --from-turn 测的是"从该回合 0:0 开始"的子博弈，用于量搜索规模。
    const int v = search(init, 'R', 0, true, true, 0, kLose, kWin, kTurnN - from_turn);

    // —— teacher 模式：求解完成后直接产出蒸馏数据 ——
    if (teacher_prefix != nullptr) {
        std::printf("\nteacher 自对弈: %d 局 / ε=%.2f → %s\n", teacher_games,
                    teacher_eps, teacher_prefix);
        std::fflush(stdout);
        teacher::run(teacher_prefix, teacher_games, teacher_eps, from_turn);
    }

    const char* name[] = {"红负", "平", "红胜"};
    std::printf("\n════════ 结果 ════════\n");
    std::printf("  值            %s\n", name[v + 1]);

    // 打印根节点的最优首手。一个荒谬的首手（原地不动、对着墙开火）能立刻
    // 证伪整个搜索；合理的话只是必要不充分条件。
    {
        const int pr = encode_pose(init.red), pb = encode_pose(init.blue);
        const std::uint64_t rk = make_key(pr, pb, init.turn, 0, 0, 3, 0);
        auto it = g_tt.find(rk);
        if (it != g_tt.end() && it->second.best != 255) {
            const int mi = it->second.best;
            std::vector<Move> mv;
            gen_moves(init, 'R', mv);
            if (mi == 254) {
                std::printf("  最优首手      收手（结束本阶段）\n");
            } else if (mi < static_cast<int>(mv.size())) {
                const char* an[] = {"move", "turn", "fire", "scan"};
                std::printf("  最优首手      %s%s%c   红方位置 (%d,%d) 朝 %c\n",
                            an[mv[mi].action],
                            mv[mi].action == sim::kTurn ? " " : "",
                            mv[mi].action == sim::kTurn ? mv[mi].arg : ' ',
                            init.red.last_known_pos.x, init.red.last_known_pos.y,
                            init.red.last_known_facing);
            }
        } else {
            std::printf("  最优首手      （置换表里没找到根节点）\n");
        }
    }
    std::printf("  展开状态      %lld\n", g_nodes);
    std::printf("  置换表命中    %lld\n", g_tt_hits);
    std::printf("  上界截断      %lld\n", g_futile_cuts);
    std::printf("  置换表条目    %zu\n", g_tt.size());
    return 0;
}

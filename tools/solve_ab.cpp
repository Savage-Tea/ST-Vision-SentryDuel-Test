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
#include "sim/rules.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

int main(int argc, char** argv) {
    int from_turn = 0;
    std::size_t tt_reserve = 1u << 24;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--from-turn") == 0 && i + 1 < argc) {
            from_turn = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--tt") == 0 && i + 1 < argc) {
            tt_reserve = static_cast<std::size_t>(std::atoll(argv[++i]));
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

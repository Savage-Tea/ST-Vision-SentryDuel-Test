#include "brain/ab_search.h"

#include <chrono>
#include <cstdint>
#include "brain/value_net.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace brain {
namespace ab {

using Clock = std::chrono::steady_clock;

// ── 常量 ──
constexpr double kWin = 1e6;
constexpr double kInf = 2e6;

// ── 上下文 ──
struct Ctx {
    const Weights* w;
    const sim::Belief* root_belief;
    bool acts_first;
    bool enemy_visible;
    const Deadline* deadline;
    SearchStats* stats;
    long long nodes;
    long long budget;
    bool aborted;

    bool out_of_time() const {
        return aborted || (nodes & 1023) == 0 &&
               Clock::now() > *deadline;
    }
};

// ── 置换表 ──
struct TTE {
    std::uint64_t key;
    double score;
    int depth;
    std::uint8_t flag; // 0=exact, 1=lower, 2=upper
};
// 64k 槽——.so 里分配过多内存在某些环境下会导致段错误
constexpr std::size_t kTTSize = 1 << 16; // 64k 槽
std::vector<TTE> g_tt(kTTSize, {0, 0.0, 0, 0});
std::vector<bool> g_tt_used(kTTSize, false);

inline std::size_t tt_idx(std::uint64_t key) { return key % kTTSize; }


// ── 状态键 ──
std::uint64_t make_key(const sim::State& s, int side, int ac, bool fr, bool fb) {
    const auto enc = [](const Sentry& e) -> std::uint64_t {
        const int cell = e.last_known_pos.y * 7 + e.last_known_pos.x;
        int fi = 0;
        for (int i = 0; i < 4; ++i) { if ("NESW"[i] == e.last_known_facing) fi = i; }
        const int fc = e.fire_cd < 0 ? 0 : std::min(e.fire_cd, 2);
        const int sc = e.scan_cd < 0 ? 0 : std::min(e.scan_cd, 3);
        return static_cast<std::uint64_t>(((cell * 4 + fi) * 12) + (fc * 4 + sc));
    };
    std::uint64_t v = enc(s.red);
    v = v * 2352 + enc(s.blue);
    v = v * 26 + static_cast<std::uint64_t>(s.turn);
    v = v * 2 + static_cast<std::uint64_t>(side);
    v = v * 4 + static_cast<std::uint64_t>(ac);
    v = v * 4 + static_cast<std::uint64_t>((fr ? 2 : 0) | (fb ? 1 : 0));
    return v;
}

// ── 核心递归 ──
double ab_rec(const sim::State& s, int side, int ac, bool fr, bool fb,
              bool enemy_vis, int depth, double alpha, double beta, Ctx& ctx) {
    ++ctx.nodes;
    if (ctx.nodes >= ctx.budget) { ctx.aborted = true; return 0.0; }
    if (Clock::now() > *ctx.deadline) { ctx.aborted = true; return 0.0; }

    // 终局
    if (brain::is_terminal(s)) {
        const int d = s.red.score - s.blue.score;
        return d > 0 ? kWin : (d < 0 ? -kWin : 0.0);
    }

    // 叶节点
    if (depth <= 0) {
        return brain::evaluate(s, *ctx.w, *ctx.root_belief, ctx.acts_first);
    }

    // 置换表
    // TT 暂禁（调试）
    // const auto key = make_key(s, side, ac, fr, fb);
    // const auto idx = tt_idx(key);

    const char side_c = side == 0 ? 'R' : 'B';
    const bool my_free = (side == 0) ? fr : fb;

    std::vector<brain::Cand> cands;
    brain::collect_candidates(s, side_c, my_free, enemy_vis, nullptr, nullptr, cands);

    double best;
    if (side == 0) best = -kInf; else best = kInf;

    // ── 收手分支（永远可用）──
    {
        sim::State es = s;
        const int br = es.red.score, bb = es.blue.score;
        sim::end_side_turn(es, side_c);

        sim::State er = es;
        if (side == 1) sim::end_round(er); // 蓝方结束 = 整回合结束

        if (brain::is_terminal(er)) {
            const int d = er.red.score - er.blue.score;
            best = d > 0 ? kWin : (d < 0 ? -kWin : 0.0);
        } else {
            const int ns = side == 0 ? 1 : 0;
            const int nd = side == 1 ? depth - 1 : depth;
            const bool nfr = (side == 0) ? fr : fr; // 红方旗标不变
            const bool nfb = (side == 1) ? fb : fb; // 蓝方旗标不变
            best = ab_rec(er, side == 0 ? 'B' : 'R', 0, nfr, nfb,
                          enemy_vis, nd, alpha, beta, ctx);
        }
        if (side == 0) { if (best > alpha) alpha = best; }
        else { if (best < beta) beta = best; }
    }

    // ── 行动分支 ──
    if (ac < 3 && alpha < beta) {
        for (const auto& c : cands) {
            sim::State ns = s;
            int nud = ac;
            bool nft = my_free;
            bool hit = false;
            if (!brain::apply_step(ns, side_c, c, nud, nft, &hit)) continue;

            const double v = ab_rec(ns, side, nud,
                                    (side == 0) ? nft : fr,
                                    (side == 1) ? nft : fb,
                                    enemy_vis, depth - 1, alpha, beta, ctx);

            if (side == 0) {
                if (v > best) best = v;
                if (best > alpha) alpha = best;
            } else {
                if (v < best) best = v;
                if (best < beta) beta = best;
            }
            if (alpha >= beta) break;
        }
    }

    // g_tt[idx] = {key, best, depth, 0};
    // g_tt_used[idx] = true;
    return best;
}

// ── 搜索入口：枚举我方第一手，每手调 ab_rec 得分，取最大 ──
} // namespace ab

Plan ab_search_turn(const TurnInput& in, const Weights& w,
                    std::chrono::steady_clock::time_point deadline,
                    SearchStats* stats) {
    ab::Ctx ctx;
    ctx.w = &w;
    ctx.root_belief = &in.belief;
    ctx.acts_first = in.acts_first_world;
    ctx.enemy_visible = in.enemy_visible;
    ctx.nodes = 0;
    ctx.budget = 500000;
    ctx.aborted = false;
    ctx.deadline = &deadline;
    ctx.stats = stats;

    if (stats) { stats->our_nodes = 0; stats->opp_nodes = 0; stats->refined = 0; }

    // 构建起始状态
    sim::State root = in.state;

    const char my_color = in.acts_first_world ? 'R' : 'B';
    const bool my_free = in.free_turn_available;

    // 我方第一手候选
    std::vector<brain::Cand> cands;
    brain::collect_candidates(root, my_color, my_free, in.enemy_visible,
                              nullptr, &in.belief, cands);

    if (cands.empty()) {
        if (stats) *stats = *ctx.stats;
        Plan p; p.valid = false;
        return p;
    }

    struct RootChild {
        brain::Cand move;
        double score;
    };
    std::vector<RootChild> children;

    double alpha = -ab::kInf;

    for (const auto& c : cands) {
        sim::State ns = root;
        int nud = in.used_by_now;
        bool nft = my_free;
        bool hit = false;
        if (!brain::apply_step(ns, my_color, c, nud, nft, &hit)) continue;

        // 我方行动阶段结束
        sim::State es = ns;
        const int br = es.red.score, bb = es.blue.score;
        sim::end_side_turn(es, my_color);
        const int my_gain = es.red.score - br;

        // 对手阶段 + end_round
        sim::State er = es;
        if (my_color == 'R') {
            // 我方是红 → 对手是蓝行动 → end_round
        }
        // 对手阶段后蓝方结束 → end_round（蓝色始终跟 end_round）
        // 上面在 ab_rec 内处理

        // 计算这手的值
        int ns_side = my_color == 'R' ? 1 : 0;
        int next_depth = 3 * 2 - 1; // 5 ply remaining (of ~6)

        sim::State er2 = es;
        if (my_color == 'B') sim::end_round(er2);

        // 我方是蓝 → 蓝行动后对手（红）行动
        // 我方是红 → 蓝行动后我方（红... 不对，蓝行动后轮到我方回合但我们已行动完
        // 在模拟中：我方阶段结束 → 对手阶段 → end_round → 我方下一阶段
        // 对搜索而言：es 后是对手阶段，er2（蓝方时）后是新回合

        double v;
        if (my_color == 'R') {
            // 我方红，阶段结束 → 蓝方阶段
            // end_round 会在蓝方阶段结束后调用
            v = ab::ab_rec(es, 'B', 0, in.free_turn_available, in.free_turn_available,
                           in.enemy_visible, next_depth, -ab::kInf, ab::kInf, ctx);
        } else {
            // 我方蓝，阶段结束 → end_round → 红方阶段
            sim::State er = es;
            sim::end_round(er);
            v = ab::ab_rec(er, 'R', 0, nft, in.free_turn_available,
                           in.enemy_visible, next_depth, -ab::kInf, ab::kInf, ctx);
        }

        children.push_back({c, v});
        if (v > alpha) alpha = v;
    }

    // 取最优
    if (children.empty()) {
        Plan p; p.valid = false;
        if (stats) *stats = *ctx.stats;
        return p;
    }

    const auto& best_child = *std::max_element(
        children.begin(), children.end(),
        [](const auto& a, const auto& b) { return a.score < b.score; });

    Plan p;
    p.valid = true;
    p.count = 1;
    p.actions[0] = best_child.move.action;
    p.args[0] = best_child.move.arg;
    p.value = best_child.score;

    if (stats) {
        stats->our_nodes = ctx.nodes;
        stats->refined = static_cast<long long>(children.size());
    }
    return p;
}

} // namespace brain

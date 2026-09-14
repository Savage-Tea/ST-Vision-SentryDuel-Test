#include "brain/search.h"

#include <algorithm>
#include <vector>

namespace brain {
namespace {

struct Cand {
    int action;
    char arg;
};

struct Leaf {
    double value = 0.0;
    Plan plan;
    sim::State state; // 我方阶段结束（已计占点分）后的状态
};

// 生成某方的候选行动。剪枝掉的都是"引擎允许但纯属浪费"的分支：
//   · 转向到当前朝向且不免费 —— 成功、消耗额度、状态不变
//   · 火力未就绪 / 雷达未就绪 —— 必然失败
//   · 看不见对手时开火 —— 盲射要付整整一回合的火力空窗，阶段① 先不开这个口子
void collect_candidates(const sim::State& s, char side, bool free_turn,
                        bool can_see_enemy, std::vector<Cand>& out,
                        const TurnInput* bans = nullptr) {
    out.clear();
    const Sentry& me = s.sentry_for(side);
    const bool at_spawn = sim::is_at_spawn(s, side);

    auto push = [&](int action, char arg) {
        if (bans && bans->is_banned(action, arg)) return;
        out.push_back({action, arg});
    };

    push(sim::kMove, 0);

    for (char f : {'N', 'E', 'S', 'W'}) {
        if (f == me.last_known_facing && !(free_turn && at_spawn)) continue;
        push(sim::kTurn, f);
    }
    if (me.fire_cd == 0 && can_see_enemy) push(sim::kFire, 0);
    if (me.scan_cd == 0) push(sim::kScan, 0);
}

// 应用一个行动，并复刻引擎 Match::do_action 的额度与免费转向语义。
// 失败返回 false（不消耗额度，也不会改变 s）。
bool step(sim::State& s, char side, const Cand& cand, int& used, bool& free_turn) {
    if (used >= kMaxActions) return false;

    // 引擎 do_action 顶部：一旦不在出生点，免费资格立即失效
    const bool was_at_spawn = sim::is_at_spawn(s, side);
    if (free_turn && !was_at_spawn) free_turn = false;

    const sim::Outcome o = sim::apply_action(s, side, cand.action, cand.arg);
    if (!o.success) return false;

    // TURN 在出生点且免费资格仍在 → 不占额度（位置不变，故 was_at_spawn 即当前状态）
    if (cand.action == sim::kTurn && free_turn && was_at_spawn) {
        free_turn = false;
    } else {
        ++used;
    }
    return true;
}

// 我方第一层：枚举 0..3 个行动，每个深度都作为候选方案记录下来
void dfs_our(const sim::State& s, int used, bool free_turn, bool enemy_visible, int depth,
             Plan& cur, std::vector<Leaf>& out, const Deadline& dl, SearchStats& st,
             const Weights& w, const TurnInput* bans) {
    {
        // 我方行动阶段结束 → 本回合占点分入账
        sim::State after = s;
        sim::end_side_turn(after, 'R');
        Leaf leaf;
        leaf.plan = cur;
        leaf.plan.count = depth;
        // 行动是有限资源：同等局面下优先选消耗更少的方案（仅用于打破平局）
        leaf.value = evaluate(after, w) - w.w_waste * static_cast<double>(used);
        leaf.state = after;
        leaf.plan.valid = true;
        out.push_back(leaf);
        ++st.our_nodes;
    }

    if (depth >= kMaxActions || used >= kMaxActions) return;
    if (st.time_exhausted || out_of_time(dl)) {
        st.time_exhausted = true;
        return;
    }

    std::vector<Cand> cands;
    collect_candidates(s, 'R', free_turn, enemy_visible, cands, bans);
    for (const Cand& c : cands) {
        sim::State ns = s;
        int nused = used;
        bool nfree = free_turn;
        if (!step(ns, 'R', c, nused, nfree)) continue;
        cur.actions[depth] = c.action;
        cur.args[depth] = c.arg;
        dfs_our(ns, nused, nfree, enemy_visible, depth + 1, cur, out, dl, st, w, bans);
    }
}

// 对手第二层：枚举 0..3 个行动，取让我们最不利的（minimize）
void dfs_opp(const sim::State& s, int used, bool free_turn, bool sees_us, int depth,
             double& worst, const Deadline& dl, SearchStats& st, const Weights& w) {
    {
        sim::State after = s;
        sim::end_side_turn(after, 'B');
        worst = std::min(worst, evaluate(after, w));
        ++st.opp_nodes;
    }

    if (depth >= kMaxActions || used >= kMaxActions) return;
    if (st.time_exhausted || out_of_time(dl)) {
        st.time_exhausted = true;
        return;
    }

    std::vector<Cand> cands;
    collect_candidates(s, 'B', free_turn, sees_us, cands);
    for (const Cand& c : cands) {
        sim::State ns = s;
        int nused = used;
        bool nfree = free_turn;
        if (!step(ns, 'B', c, nused, nfree)) continue;
        dfs_opp(ns, nused, nfree, sees_us, depth + 1, worst, dl, st, w);
    }
}

// 对手在我方叶子状态上的最优回应值
double opponent_best(const sim::State& s, const Weights& w, const Deadline& dl,
                     SearchStats& st) {
    sim::State base = s;
    // 对手的 CD 在观测里恒为 -1（引擎不暴露），保守假设其随时可开火
    base.blue.fire_cd = kAssumedEnemyFireCd;
    base.blue.scan_cd = 0;

    // 对手能否看见我们：这是确定可算的，决定了他能否开火
    const bool opp_sees_us = sim::can_see(base.blue, base.red.last_known_pos, base.obstacles);

    double worst = evaluate(base, w);
    const bool free_turn = sim::is_at_spawn(base, 'B');
    dfs_opp(base, 0, free_turn, opp_sees_us, 0, worst, dl, st, w);
    return worst;
}

} // namespace

Plan search_turn(const TurnInput& in, const Weights& w, const Deadline& deadline,
                 SearchStats* stats) {
    SearchStats st;
    std::vector<Leaf> leaves;
    Plan cur;
    dfs_our(in.state, in.used_by_now, in.free_turn_available, in.enemy_visible, 0, cur,
            leaves, deadline, st, w, &in);

    if (leaves.empty()) {
        if (stats) *stats = st;
        return Plan{};
    }

    // 先按"对手不作为"的乐观值排序，只对最有希望的前 K 个做对手回应细化。
    // 这样即使时间紧张，被细化的也是真正有价值的候选。
    std::sort(leaves.begin(), leaves.end(),
              [](const Leaf& a, const Leaf& b) { return a.value > b.value; });

    const int top_k = std::min<int>(static_cast<int>(leaves.size()), kRefineTopK);
    int refined = 0;
    for (int i = 0; i < top_k; ++i) {
        if (out_of_time(deadline)) {
            st.time_exhausted = true;
            break;
        }
        leaves[i].value = opponent_best(leaves[i].state, w, deadline, st);
        ++refined;
    }
    st.refined = refined;

    // 细化值（对手最优回应下的最小值）与未细化值（乐观值）不可直接比较，
    // 所以只要细化成功过，就只在已细化的候选里选。
    int best_index = -1;
    if (refined > 0) {
        for (int i = 0; i < refined; ++i) {
            if (best_index < 0 || leaves[i].value > leaves[best_index].value) best_index = i;
        }
    } else {
        best_index = 0; // 已按乐观值排序
    }

    Plan best = leaves[best_index].plan;
    best.value = leaves[best_index].value;
    best.valid = true;
    if (best.count > kMaxActions) best.count = kMaxActions;
    if (stats) *stats = st;
    return best;
}

} // namespace brain

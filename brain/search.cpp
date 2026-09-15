#include "brain/search.h"

#include <algorithm>
#include <vector>

namespace brain {
namespace {

struct Leaf {
    double value = 0.0;
    Plan plan;
    sim::State state;  // 我方阶段结束（已计占点分）后的状态
    sim::Belief belief; // 该叶子上（经我方视野证伪后）的敌方可能位置集合
};

// 候选生成与额度语义已抽到 brain/actions.*，与 MCTS 共用一份实现

// 我方第一层：枚举 0..3 个行动，每个深度都作为候选方案记录下来
void dfs_our(const sim::State& s, const sim::Belief& b, int used, bool free_turn,
             bool enemy_visible, int depth, Plan& cur, std::vector<Leaf>& out,
             const Deadline& dl, SearchStats& st, const Weights& w, bool acts_first,
             const TurnInput* bans) {
    {
        // 我方行动阶段结束 → 本回合占点分入账
        sim::State after = s;
        sim::end_side_turn(after, 'R');
        Leaf leaf;
        leaf.plan = cur;
        leaf.plan.count = depth;
        // 行动是有限资源：同等局面下优先选消耗更少的方案（仅用于打破平局）
        leaf.value = evaluate(after, w, b, acts_first) - w.w_waste * static_cast<double>(used);
        leaf.state = after;
        leaf.belief = b;
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
    collect_candidates(s, 'R', free_turn, enemy_visible,
                       bans != nullptr ? &bans->bans : nullptr, &b, cands);
    for (const Cand& c : cands) {
        sim::State ns = s;
        int nused = used;
        bool nfree = free_turn;
        if (!apply_step(ns, 'R', c, nused, nfree)) continue;

        // 我方这一步之后视野变了 → 用新视野对信念做证伪。
        // 于是"走到能看见更多地方的位置"本身就有价值，不需要额外写规则。
        sim::Belief nb = b;
        sim::prune_by_vision(nb, ns.red, ns.obstacles);
        sim::prune_impossible(nb, ns.red.last_known_pos, ns.obstacles);
        if (nb.empty()) nb = b; // 证伪后为空 = 模型不一致，保守回退

        cur.actions[depth] = c.action;
        cur.args[depth] = c.arg;
        dfs_our(ns, nb, nused, nfree, enemy_visible, depth + 1, cur, out, dl, st, w,
                acts_first, bans);
    }
}

// 对手第二层：枚举 0..3 个行动，取让我们最不利的（minimize）
void dfs_opp(const sim::State& s, const sim::Belief& b, int used, bool free_turn,
             bool sees_us, int depth, double& worst, const Deadline& dl,
             SearchStats& st, const Weights& w, bool acts_first) {
    {
        sim::State after = s;
        sim::end_side_turn(after, 'B');
        worst = std::min(worst, evaluate(after, w, b, acts_first));
        ++st.opp_nodes;
    }

    if (depth >= kMaxActions || used >= kMaxActions) return;
    if (st.time_exhausted || out_of_time(dl)) {
        st.time_exhausted = true;
        return;
    }

    std::vector<Cand> cands;
    collect_candidates(s, 'B', free_turn, sees_us, nullptr, nullptr, cands);
    for (const Cand& c : cands) {
        sim::State ns = s;
        int nused = used;
        bool nfree = free_turn;
        if (!apply_step(ns, 'B', c, nused, nfree)) continue;
        dfs_opp(ns, b, nused, nfree, sees_us, depth + 1, worst, dl, st, w, acts_first);
    }
}

// 对手在我方叶子状态上的最优回应值
double opponent_best(const sim::State& s, const sim::Belief& b, const Weights& w,
                     bool acts_first, int opp_def_until, const Deadline& dl,
                     SearchStats& st) {
    sim::State base = s;
    // 对手的 CD 在观测里恒为 -1（引擎不暴露），保守假设其随时可开火；
    // 唯一例外是被击中推断出的无力窗口——那是对手刚开火的确定性情报，
    // 窗口内保持状态里的 CD（=2，无法开火），让搜索敢压近。
    if (!(base.turn <= opp_def_until)) {
        base.blue.fire_cd = kAssumedEnemyFireCd;
    }
    base.blue.scan_cd = 0;

    // 对手能否看见我们：这是确定可算的，决定了他能否开火
    const bool opp_sees_us = sim::can_see(base.blue, base.red.last_known_pos, base.obstacles);

    double worst = evaluate(base, w, b, acts_first);
    const bool free_turn = sim::is_at_spawn(base, 'B');
    dfs_opp(base, b, 0, free_turn, opp_sees_us, 0, worst, dl, st, w, acts_first);
    return worst;
}

} // namespace

Plan search_turn(const TurnInput& in, const Weights& w, const Deadline& deadline,
                 SearchStats* stats) {
    SearchStats st;
    std::vector<Leaf> leaves;
    Plan cur;
    dfs_our(in.state, in.belief, in.used_by_now, in.free_turn_available, in.enemy_visible,
            0, cur, leaves, deadline, st, w, in.acts_first_world, &in);

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
        leaves[i].value =
            opponent_best(leaves[i].state, leaves[i].belief, w, in.acts_first_world,
                          in.opp_defenseless_until, deadline, st);
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

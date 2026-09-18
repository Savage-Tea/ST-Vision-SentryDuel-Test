#include "brain/search.h"

#include "brain/value_net.h"

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

// 叶评估：值蒸馏网络启用时用 V* 网络替换手工评估。
//
// 叶的相位语义（与求解器状态一一对应）：
//   after = "我方"阶段已结束（占点分已入账）→ 对手行动中 → 红方行动中
//   当且仅当我方是世界红方；反之红方行动中 = 我方。
// free 旗标在搜索里没有完整跟踪，用出生点近似（只在免费转向上有差，
// 且只影响免费转向这种小代价动作——对值的影响是二阶的）。
double leaf_eval(const sim::State& after, const sim::Belief& b, const Weights& w,
                 bool acts_first, bool use_vnet, double vscale, int used,
                 bool opp_to_move, bool my_free_left) {
    if (use_vnet && brain::value_net_available()) {
        float feats[brain::kValueObsDim];
        // v2 网络是信徒相对的（颜色无关）：局部帧直喂，无需 swap/取反。
        // 对手免费旗标用出生点近似（免费转向只影响小代价动作，二阶项）。
        const bool opp_free = sim::is_at_spawn(after, 'B');
        brain::value_features(after, b, /*me_to_move=*/!opp_to_move,
                              my_free_left, opp_free, feats);
        float v = 0.0f;
        brain::value_eval(feats, &v);
        return static_cast<double>(v) * vscale - w.w_waste * static_cast<double>(used);
    }
    return evaluate(after, w, b, acts_first) - w.w_waste * static_cast<double>(used);
}

// 我方第一层：枚举 0..3 个行动，每个深度都作为候选方案记录下来
void dfs_our(const sim::State& s, const sim::Belief& b, int used, bool free_turn,
             bool enemy_visible, int depth, Plan& cur, std::vector<Leaf>& out,
             const Deadline& dl, SearchStats& st, const Weights& w, bool acts_first,
             bool use_vnet, double vscale, const TurnInput* bans) {
    {
        // 我方行动阶段结束 → 本回合占点分入账
        sim::State after = s;
        sim::end_side_turn(after, 'R');
        Leaf leaf;
        leaf.plan = cur;
        leaf.plan.count = depth;
        // 行动是有限资源：同等局面下优先选消耗更少的方案（仅用于打破平局）
        leaf.value = leaf_eval(after, b, w, acts_first, use_vnet, vscale, used,
                               /*opp_to_move=*/true, free_turn);
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
                acts_first, use_vnet, vscale, bans);
    }
}

// 对手第二层：枚举 0..3 个行动，取让我们最不利的（minimize）
void dfs_opp(const sim::State& s, const sim::Belief& b, int used, bool free_turn,
             bool sees_us, int depth, double& worst, const Deadline& dl,
             SearchStats& st, const Weights& w, bool acts_first, bool use_vnet,
             double vscale) {
    {
        sim::State after = s;
        sim::end_side_turn(after, 'B');
        worst = std::min(worst, leaf_eval(after, b, w, acts_first, use_vnet, vscale,
                                          used, /*opp_to_move=*/false,
                                          sim::is_at_spawn(after, 'R')));
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
        dfs_opp(ns, b, nused, nfree, sees_us, depth + 1, worst, dl, st, w, acts_first,
                use_vnet, vscale);
    }
}

// ═════════ 深模式：两回合前瞻（我极大 → 对手极小 → end_round → 循环）═════════
//
// 浅层搜索只看"我方阶段 + 对手回应"，看不见"这步走进的走廊两回合后把我
// 关进枪线"的链条——最优策略线的挖掘（build/strategy_lines.csv）表明均衡
// 策略本质是耐心诱导，恰好需要这个深度才能看见。
//
// 预算与可靠性：节点预算耗尽 → complete=false → 上层回退浅层值。
// 部分探索的极小值对对手是乐观的（未探索=未惩罚），会骗我方走险手，
// 所以宁可弃用。终局/截止时间同理。

// 递归体：turns_left 还剩几轮"我+对手"完整回合。complete 置 false = 不可信。
// is_my_turn: 当前行动阶段属于我方（true）或对手（false）。
double deep_value(const sim::State& s, const sim::Belief& b, int used,
                  bool my_free, bool opp_free, bool enemy_visible, int turns_left,
                  bool is_my_turn, const Deadline& dl, SearchStats& st,
                  const Weights& w, bool use_vnet, double vscale,
                  long long& budget, bool& complete) {
    if (budget <= 0) { complete = false; return 0.0f; }
    if (out_of_time(dl)) { st.time_exhausted = true; complete = false; return 0.0f; }
    --budget;

    // 行动阶段结束 + 回合收尾的值（终局/静态都走这里）
    const auto phase_end_value = [&](const sim::State& phase_done,
                                     bool opp_just_acted) -> double {
        sim::State er = phase_done;
        if (opp_just_acted) sim::end_round(er); // 对手阶段结束 = 整回合结束
        if (brain::is_terminal(er)) {
            // 终局：用分差排序（与手工评估同尺度）
            return static_cast<double>(er.red.score - er.blue.score) * vscale;
        }
        if (turns_left <= (opp_just_acted ? 1 : 0)) {
            return leaf_eval(er, b, w, /*acts_first=*/true, use_vnet, vscale, used,
                             /*opp_to_move=*/!opp_just_acted,
                             /*my_free_left=*/sim::is_at_spawn(er, 'R'));
        }
        if (opp_just_acted) {
            // 整回合结束 → 我方下一阶段
            return deep_value(er, b, 0, sim::is_at_spawn(er, 'R'),
                              sim::is_at_spawn(er, 'B'), enemy_visible,
                              turns_left - 1, /*is_my_turn=*/true, dl, st, w,
                              use_vnet, vscale, budget, complete);
        }
        // 我方阶段结束 → 对手阶段（同一回合）
        return deep_value(er, b, 0, my_free, opp_free, enemy_visible, turns_left,
                          /*is_my_turn=*/false, dl, st, w, use_vnet, vscale, budget,
                          complete);
    };

    const char side = is_my_turn ? 'R' : 'B';
    const bool my_free_eff = is_my_turn ? my_free : opp_free;
    const bool opp_free_eff = is_my_turn ? opp_free : my_free;

    std::vector<Cand> cands;
    collect_candidates(s, side, my_free_eff, enemy_visible, nullptr, nullptr, cands);

    // 收手（结束当前阶段）永远是候选
    {
        sim::State es = s;
        sim::end_side_turn(es, side);
        const double v = phase_end_value(es, /*opp_just_acted=*/!is_my_turn);
        if (!complete) return 0.0f;
        // 记为当前最好值；动作候选在下面尝试超越它
        double cur_best = v;
        for (const Cand& c : cands) {
            sim::State ns = s;
            int nused = used;
            bool nft = my_free_eff;
            if (!apply_step(ns, side, c, nused, nft)) continue;
            if (is_my_turn) {
                sim::Belief nb = b;
                sim::prune_by_vision(nb, ns.red, ns.obstacles);
                if (nb.empty()) nb = b;
                const double cv = deep_value(ns, nb, nused, nft, opp_free_eff,
                                             enemy_visible, turns_left, true, dl, st,
                                             w, use_vnet, vscale, budget, complete);
                if (!complete) return 0.0f;
                if (cv > cur_best) cur_best = cv;
            } else {
                // 对手行动后我方视野不变（我们看不见他的移动）
                const double cv = deep_value(ns, b, nused, my_free, nft,
                                             enemy_visible, turns_left, false, dl, st,
                                             w, use_vnet, vscale, budget, complete);
                if (!complete) return 0.0f;
                if (cv < cur_best) cur_best = cv;
            }
        }
        return cur_best;
    }
}

// 对手在我方叶子状态上的最优回应值
double opponent_best(const sim::State& s, const sim::Belief& b, const Weights& w,
                     bool acts_first, bool use_vnet, double vscale,
                     int opp_def_until, const Deadline& dl, SearchStats& st,
                     int deep_turns) {
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

    const bool free_turn = sim::is_at_spawn(base, 'B');

    // 深模式：两回合前瞻。预算耗尽 → 回退浅层值（部分探索的极小值对对手
    // 是乐观的，会骗我方走险手，不可信）。
    if (deep_turns > 0) {
        long long budget = kDeepNodeBudget;
        bool complete = true;
        const double v = deep_value(base, b, 0, free_turn,
                                    /*opp_free=*/sim::is_at_spawn(base, 'R'),
                                    /*enemy_visible=*/false, deep_turns,
                                    /*is_my_turn=*/false, dl, st, w, use_vnet,
                                    vscale, budget, complete);
        if (complete) return v;
        // 回退浅层
    }

    double worst = leaf_eval(base, b, w, acts_first, use_vnet, vscale, 0,
                             /*opp_to_move=*/true, free_turn);
    dfs_opp(base, b, 0, free_turn, opp_sees_us, 0, worst, dl, st, w, acts_first,
            use_vnet, vscale);
    return worst;
}

} // namespace

Plan search_turn(const TurnInput& in, const Weights& w, const Deadline& deadline,
                 SearchStats* stats) {
    SearchStats st;
    std::vector<Leaf> leaves;
    Plan cur;
    dfs_our(in.state, in.belief, in.used_by_now, in.free_turn_available, in.enemy_visible,
            0, cur, leaves, deadline, st, w, in.acts_first_world, in.use_value_net,
            in.value_scale, &in);

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
                          in.use_value_net, in.value_scale,
                          in.opp_defenseless_until, deadline, st, in.deep_turns);
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

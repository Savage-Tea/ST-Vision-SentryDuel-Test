#include "brain/actions.h"

namespace brain {

Cand local_to_world(const Cand& local, char side) {
    Cand c = local;
    if (side == 'B' && c.action == sim::kTurn) {
        c.arg = sim::mirror_facing(c.arg);
    }
    return c;
}

void collect_candidates(const sim::State& s, char side, bool free_turn,
                        bool can_see_enemy, const Bans* bans,
                        const sim::Belief* belief, std::vector<Cand>& out) {
    out.clear();
    const Sentry& me = s.sentry_for(side);
    const bool at_spawn = sim::is_at_spawn(s, side);

    auto push = [&](int action, char arg) {
        if (bans != nullptr && bans->has(action, arg)) return;
        out.push_back({action, arg});
    };

    push(sim::kMove, 0);

    for (char f : {'N', 'E', 'S', 'W'}) {
        if (f == me.last_known_facing && !(free_turn && at_spawn)) continue;
        push(sim::kTurn, f);
    }

    // 开火候选的门槛。
    //
    // 【原版】can_see_enemy || 信念里**每一格**都在火力通道内。
    // 后者近似恒假：SideBelief::begin_turn 每回合都把信念按 3 格 dilate
    // （belief_state.cpp 的"① 扩张"），之后 certain=false、信念是一大团，
    // 而 all_in_fire_lane 要求全中才返回 true。于是门槛实际退化成
    // "肉眼看得见"——scan(CD 3) 之所以能开火，只是因为 scan 当回合会把
    // 信念塌缩成单点。净效果：**每 3 回合才有一次开火机会**；其余回合若
    // 敌人不在 2 格视野内，AI 连"开火"这个选项都不存在，只能 move/turn，
    // 而评估函数里唯一给分的就是往得分区走 —— 这正是回放里那条 3 回合
    // 死亡循环（扫描 → 走进枪口 → 死 → 复活）的成因。
    //
    // 【改】改为看搜索当前的敌方**代表位置**（= 最后已知位置，由
    // apply_belief → write_anchor_into 写回）。那才是搜索里真正在推演的
    // 那个点，也是开火时瞄准的目标。打空的概率该由叶评估的 threat_prob
    // 折扣承担，不该由候选生成器用"全知"标准一票否决。
    //
    // 保留 all_in_fire_lane 作为额外分支：新条件是它的超集，只增不减候选，
    // 不会让任何原本可用的选项消失。
    if (me.fire_cd == 0) {
        const Sentry& opp = s.sentry_for(side == 'R' ? 'B' : 'R');
        const bool anchor_hittable =
            sim::fire_hit(me, opp.last_known_pos, s.obstacles).x >= 0;
        const bool all_belief_hittable =
            belief != nullptr && sim::all_in_fire_lane(*belief, me, s.obstacles);
        if (can_see_enemy || anchor_hittable || all_belief_hittable) {
            push(sim::kFire, 0);
        }
    }
    if (me.scan_cd == 0) push(sim::kScan, 0);
}

bool apply_step(sim::State& s, char side, const Cand& c, int& used, bool& free_turn,
                bool* hit) {
    if (hit != nullptr) *hit = false;
    if (used >= kMaxActionsPerTurn) return false;

    // 引擎 do_action 顶部：一旦不在出生点，免费资格立即失效
    const bool was_at_spawn = sim::is_at_spawn(s, side);
    if (free_turn && !was_at_spawn) free_turn = false;

    const sim::Outcome o = sim::apply_action(s, side, c.action, c.arg);
    if (!o.success) return false;
    if (hit != nullptr) *hit = o.hit;

    // TURN 在出生点且免费资格仍在 → 不占额度（转向不改变位置，故 was_at_spawn 即当前状态）
    if (c.action == sim::kTurn && free_turn && was_at_spawn) {
        free_turn = false;
    } else {
        ++used;
    }
    return true;
}

void end_phase(sim::State& s, char side) {
    sim::end_side_turn(s, side);
    // 后手阶段结束 = 整回合结束：CD 递减、回合数 +1
    if (side == 'B') sim::end_round(s);
}

bool is_terminal(const sim::State& s) {
    if (s.turn < 20) return false;
    if (s.turn >= 25) return true;
    // 加时赛：每个回合结束时比分不再相等即决出胜负
    return s.red.score != s.blue.score;
}

double terminal_value(const sim::State& s, char perspective) {
    const Sentry& mine = s.sentry_for(perspective);
    const Sentry& theirs = s.sentry_for(perspective == 'R' ? 'B' : 'R');
    if (mine.score > theirs.score) return kWinValue;
    if (mine.score < theirs.score) return -kWinValue;
    return 0.0;
}

} // namespace brain

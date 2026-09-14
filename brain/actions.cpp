#include "brain/actions.h"

namespace brain {

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

    if (me.fire_cd == 0) {
        if (can_see_enemy) {
            push(sim::kFire, 0);
        } else if (belief != nullptr && sim::all_in_fire_lane(*belief, me, s.obstacles)) {
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

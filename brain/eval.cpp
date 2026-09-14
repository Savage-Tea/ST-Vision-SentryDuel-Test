#include "brain/eval.h"

#include <algorithm>
#include <cstdlib>

namespace brain {

int dist_to_zone(const Pos& p, const std::vector<Pos>& zones) {
    int best = 1 << 20;
    for (const Pos& z : zones) {
        best = std::min(best, sim::manhattan_distance(p, z));
    }
    return best == (1 << 20) ? 0 : best;
}

bool lane_clear(const Sentry& shooter, const Pos& target,
                const std::vector<Pos>& obstacles) {
    // 直接用 sim 的火力判定：命中点 >= 0 说明目标落在 3x3 内且通道无遮挡
    const Pos hit = sim::fire_hit(shooter, target, obstacles);
    return hit.x >= 0;
}

bool lane_clear_any_facing(const Sentry& shooter, const Pos& target,
                           const std::vector<Pos>& obstacles) {
    Sentry probe = shooter;
    for (char f : {'N', 'E', 'S', 'W'}) {
        probe.last_known_facing = f;
        if (lane_clear(probe, target, obstacles)) return true;
    }
    return false;
}

double evaluate(const sim::State& s, const Weights& w) {
    // 局部框架下：red = 我方，blue = 对手
    const Sentry& me = s.red;
    const Sentry& opp = s.blue;

    double v = 0.0;

    // ① 分差。搜索在我方行动序列之后会调用 end_side_turn('R')，
    //    所以这里的 me.score 已经包含本回合的占点分。
    v += w.w_diff * static_cast<double>(me.score - opp.score);

    // ② 占点状态与距离：刻画"未来还能拿多少分"
    const bool me_in = sim::in_score_zone(me.last_known_pos, s.score_zones);
    const bool opp_in = sim::in_score_zone(opp.last_known_pos, s.score_zones);
    if (me_in) v += w.w_zone;
    if (opp_in) v -= w.w_zone;

    const int my_d = dist_to_zone(me.last_known_pos, s.score_zones);
    const int op_d = dist_to_zone(opp.last_known_pos, s.score_zones);
    v += w.w_dist * static_cast<double>(op_d - my_d);

    // ③ 火力对峙。对手的朝向可能是过期情报，这里只用"最后已知朝向"，
    //    结果偏乐观或偏悲观都不可避免——阶段③ 用 RNN 压历史正是为了解决它。
    if (lane_clear(me, opp.last_known_pos, s.obstacles)) {
        v += w.w_threat;
        if (me.fire_cd == 0) v += w.w_ready;
    }
    // 危险项：对手朝向已知就按已知算；未知/过期则按最坏情况（任意朝向），
    // 否则会默认自己是安全的——对手从视野外接近时这就是致命的乐观。
    const bool danger = (opp.last_known_facing == '?')
                            ? lane_clear_any_facing(opp, me.last_known_pos, s.obstacles)
                            : lane_clear(opp, me.last_known_pos, s.obstacles);
    if (danger) {
        v -= w.w_danger;
    }

    return v;
}

} // namespace brain

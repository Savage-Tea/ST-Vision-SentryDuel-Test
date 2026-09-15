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

ThreatStats threat_stats(const sim::State& s, const sim::Belief& belief) {
    ThreatStats stats;
    const int n = belief.count();
    if (n <= 0) return stats;
    const Sentry& me = s.red;
    // 锚点 = 最后已知位置，即对局状态里给搜索用的那个"代表位置"
    const Pos anchor = s.blue.last_known_pos;

    int threat = 0;
    int unknown_total = 0;
    int unknown_danger = 0;
    for (int y = 0; y < sim::kBoardSize; ++y) {
        for (int x = 0; x < sim::kBoardSize; ++x) {
            if (!belief.has(x, y)) continue;
            const Pos c{x, y};
            // 对手在那里的话能否打到我？朝向未知 → 按最坏情况（任意朝向）
            Sentry enemy{};
            enemy.last_known_pos = c;
            const bool can_hit = lane_clear_any_facing(enemy, me.last_known_pos, s.obstacles);
            if (c.x == anchor.x && c.y == anchor.y) {
                stats.danger_known = can_hit ? 1.0 : 0.0;
            } else {
                ++unknown_total;
                if (can_hit) ++unknown_danger;
            }
            // 我在那里的话能否打到他？（用我方真实朝向）
            if (lane_clear(me, c, s.obstacles)) ++threat;
        }
    }
    // 锚点不在信念里时（例如击杀后锚点被重置），退化为按整集统计
    if (unknown_total == 0) {
        for (int y = 0; y < sim::kBoardSize; ++y)
            for (int x = 0; x < sim::kBoardSize; ++x) {
                if (!belief.has(x, y)) continue;
                ++unknown_total;
                Sentry enemy{};
                enemy.last_known_pos = {x, y};
                if (lane_clear_any_facing(enemy, me.last_known_pos, s.obstacles)) {
                    ++unknown_danger;
                }
            }
    }
    stats.danger_unknown =
        unknown_total > 0 ? static_cast<double>(unknown_danger) / unknown_total : 0.0;
    stats.threat_prob = static_cast<double>(threat) / static_cast<double>(n);
    return stats;
}

double evaluate(const sim::State& s, const Weights& w, const sim::Belief& belief,
                bool acts_first_world) {
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

    // ③ 火力对峙：对**敌方可能位置集合**取期望，而不是只看最后已知的那一点。
    //    这样"对手可能已经从视野外绕到我旁边"会如实体现在分数里。
    const ThreatStats ts = threat_stats(s, belief);
    v += w.w_threat * ts.threat_prob;
    if (ts.threat_prob > 0.0 && me.fire_cd == 0) v += w.w_ready * ts.threat_prob;
    // 确定的威胁给全权重；由信念推测出来的威胁打折，否则会瘫痪。
    //
    // 【CD 时序不对称】打不了还手（fire_cd>0）时，"被击杀/被压制的暴露期"
    // 红蓝不等：红方要熬 2 个对手惩罚窗口，蓝方只 1 个。同一格危险，
    // 对先手方更致命——这正是 #14 观察到的"对峙时红开火必胜"的机制面。
    // 让搜索看到这一点后，它应当自己学出：红方开火更谨慎（打空代价翻倍），
    // 蓝方更敢开火、且对峙时主动破势而非硬顶。
    const double exposure = (s.red.fire_cd > 0 && acts_first_world)
                                ? w.w_danger_red_scale
                                : 1.0;
    // 对手 CD 未恢复时他无法开火——树里已经禁止他开火，叶子上的危险是
    // "他恢复后的下一发"，给一半权重而不是全额（全额会让无力窗口形同虚设）。
    const double opp_ready = (s.blue.fire_cd > 0) ? 0.5 : 1.0;
    v -= w.w_danger * ts.danger_known * exposure * opp_ready;
    v -= w.w_danger * w.w_uncertain * ts.danger_unknown * exposure * opp_ready;

    return v;
}

double evaluate_for(const sim::State& s, const Weights& w, const sim::Belief& belief,
                    char side, bool acts_first_world) {
    if (side == 'R') return evaluate(s, w, belief, acts_first_world);
    // 把双方对调再按同一套公式算，得到的就是"轮到 'B' 时这局面对他有多好"。
    // 注意这不是简单取负：威胁/危险两项要换成从 'B' 的朝向与位置来算，
    // 对调红蓝正好做到这一点。
    //
    // 先手性也要跟着对调：我是先手 ⟹ 对手是后手。
    sim::State flipped = s;
    std::swap(flipped.red, flipped.blue);
    return evaluate(flipped, w, belief, !acts_first_world);
}

} // namespace brain

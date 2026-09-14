#include "obs/encode.h"

#include <algorithm>
#include <cstring>

namespace obs {
namespace {

void set_plane(float* out, int plane, int x, int y) {
    out[plane * kCells + cell(x, y)] = 1.0f;
}

float clamp01(float v) { return std::min(1.0f, std::max(0.0f, v)); }

} // namespace

void encode(const sim::State& s, const sim::Belief& belief, bool enemy_visible,
            int used_by_now, bool free_turn, float* out) {
    std::memset(out, 0, sizeof(float) * kObsDim);

    const Sentry& me = s.red;       // 局部视角下我方恒在 red 槽位
    const Sentry& opp = s.blue;

    // —— 平面 0/1：静态地图 ——
    for (const Pos& o : s.obstacles) set_plane(out, 0, o.x, o.y);
    for (const Pos& z : s.score_zones) set_plane(out, 1, z.x, z.y);

    // —— 平面 2：我方位置 ——
    if (sim::Belief::in_bounds(me.last_known_pos)) {
        set_plane(out, 2, me.last_known_pos.x, me.last_known_pos.y);
    }

    // —— 平面 3：敌方可达集信念（多热） ——
    for (int y = 0; y < sim::kBoardSize; ++y) {
        for (int x = 0; x < sim::kBoardSize; ++x) {
            if (belief.has(x, y)) set_plane(out, 3, x, y);
        }
    }

    // —— 平面 4：敌方当前直接可见位置 ——
    if (enemy_visible && sim::Belief::in_bounds(opp.last_known_pos)) {
        set_plane(out, 4, opp.last_known_pos.x, opp.last_known_pos.y);
    }

    // —— 平面 5：敌方最后已知位置（锚点） ——
    if (sim::Belief::in_bounds(opp.last_known_pos)) {
        set_plane(out, 5, opp.last_known_pos.x, opp.last_known_pos.y);
    }

    // —— 平面 6：我方火力覆盖（哪些格子我现在打得到） ——
    for (int y = 0; y < sim::kBoardSize; ++y) {
        for (int x = 0; x < sim::kBoardSize; ++x) {
            if (sim::fire_hit(me, Pos{x, y}, s.obstacles).x >= 0) {
                set_plane(out, 6, x, y);
            }
        }
    }

    // —— 平面 7：危险格（信念中能从那里打到我方的格子） ——
    // 这是部分可观测下"别从视野外走进对方枪口"的核心特征。
    for (int y = 0; y < sim::kBoardSize; ++y) {
        for (int x = 0; x < sim::kBoardSize; ++x) {
            if (!belief.has(x, y)) continue;
            Sentry probe{};
            probe.last_known_pos = {x, y};
            bool threatens = false;
            for (char f : {'N', 'E', 'S', 'W'}) {
                probe.last_known_facing = f;
                if (sim::fire_hit(probe, me.last_known_pos, s.obstacles).x >= 0) {
                    threatens = true;
                    break;
                }
            }
            if (threatens) set_plane(out, 7, x, y);
        }
    }

    // —— 标量 ——
    float* sc = out + kPlaneDim;
    int n = 0;

    // 我方朝向 one-hot
    switch (me.last_known_facing) {
        case 'N': sc[n + 0] = 1.0f; break;
        case 'E': sc[n + 1] = 1.0f; break;
        case 'S': sc[n + 2] = 1.0f; break;
        case 'W': sc[n + 3] = 1.0f; break;
        default: break;
    }
    n += 4;

    sc[n++] = clamp01(static_cast<float>(me.fire_cd) / 2.0f);
    sc[n++] = clamp01(static_cast<float>(me.scan_cd) / 3.0f);
    sc[n++] = clamp01(static_cast<float>(me.score) / 40.0f);
    sc[n++] = clamp01(static_cast<float>(opp.score) / 40.0f);
    sc[n++] = clamp01(static_cast<float>(s.turn) / 25.0f);
    sc[n++] = clamp01(static_cast<float>(used_by_now) / 3.0f);
    sc[n++] = free_turn ? 1.0f : 0.0f;

    // 敌方朝向 one-hot（4 个方向 + 未知）
    switch (opp.last_known_facing) {
        case 'N': sc[n + 0] = 1.0f; break;
        case 'E': sc[n + 1] = 1.0f; break;
        case 'S': sc[n + 2] = 1.0f; break;
        case 'W': sc[n + 3] = 1.0f; break;
        default: sc[n + 4] = 1.0f; break;
    }
    n += 5;

    sc[n++] = enemy_visible ? 1.0f : 0.0f;
    sc[n++] = clamp01(static_cast<float>(belief.count()) / static_cast<float>(kCells));
}

} // namespace obs

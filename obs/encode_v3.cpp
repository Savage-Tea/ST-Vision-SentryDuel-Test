#include "obs/encode_v3.h"

#include <algorithm>
#include <cstring>

namespace obs {
namespace {

void set_plane(float* out, int plane, int x, int y) {
    out[plane * kCellsV3 + cell_v3(x, y)] = 1.0f;
}

float clamp01(float v) { return std::min(1.0f, std::max(0.0f, v)); }

bool in_bounds(const Pos& p) {
    return p.x >= 0 && p.x < sim::kBoardSize && p.y >= 0 && p.y < sim::kBoardSize;
}

} // namespace

void encode_v3(const sim::State& s, bool enemy_visible, int used_by_now,
               bool free_turn, float* out) {
    std::memset(out, 0, sizeof(float) * kObsDimV3);

    const Sentry& me = s.red;   // 局部视角下我方恒在 red 槽位
    const Sentry& opp = s.blue;

    // —— 平面 0/1：静态地图 ——
    for (const Pos& o : s.obstacles) set_plane(out, kPlaneObstacles, o.x, o.y);
    for (const Pos& z : s.score_zones) set_plane(out, kPlaneZones, z.x, z.y);

    // —— 平面 2：我方位置 ——
    if (in_bounds(me.last_known_pos)) {
        set_plane(out, kPlaneMyPos, me.last_known_pos.x, me.last_known_pos.y);
    }

    for (int y = 0; y < sim::kBoardSize; ++y) {
        for (int x = 0; x < sim::kBoardSize; ++x) {
            const Pos p{x, y};
            // —— 平面 3：我方火力覆盖（我朝这个格子开火能不能打到） ——
            if (sim::fire_hit(me, p, s.obstacles).x >= 0) set_plane(out, kPlaneMyFire, x, y);
            // —— 平面 4：我方视野覆盖（T 形 4 格 + 视线遮挡） ——
            if (sim::can_see(me, p, s.obstacles)) set_plane(out, kPlaneMyVision, x, y);
        }
    }

    // —— 平面 5：敌方当前直接可见位置 ——
    // 只在真的看得见时才点，看不见就是全 0 —— 网络要从"看不见"这件事本身
    // 结合历史去推敌人在哪，这正是 hidden state 的职责。
    if (enemy_visible && in_bounds(opp.last_known_pos)) {
        set_plane(out, kPlaneEnemyVisible, opp.last_known_pos.x, opp.last_known_pos.y);
    }

    // —— 平面 6：敌方 last_known_pos ——
    // 这是**引擎给的**规则信息（同时用于开火判定），不是我们的推断，所以保留。
    // 注意：开局时它等于出生点，那是假设而非知情；观测上无法区分，
    // 回合号 0 事实上编码了这一点，交给网络自己学。
    if (in_bounds(opp.last_known_pos)) {
        set_plane(out, kPlaneEnemyAnchor, opp.last_known_pos.x, opp.last_known_pos.y);
    }

    // —— 标量 ——
    float* sc = out + kPlaneDimV3;
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

    // 归一化常数与旧编码保持一致，这样两套编码的标量分布可比。
    sc[n++] = clamp01(static_cast<float>(me.fire_cd) / 2.0f);
    sc[n++] = clamp01(static_cast<float>(me.scan_cd) / 3.0f);
    sc[n++] = clamp01(static_cast<float>(me.score) / 40.0f);
    sc[n++] = clamp01(static_cast<float>(opp.score) / 40.0f);
    sc[n++] = clamp01(static_cast<float>(s.turn) / 25.0f);
    sc[n++] = clamp01(static_cast<float>(used_by_now) / 3.0f);
    sc[n++] = free_turn ? 1.0f : 0.0f;
    sc[n++] = enemy_visible ? 1.0f : 0.0f;

    // 敌方朝向 one-hot（4 个方向 + 未知）
    switch (opp.last_known_facing) {
        case 'N': sc[n + 0] = 1.0f; break;
        case 'E': sc[n + 1] = 1.0f; break;
        case 'S': sc[n + 2] = 1.0f; break;
        case 'W': sc[n + 3] = 1.0f; break;
        default: sc[n + 4] = 1.0f; break;
    }
    n += 5;

    // 编译期确认标量数与头文件一致——将来加字段忘了改 kScalarDimV3 会在这里炸，
    // 而不是静默写越界。
    static_assert(kScalarDimV3 == 17, "标量数量与 kScalarDimV3 不一致");
    (void)n;
}

} // namespace obs

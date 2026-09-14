#include "sim/view_mirror.h"

namespace sim {
namespace {

// 对手的坐标槽位：我方是 'R' 时对手在 blue，反之亦然。
char opponent_of(char side) { return side == 'R' ? 'B' : 'R'; }

} // namespace

void ViewMirror::reset() {
    for (int i = 0; i < 2; ++i) {
        intel_[i] = Intel{};
        visible_[i] = false;
        scanned_[i] = false;
    }
}

bool ViewMirror::direct_vision(const State& world, char side) const {
    // 对应 Match::direct_vision：在**世界坐标**下判定，用的是敌方真实位置。
    // 这正是情报的来源——不是记忆，而是当下这一眼。
    const Sentry& me = world.sentry_for(side);
    const Sentry& enemy = world.sentry_for(opponent_of(side));
    return can_see(me, enemy.last_known_pos, world.obstacles);
}

void ViewMirror::remember(const State& world, char side) {
    const Sentry& enemy = world.sentry_for(opponent_of(side));
    Intel& in = intel_[idx(side)];
    in.pos = enemy.last_known_pos;
    in.facing = enemy.last_known_facing;
    in.known = true;
}

Pos ViewMirror::local_pos(Pos p, char side, int size) {
    return side == 'B' ? mirror_pos(p, size) : p;
}

char ViewMirror::local_facing(char f, char side) {
    return side == 'B' ? mirror_facing(f) : f;
}

void ViewMirror::begin_phase(const State& world, char side) {
    // Match::act_phase 开头：scanned_this_act_ = false，然后 view_for(side)
    scanned_[idx(side)] = false;
    const bool dv = direct_vision(world, side);
    if (dv) remember(world, side);
    // view_for 里的 visible 就是当下的直接视野（此时 scanned 已清）
    visible_[idx(side)] = dv;
}

void ViewMirror::after_action(const State& world, char side, int action) {
    const int i = idx(side);
    if (action == kScan) {
        // 引擎里 SCAN 成功是无条件 remember，不要求"看得见"
        scanned_[i] = true;
        remember(world, side);
    }
    // 【注意】action 是 FIRE 且命中时，这里**什么也不做**。
    // 引擎命中后只置 respawn_turn_pending(对手)，那只影响对方下次的免费转向；
    // Intel 保持不动——它还停在我们开枪时对手所在的位置。
    // 自对弈侧若"顺手"把它重置到出生点，就会与真机观测不一致。

    // refresh_current_vision()
    const bool dv = direct_vision(world, side);
    if (dv) remember(world, side);
    visible_[i] = scanned_[i] || dv;
}

State ViewMirror::local_view(const State& world, char side) const {
    State v = to_local(world, side); // 蓝方会得到 180° 镜像并交换红蓝槽位

    const Intel& in = intel_[idx(side)];
    if (in.known) {
        v.blue.last_known_pos = local_pos(in.pos, side, world.size);
        v.blue.last_known_facing = local_facing(in.facing, side);
    } else {
        // 开局（以及从未见过）时引擎给的就是这个，不是出生点
        v.blue.last_known_pos = Pos{-1, -1};
        v.blue.last_known_facing = '?';
    }
    v.blue.visible = visible_[idx(side)];
    // 引擎不向选手暴露对手的 CD（恒为 -1），这里必须一致
    v.blue.fire_cd = -1;
    v.blue.scan_cd = -1;
    return v;
}

} // namespace sim

#include "brain/belief_state.h"

namespace brain {
namespace {
// 敌方一个行动阶段最多走 3 格（与 kMaxActionsPerTurn 一致）
constexpr int kEnemyStepsPerTurn = 3;
} // namespace

void SideBelief::clear() {
    set.clear();
    anchor = {-1, -1};
    anchor_facing = '?';
    known = false;
    certain = false;
}

void SideBelief::collapse(const Pos& p, char facing) {
    set.reset_to(p);
    anchor = p;
    if (facing != 0) anchor_facing = facing;
    known = true;
    certain = true;
}

void SideBelief::infer_at(const Pos& p) {
    set.reset_to(p);
    anchor = p;
    known = true;
    certain = true;
}

void SideBelief::on_our_hit(int size) {
    collapse(sim::spawn_of('B', size), sim::spawn_facing('B'));
}

void SideBelief::begin_turn(const sim::State& local, int turn, bool enemy_visible) {
    const Pos reported = local.blue.last_known_pos;       // 引擎给的敌方位置
    const char reported_facing = local.blue.last_known_facing;

    // ① 扩张：对手自上次我方行动后走完了一个行动阶段
    if (turn != 0 && known) {
        sim::dilate(set, kEnemyStepsPerTurn, local.red.last_known_pos, local.obstacles);
        certain = false; // 位置不再确定 —— 正是该开雷达的时刻
    }
    // ② 塌缩：看得见 → 精确位置；否则若还没有任何情报，用引擎给的兜底
    if (enemy_visible && reported.x >= 0) {
        collapse(reported, reported_facing);
    } else if (!known && reported.x >= 0) {
        collapse(reported, reported_facing);
    }
    // ③ 从未得知 → 敌方出生点（局部坐标 (6,6)）
    if (!known) {
        collapse(sim::spawn_of('B', local.size), sim::spawn_facing('B'));
    }
    // ④ 证伪：当前 T 形视野内确认无人的格子剔除
    sim::prune_by_vision(set, local.red, local.obstacles);
    sim::prune_impossible(set, local.red.last_known_pos, local.obstacles);
    if (set.empty()) set.reset_to(anchor);
}

void SideBelief::after_action(const sim::State& local) {
    sim::prune_by_vision(set, local.red, local.obstacles);
    sim::prune_impossible(set, local.red.last_known_pos, local.obstacles);
    if (set.empty()) set.reset_to(anchor);
}

void SideBelief::write_anchor_into(sim::State& local) const {
    local.blue.last_known_pos = anchor;
    local.blue.last_known_facing = anchor_facing;
}

} // namespace brain

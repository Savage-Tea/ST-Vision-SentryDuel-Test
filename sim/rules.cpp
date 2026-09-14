// sim/rules.cpp —— 规则核心实现
//
// 【重要】本文件是 engine/src/board.cpp 与 engine/src/utils.cpp 的忠实重实现。
// 任何一行偏离都会让搜索基于错误的世界模型而**静默地**变蠢，
// 所以 tests/difftest_rules.cpp 会对全状态空间与引擎逐字段比对。
// 修改规则时：先改这里 → 跑差分测试 → 若引擎也变了，两边都要改。

#include "sim/rules.h"

#include <algorithm>
#include <cmath>

namespace sim {
namespace {

// 对应 engine/src/board.cpp:36-59 的 fire_offsets
void fire_offsets(char facing, int dx[9], int dy[9]) {
    // 引擎此处的 switch 没有 default 分支，非法朝向会读到未初始化数组（UB）。
    // 真实对局中朝向恒为 NESW，这里显式清零给出确定行为，避免我们自己也踩 UB。
    std::fill(dx, dx + 9, 0);
    std::fill(dy, dy + 9, 0);
    switch (facing) {
        case 'N':
            dx[0] = -1; dy[0] = -1; dx[1] = 0; dy[1] = -1; dx[2] = 1; dy[2] = -1;
            dx[3] = -1; dy[3] = -2; dx[4] = 0; dy[4] = -2; dx[5] = 1; dy[5] = -2;
            dx[6] = -1; dy[6] = -3; dx[7] = 0; dy[7] = -3; dx[8] = 1; dy[8] = -3;
            break;
        case 'E':
            dx[0] = 1; dy[0] = -1; dx[1] = 1; dy[1] = 0; dx[2] = 1; dy[2] = 1;
            dx[3] = 2; dy[3] = -1; dx[4] = 2; dy[4] = 0; dx[5] = 2; dy[5] = 1;
            dx[6] = 3; dy[6] = -1; dx[7] = 3; dy[7] = 0; dx[8] = 3; dy[8] = 1;
            break;
        case 'S':
            dx[0] = -1; dy[0] = 1; dx[1] = 0; dy[1] = 1; dx[2] = 1; dy[2] = 1;
            dx[3] = -1; dy[3] = 2; dx[4] = 0; dy[4] = 2; dx[5] = 1; dy[5] = 2;
            dx[6] = -1; dy[6] = 3; dx[7] = 0; dy[7] = 3; dx[8] = 1; dy[8] = 3;
            break;
        case 'W':
            dx[0] = -1; dy[0] = -1; dx[1] = -1; dy[1] = 0; dx[2] = -1; dy[2] = 1;
            dx[3] = -2; dy[3] = -1; dx[4] = -2; dy[4] = 0; dx[5] = -2; dy[5] = 1;
            dx[6] = -3; dy[6] = -1; dx[7] = -3; dy[7] = 0; dx[8] = -3; dy[8] = 1;
            break;
        default:
            break;
    }
}

// 对应 engine/src/utils.cpp:31-60 的 vision_offsets（T 形 4 格）
void vision_offsets(char facing, int dx[4], int dy[4]) {
    std::fill(dx, dx + 4, 0);
    std::fill(dy, dy + 4, 0);
    switch (facing) {
        case 'N':
            dx[0] = 0;  dy[0] = -1;
            dx[1] = -1; dy[1] = -2; dx[2] = 0; dy[2] = -2; dx[3] = 1; dy[3] = -2;
            break;
        case 'E':
            dx[0] = 1;  dy[0] = 0;
            dx[1] = 2;  dy[1] = -1; dx[2] = 2; dy[2] = 0; dx[3] = 2; dy[3] = 1;
            break;
        case 'S':
            dx[0] = 0;  dy[0] = 1;
            dx[1] = -1; dy[1] = 2; dx[2] = 0; dy[2] = 2; dx[3] = 1; dy[3] = 2;
            break;
        case 'W':
            dx[0] = -1; dy[0] = 0;
            dx[1] = -2; dy[1] = -1; dx[2] = -2; dy[2] = 0; dx[3] = -2; dy[3] = 1;
            break;
        default:
            break;
    }
}

// 对应 engine/src/utils.cpp:64-77 的 blocked_by_obstacle。
// 注意引擎是「Bresenham 风格线段采样」：只走 steps-1 步，
// 所以距离 1 的目标永不判遮挡，距离 2 的斜向侧格检查的是斜前中间格。
bool blocked_by_obstacle(int mx, int my, int tx, int ty,
                         const std::vector<Pos>& obstacles) {
    int x = mx;
    int y = my;
    const int sx = (tx > x) ? 1 : (tx < x ? -1 : 0);
    const int sy = (ty > y) ? 1 : (ty < y ? -1 : 0);
    const int steps = std::max(std::abs(tx - x), std::abs(ty - y));
    for (int i = 1; i < steps; ++i) {
        x += sx;
        y += sy;
        if (obstacle_at(x, y, obstacles)) return true;
    }
    return false;
}

} // namespace

bool obstacle_at(int x, int y, const std::vector<Pos>& obstacles) {
    for (const Pos& o : obstacles) {
        if (o.x == x && o.y == y) return true;
    }
    return false;
}

void facing_delta(char f, int& dx, int& dy) {
    dx = 0;
    dy = 0;
    switch (f) {
        case 'N': dx = 0;  dy = -1; break;
        case 'E': dx = 1;  dy = 0;  break;
        case 'S': dx = 0;  dy = 1;  break;
        case 'W': dx = -1; dy = 0;  break;
        default: break;
    }
}

Pos mirror_pos(Pos p, int size) { return {size - 1 - p.x, size - 1 - p.y}; }

char mirror_facing(char f) {
    switch (f) {
        case 'N': return 'S';
        case 'E': return 'W';
        case 'S': return 'N';
        case 'W': return 'E';
        default: return '?';
    }
}

Pos spawn_of(char side, int size) {
    return side == 'R' ? Pos{0, 0} : Pos{size - 1, size - 1};
}

char spawn_facing(char side) { return side == 'R' ? 'E' : 'W'; }

bool is_at_spawn(const State& s, char side) {
    const Pos& pos = s.sentry_for(side).last_known_pos;
    const Pos spawn = spawn_of(side, s.size);
    return pos.x == spawn.x && pos.y == spawn.y;
}

State make_initial_state(int size) {
    State s;
    s.size = size;
    s.turn = 0;

    // 与引擎一致：Board 里的 visible 字段在交给选手前会被重算，此处先置 false
    s.red.last_known_pos = {0, 0};
    s.red.last_known_facing = 'E';
    s.red.visible = false;
    s.red.fire_cd = 0;
    s.red.scan_cd = 0;
    s.red.score = 0;

    s.blue.last_known_pos = {size - 1, size - 1};
    s.blue.last_known_facing = 'W';
    s.blue.visible = false;
    s.blue.fire_cd = 0;
    s.blue.scan_cd = 0;
    s.blue.score = 0;

    s.obstacles = {{1, 1}, {5, 5}};

    // 得分区：中心十字 5 格
    const int m = size / 2;
    s.score_zones = {{m, m - 1}, {m - 1, m}, {m, m}, {m + 1, m}, {m, m + 1}};

    return s;
}

Pos fire_hit(const Sentry& me, const Pos& opp_pos,
             const std::vector<Pos>& obstacles) {
    int odx[9], ody[9];
    fire_offsets(me.last_known_facing, odx, ody);

    int fx = -1;
    int fy = -1;
    for (int i = 0; i < 9; ++i) {
        const int tx = me.last_known_pos.x + odx[i];
        const int ty = me.last_known_pos.y + ody[i];
        if (tx == opp_pos.x && ty == opp_pos.y) {
            fx = tx;
            fy = ty;
            break;
        }
    }
    if (fx < 0) return {-1, -1};

    const int forward = (me.last_known_facing == 'E' || me.last_known_facing == 'W')
                            ? std::abs(fx - me.last_known_pos.x)
                            : std::abs(fy - me.last_known_pos.y);
    for (int distance = 1; distance < forward; ++distance) {
        int x = me.last_known_pos.x;
        int y = me.last_known_pos.y;
        if (me.last_known_facing == 'E') x += distance;
        if (me.last_known_facing == 'W') x -= distance;
        if (me.last_known_facing == 'S') y += distance;
        if (me.last_known_facing == 'N') y -= distance;
        // 对齐到目标的横向坐标，形成一条平行于朝向的通道
        if (me.last_known_facing == 'E' || me.last_known_facing == 'W') {
            y = fy;
        } else {
            x = fx;
        }
        if (obstacle_at(x, y, obstacles)) return {-1, -1};
    }
    return {fx, fy};
}

Outcome apply_action(State& s, char side, int action, char arg) {
    Sentry& me = s.sentry_for(side);
    Sentry& opp = s.sentry_for(side == 'R' ? 'B' : 'R');
    Pos& my_pos = me.last_known_pos;
    const Pos opp_pos = opp.last_known_pos;
    const std::vector<Pos>& obstacles = s.obstacles;

    if (action == kMove) {
        int dx = 0;
        int dy = 0;
        facing_delta(me.last_known_facing, dx, dy);
        const int nx = my_pos.x + dx;
        const int ny = my_pos.y + dy;
        if (nx < 0 || nx >= s.size || ny < 0 || ny >= s.size) return {false, false};
        if (obstacle_at(nx, ny, obstacles)) return {false, false};
        if (nx == opp_pos.x && ny == opp_pos.y) return {false, false};
        my_pos.x = nx;
        my_pos.y = ny;
        return {true, false};
    }

    if (action == kTurn) {
        if (arg != 'N' && arg != 'E' && arg != 'S' && arg != 'W') return {false, false};
        // 即使转向不变（已是该朝向）也正常生效
        me.last_known_facing = arg;
        return {true, false};
    }

    if (action == kFire) {
        if (me.fire_cd > 0) return {false, false};
        const Pos hit = fire_hit(me, opp_pos, obstacles);
        me.fire_cd = 2;
        if (hit.x >= 0) {
            // 命中：对方回出生点、朝向重置，但 fire_cd / scan_cd 保留
            opp.last_known_pos = spawn_of(side == 'R' ? 'B' : 'R', s.size);
            opp.last_known_facing = spawn_facing(side == 'R' ? 'B' : 'R');
            me.score += 2;
            return {true, true};
        }
        return {true, false};
    }

    if (action == kScan) {
        if (me.scan_cd > 0) return {false, false};
        me.scan_cd = 3;
        return {true, false};
    }

    return {false, false};
}

void end_side_turn(State& s, char side) {
    Sentry& sentry = s.sentry_for(side);
    if (in_score_zone(sentry.last_known_pos, s.score_zones)) sentry.score++;
}

void end_round(State& s) {
    for (char side : {'R', 'B'}) {
        Sentry& sentry = s.sentry_for(side);
        if (sentry.fire_cd > 0) sentry.fire_cd--;
        if (sentry.scan_cd > 0) sentry.scan_cd--;
    }
    s.turn++;
}

bool can_see(const Sentry& me, const Pos& target,
             const std::vector<Pos>& obstacles) {
    const int mx = me.last_known_pos.x;
    const int my = me.last_known_pos.y;
    int dx[4], dy[4];
    vision_offsets(me.last_known_facing, dx, dy);

    for (int i = 0; i < 4; ++i) {
        const int tx = mx + dx[i];
        const int ty = my + dy[i];
        if (tx == target.x && ty == target.y) {
            return !blocked_by_obstacle(mx, my, tx, ty, obstacles);
        }
    }
    // 自己所在的格子也算可见
    if (target.x == mx && target.y == my) return true;
    return false;
}

int manhattan_distance(const Pos& a, const Pos& b) {
    return std::abs(a.x - b.x) + std::abs(a.y - b.y);
}

bool in_score_zone(const Pos& pos, const std::vector<Pos>& zones) {
    for (const Pos& p : zones) {
        if (p.x == pos.x && p.y == pos.y) return true;
    }
    return false;
}

char best_turn_to_face(const Pos& from, const Pos& to) {
    const int dx = to.x - from.x;
    const int dy = to.y - from.y;
    if (std::abs(dx) >= std::abs(dy)) {
        return (dx >= 0) ? 'E' : 'W';
    }
    return (dy >= 0) ? 'S' : 'N';
}

bool can_move_forward(const Sentry& me, const Pos& opp_pos,
                      const std::vector<Pos>& obstacles, int board_size) {
    int dx = 0;
    int dy = 0;
    facing_delta(me.last_known_facing, dx, dy);
    const int nx = me.last_known_pos.x + dx;
    const int ny = me.last_known_pos.y + dy;

    if (nx < 0 || nx >= board_size || ny < 0 || ny >= board_size) return false;
    if (obstacle_at(nx, ny, obstacles)) return false;
    if (nx == opp_pos.x && ny == opp_pos.y) return false;
    return true;
}

} // namespace sim

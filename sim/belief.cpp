#include "sim/belief.h"

#include <cstdlib>

namespace sim {
namespace {

constexpr int kDx[4] = {0, 1, 0, -1};
constexpr int kDy[4] = {-1, 0, 1, 0};

} // namespace

int Belief::count() const {
    int n = 0;
    for (unsigned char c : cells) {
        if (c != 0) ++n;
    }
    return n;
}

Pos Belief::nearest_to(const Pos& p) const {
    Pos best{-1, -1};
    int best_d = 1 << 20;
    for (int y = 0; y < kBoardSize; ++y) {
        for (int x = 0; x < kBoardSize; ++x) {
            if (!has(x, y)) continue;
            const int d = std::abs(x - p.x) + std::abs(y - p.y);
            if (d < best_d) {
                best_d = d;
                best = {x, y};
            }
        }
    }
    return best;
}

void dilate(Belief& b, int steps, const Pos& my_pos, const std::vector<Pos>& obstacles) {
    for (int s = 0; s < steps; ++s) {
        Belief next = b;
        for (int y = 0; y < kBoardSize; ++y) {
            for (int x = 0; x < kBoardSize; ++x) {
                if (!b.has(x, y)) continue;
                for (int d = 0; d < 4; ++d) {
                    const int nx = x + kDx[d];
                    const int ny = y + kDy[d];
                    if (!Belief::in_bounds({nx, ny})) continue;
                    if (obstacle_at(nx, ny, obstacles)) continue;
                    if (nx == my_pos.x && ny == my_pos.y) continue;
                    next.set(nx, ny);
                }
            }
        }
        b = next;
    }
}

void prune_impossible(Belief& b, const Pos& my_pos, const std::vector<Pos>& obstacles) {
    for (int y = 0; y < kBoardSize; ++y) {
        for (int x = 0; x < kBoardSize; ++x) {
            if (!b.has(x, y)) continue;
            if (obstacle_at(x, y, obstacles) || (x == my_pos.x && y == my_pos.y)) {
                b.cells[cell_index(x, y)] = 0;
            }
        }
    }
}

void prune_by_vision(Belief& b, const Sentry& me, const std::vector<Pos>& obstacles) {
    for (int y = 0; y < kBoardSize; ++y) {
        for (int x = 0; x < kBoardSize; ++x) {
            if (!b.has(x, y)) continue;
            if (can_see(me, Pos{x, y}, obstacles)) b.cells[cell_index(x, y)] = 0;
        }
    }
}

bool any_in_fire_lane(const Belief& b, const Sentry& me,
                      const std::vector<Pos>& obstacles) {
    for (int y = 0; y < kBoardSize; ++y) {
        for (int x = 0; x < kBoardSize; ++x) {
            if (!b.has(x, y)) continue;
            if (fire_hit(me, Pos{x, y}, obstacles).x >= 0) return true;
        }
    }
    return false;
}

bool all_in_fire_lane(const Belief& b, const Sentry& me,
                      const std::vector<Pos>& obstacles) {
    bool any = false;
    for (int y = 0; y < kBoardSize; ++y) {
        for (int x = 0; x < kBoardSize; ++x) {
            if (!b.has(x, y)) continue;
            any = true;
            if (fire_hit(me, Pos{x, y}, obstacles).x < 0) return false;
        }
    }
    return any;
}

bool intersect_zone(Belief& b, const std::vector<Pos>& zones) {
    Belief cand = b;
    int kept = 0;
    for (int y = 0; y < kBoardSize; ++y) {
        for (int x = 0; x < kBoardSize; ++x) {
            if (!cand.has(x, y)) continue;
            bool in_zone = false;
            for (const Pos& z : zones) {
                if (z.x == x && z.y == y) { in_zone = true; break; }
            }
            if (in_zone) {
                ++kept;
            } else {
                cand.cells[cell_index(x, y)] = 0;
            }
        }
    }
    // 交集为空 = 这条推断与信念矛盾（多半是分数 delta 算错了）。
    // 保留原信念：宁可少一条证据，也不能把真位置剔出去。
    if (kept == 0) return false;
    b = cand;
    return true;
}

} // namespace sim

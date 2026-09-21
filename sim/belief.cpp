#include "sim/belief.h"

#include <cstdlib>

namespace sim {
namespace {

constexpr int kDx[4] = {0, 1, 0, -1};
constexpr int kDy[4] = {-1, 0, 1, 0};

} // namespace

int Belief::count() const {
    int n = 0;
    for (float v : mass) {
        if (v > 0.0f) ++n;
    }
    return n;
}

float Belief::total_mass() const {
    float s = 0.0f;
    for (float v : mass) s += v;
    return s;
}

bool Belief::normalize() {
    const float s = total_mass();
    if (!(s > 0.0f)) return false;   // 空/未初始化 → 不动内容，交调用方回退
    const float inv = 1.0f / s;
    for (float& v : mass) v *= inv;
    return true;
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
    // 每个源格把自己的质量**均匀铺到它自己的可达集**上，而不是"整体并集扩张"。
    // 支持集（谁能到）两者相同，差别全在质量：并集扩张会让每个格子都拿到
    // 同样的一份，于是"从多数可能位置都到不了的死角"也分到等量质量——
    // 那正是旧版 threat_prob / danger_unknown 失真的来源。
    Belief next;
    for (int y = 0; y < kBoardSize; ++y) {
        for (int x = 0; x < kBoardSize; ++x) {
            const float m = b.at(x, y);
            if (m <= 0.0f) continue;

            Belief reach;
            reach.set(x, y);
            for (int s = 0; s < steps; ++s) {
                Belief grown = reach;
                for (int yy = 0; yy < kBoardSize; ++yy) {
                    for (int xx = 0; xx < kBoardSize; ++xx) {
                        if (reach.at(xx, yy) <= 0.0f) continue;
                        for (int d = 0; d < 4; ++d) {
                            const int nx = xx + kDx[d];
                            const int ny = yy + kDy[d];
                            if (!Belief::in_bounds({nx, ny})) continue;
                            if (obstacle_at(nx, ny, obstacles)) continue;
                            if (nx == my_pos.x && ny == my_pos.y) continue;
                            grown.set(nx, ny);
                        }
                    }
                }
                reach = grown;
            }
            const int n = reach.count();
            if (n <= 0) continue;
            const float share = m / static_cast<float>(n);
            for (int i = 0; i < kCells; ++i) {
                if (reach.mass[i] > 0.0f) next.mass[i] += share;
            }
        }
    }
    b = next;   // 质量守恒：Σnext = Σb
}

void prune_impossible(Belief& b, const Pos& my_pos, const std::vector<Pos>& obstacles) {
    const Belief backup = b;
    for (int y = 0; y < kBoardSize; ++y) {
        for (int x = 0; x < kBoardSize; ++x) {
            if (!b.has(x, y)) continue;
            if (obstacle_at(x, y, obstacles) || (x == my_pos.x && y == my_pos.y)) {
                b.clear_at(x, y);
            }
        }
    }
    // 剔完要归一，否则总质量悄悄小于 1，后续所有概率都偏低。
    // 全被剔光说明这条证据与信念矛盾 → 回退，不能把真位置丢掉。
    if (!b.normalize()) b = backup;
}

void prune_by_vision(Belief& b, const Sentry& me, const std::vector<Pos>& obstacles) {
    const Belief backup = b;
    for (int y = 0; y < kBoardSize; ++y) {
        for (int x = 0; x < kBoardSize; ++x) {
            if (!b.has(x, y)) continue;
            if (can_see(me, Pos{x, y}, obstacles)) b.clear_at(x, y);
        }
    }
    if (!b.normalize()) b = backup;
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
    float kept = 0.0f;
    for (int y = 0; y < kBoardSize; ++y) {
        for (int x = 0; x < kBoardSize; ++x) {
            if (!cand.has(x, y)) continue;
            bool in_zone = false;
            for (const Pos& z : zones) {
                if (z.x == x && z.y == y) { in_zone = true; break; }
            }
            if (in_zone) {
                kept += cand.at(x, y);
            } else {
                cand.clear_at(x, y);
            }
        }
    }
    // 留下的质量为零 = 这条推断与信念矛盾（多半是分数 delta 算错了）。
    // 保留原信念：宁可少一条证据，也不能把真位置剔出去。
    if (!(kept > 0.0f) || !cand.normalize()) return false;
    b = cand;
    return true;
}

} // namespace sim

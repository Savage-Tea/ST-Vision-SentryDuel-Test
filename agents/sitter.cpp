// agents/sitter.cpp —— S1 占点旋转流（纯反射策略，无搜索、无评估函数）
//
// 【它存在的理由】用一个 ~120 行的可解释策略检验一个对局结构假设：
// "得分区 5 格、占点不互斥，赖在区里换格躲枪线，能不吃对枪就拿到稳定分"。
// 若 sitter-蓝能赢 baseline（而 phase①-蓝是 0/100），说明该对局应该用
// 按颜色混合的策略，而不是继续调搜索/评估。
//
// 策略（按优先级，每条都可解释）：
//   1. 免费枪：对手可见、在火力通道内、fire_cd==0，开火。
//   2. 区内躲枪线：在得分区且当前格可能被对手打到 → 换到区内安全的格子。
//   3. 进区：不在区 → 朝最近的"安全"得分区格走（先转朝向再走）。
//   4. 没事就收手（省额度）。
// "可能被打到"用悲观判定：对手最后已知位置 + 任意朝向都能打到我才算数。

#include "sim/rules.h"

#include <cstdlib>
#include <vector>

namespace {

// 对手（最后已知位置，任意朝向）能否打到我所在的格子
bool cell_is_dangerous(const sim::State& s, const Pos& me) {
    Sentry probe{};
    probe.last_known_pos = s.blue.last_known_pos;
    if (probe.last_known_pos.x < 0) return false; // 从未见过对手：不算危险
    for (const char f : {'N', 'E', 'S', 'W'}) {
        probe.last_known_facing = f;
        if (sim::fire_hit(probe, me, s.obstacles).x >= 0) return true;
    }
    return false;
}

bool in_zone(const sim::State& s, const Pos& p) {
    return sim::in_score_zone(p, s.score_zones);
}

} // namespace

extern "C" void act(const Board& board, char my_color) {
    // 局部视角：我方恒在 red 槽位（引擎已对蓝方镜像）
    sim::State s{};
    s.turn = board.turn;
    s.size = board.size;
    s.obstacles = board.obstacles;
    s.score_zones = board.score_zones;
    s.red = board.red;   // 我方
    s.blue = board.blue; // 对手（观测口径）

    const Pos me = s.red.last_known_pos;
    const bool dangerous = cell_is_dangerous(s, me);

    // 1) 免费枪
    if (s.blue.visible && s.red.fire_cd == 0 &&
        sim::fire_hit(s.red, s.blue.last_known_pos, s.obstacles).x >= 0) {
        fire();
        return;
    }

    // 2) 区内躲枪线：换到区内安全的格子
    if (in_zone(s, me) && dangerous) {
        const Pos* best = nullptr;
        for (const Pos& z : s.score_zones) {
            if (z.x == me.x && z.y == me.y) continue;
            if (cell_is_dangerous(s, z)) continue;
            const int d = std::abs(z.x - me.x) + std::abs(z.y - me.y);
            int bd = 1 << 20;
            if (best != nullptr) bd = std::abs(best->x - me.x) + std::abs(best->y - me.y);
            if (d < bd) best = &z;
        }
        if (best != nullptr) {
            const char want = sim::best_turn_to_face(me, *best);
            if (s.red.last_known_facing != want) {
                turn(want);
                return;
            }
            move();
            return;
        }
        // 区内无安全格：留在原地（出区更亏），收手
        return;
    }

    // 3) 进区：朝最近的得分区格（优先不被枪线覆盖的）
    if (!in_zone(s, me)) {
        const Pos* best = nullptr;
        int bd = 1 << 20;
        for (const Pos& z : s.score_zones) {
            const bool dng = dangerous && cell_is_dangerous(s, z);
            const int d = std::abs(z.x - me.x) + std::abs(z.y - me.y) + (dng ? 8 : 0);
            if (d < bd) {
                bd = d;
                best = &z;
            }
        }
        if (best == nullptr) return;
        const char want = sim::best_turn_to_face(me, *best);
        if (s.red.last_known_facing != want) {
            turn(want);
            return;
        }
        move();
        return;
    }

    // 4) 区内且安全：收手（留在原地吃 +1）
}

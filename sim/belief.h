// sim/belief.h —— 敌方可能位置集合（可达集信念）
//
// 为什么要它：只维护"最后已知位置"这一个点时，对手从视野外接近时我们
// **以为自己是安全的**——危险估计彻底失真。这在对 hunter（每 3 回合扫描一次、
// 从 3 格外开火）时是致命的。
//
// 模型与 rl 管线的 obs_builder.h 一致，是一条廉价但有效的近似：
//   ① 扩张：敌方一个行动阶段最多走 3 格 → 按 BFS ≤3 扩张（绕障碍、不占我方格）
//   ② 证伪：我方当前 T 形视野内确认无人的格子，从信念中剔除
//   ③ 塌缩：直接看见或雷达扫描到时，信念收缩为那一个格子
//   ④ 重置：被击杀的对手会瞬移回出生点，此时信念直接重置到出生点
//
// 这里刻意不引入概率：只维护"可能在这里"。搜索里用的是"代表位置 + 评估时
// 对信念取期望"，比完整 expectimax 便宜得多，且足以修掉上面那个失真。

#pragma once

#include "sim/rules.h"

#include <array>

namespace sim {

inline constexpr int kCells = kBoardSize * kBoardSize;

inline int cell_index(int x, int y) { return y * kBoardSize + x; }

struct Belief {
    std::array<unsigned char, kCells> cells{};

    static bool in_bounds(const Pos& p) {
        return p.x >= 0 && p.x < kBoardSize && p.y >= 0 && p.y < kBoardSize;
    }

    void clear() { cells.fill(0); }
    bool has(int x, int y) const { return cells[cell_index(x, y)] != 0; }
    bool has(const Pos& p) const { return in_bounds(p) && has(p.x, p.y); }
    void set(int x, int y) { cells[cell_index(x, y)] = 1; }
    void set(const Pos& p) {
        if (in_bounds(p)) set(p.x, p.y);
    }
    void reset_to(const Pos& p) {
        clear();
        set(p);
    }
    int count() const;
    bool empty() const { return count() == 0; }

    // 距 p 最近的成员。用于在信念不唯一时给搜索挑一个"代表位置"。
    // 取最近的 = 最保守（离我们最近的敌人最危险）。
    Pos nearest_to(const Pos& p) const;
};

// 敌方一个行动阶段最多移动 steps 格：BFS 扩张（绕障碍、不占我方所在格）
void dilate(Belief& b, int steps, const Pos& my_pos, const std::vector<Pos>& obstacles);

// 剔除障碍格与我方所在格（不可能是敌人）
void prune_impossible(Belief& b, const Pos& my_pos, const std::vector<Pos>& obstacles);

// 视野证伪：当前 T 形视野内、确认无人的格子从信念中剔除。
// 前提：我方能看见的格子若有人，必然已经看到了。
void prune_by_vision(Belief& b, const Sentry& me, const std::vector<Pos>& obstacles);

// 信念中是否有能被我们直接命中的格子（用于判断"这一枪值不值得盲开"）
bool any_in_fire_lane(const Belief& b, const Sentry& me,
                      const std::vector<Pos>& obstacles);

// 信念是否全部落在我们的火力范围内（此时开火必定命中，与猜位置无关）
bool all_in_fire_lane(const Belief& b, const Sentry& me,
                      const std::vector<Pos>& obstacles);

// 用「对手上一阶段占了点」这条证据收窄信念：把不在得分区里的格子剔掉。
//
// 依据：对手分数 +1 且这不是击杀的 +2，说明他行动阶段**结束时在得分区里**。
// 这是一条免费且极强的约束（得分区只有 5 格），而我们此前完全没用过对手的
// 分数。调用方必须先做扩张——扩张覆盖的正是他那个阶段的移动，先扩张再求交
// 才是正确顺序。
//
// 返回值＝是否真的收窄了。**矛盾时保留原信念**：这条推断依赖分数 delta 的
// 算术（要减掉击杀 +2、超时 +1），算错一次就会把真位置剔出去，而
// 「support 必覆盖真位置」是信念的基本契约，不能破。
bool intersect_zone(Belief& b, const std::vector<Pos>& zones);

} // namespace sim

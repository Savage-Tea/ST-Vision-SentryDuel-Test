// brain/eval.h —— 局面评估函数（阶段①的 policy model）
//
// 全部在本方局部坐标框架下计算：我方恒为 'R'（出生点 (0,0)、初始朝 E），
// 对手恒为 'B'。引擎已经对蓝方做过 180° 镜像，所以这个框架对红蓝双方一致，
// 不需要任何按颜色的分支——这是把"视角归一化"交给引擎、我们只写一份策略的关键。
//
// 这个接口刻意保持为「纯数据 + 一个纯函数」，阶段③ 可以整体换成神经网络
// （同样的输入输出形状），三个阶段因此能递进而不是重写。

#pragma once

#include "sim/rules.h"

namespace brain {

// 评估权重。战略核心不是"我打中了吗"，而是**占点时间差 + 剥夺对手占点时间**：
// 得分区 5 格且双方各自结算，占点是稳定 +1/回合；击杀的 +2 只是小头，
// 真正的大价值是把对手打回出生点、让他 3-4 个回合拿不到占点分。
struct Weights {
    double w_diff = 1.00;    // 分差（已含本回合我方占点分）
    double w_zone = 0.80;    // 当前位于得分区（未来价值：下回合大概率还能 +1）
    double w_dist = 0.40;    // 距得分区的距离优势（每格）
    double w_threat = 0.40;  // 对手在我火力通道内（我下回合可击杀）
    double w_danger = 1.30;  // 我在对手火力通道内（下回合会被击杀）
    double w_ready = 0.30;   // 我方火力就绪
    // 行动是有限资源（每回合 3 次）。这一项很小，只用来打破"浪费行动"
    // 与"不浪费"之间的平局——没有它，搜索会选出 TURN 来 TURN 去这类
    // 状态不变却照常消耗额度的方案（它们的评估值与什么都不做完全相同）。
    double w_waste = 0.02;
};

// 对手的 fire_cd / scan_cd 在观测里恒为 -1（引擎不暴露），
// 搜索中必须假设一个值。取 0（随时可开火）是保守选择。
inline constexpr int kAssumedEnemyFireCd = 0;

// 到最近得分区格子的曼哈顿距离（不在区内时 > 0）
int dist_to_zone(const Pos& p, const std::vector<Pos>& zones);

// 火力通道是否通畅（复用 sim 的通道遮挡规则：障碍挡住其后格子）
bool lane_clear(const Sentry& shooter, const Pos& target,
                const std::vector<Pos>& obstacles);

// 任意朝向下能否打到目标。用于「敌方朝向未知」时的悲观估计：
// 对手的朝向若已过期/未知，就不能假定自己是安全的。
bool lane_clear_any_facing(const Sentry& shooter, const Pos& target,
                           const std::vector<Pos>& obstacles);

// 局面评估，分数越高对我方越有利
double evaluate(const sim::State& s, const Weights& w);

} // namespace brain

// brain/eval.h —— 局面评估函数（阶段①的 policy model）
//
// 全部在本方局部坐标框架下计算：我方恒为 'R'（出生点 (0,0)、初始朝 E），
// 对手恒为 'B'。引擎已经对蓝方做过 180° 镜像，所以这个框架对红蓝双方一致，
// 不需要任何按颜色的分支——这是把"视角归一化"交给引擎、我们只写一份策略的关键。
//
// 这个接口刻意保持为「纯数据 + 一个纯函数」，阶段③ 可以整体换成神经网络
// （同样的输入输出形状），三个阶段因此能递进而不是重写。

#pragma once

#include "sim/belief.h"
#include "sim/rules.h"

namespace brain {

// 评估权重。战略核心不是"我打中了吗"，而是**占点时间差 + 剥夺对手占点时间**：
// 得分区 5 格且双方各自结算，占点是稳定 +1/回合；击杀的 +2 只是小头，
// 真正的大价值是把对手打回出生点、让他 3-4 个回合拿不到占点分。
struct Weights {
    double w_diff = 1.00;    // 分差（已含本回合我方占点分）
    double w_zone = 0.60;    // 当前位于得分区（调优：降低以减少激进抢区）
    double w_dist = 0.30;    // 距得分区的距离优势（调优：降低以减少走廊冲动）
    double w_threat = 0.25;  // 对手在我火力通道内（调优：降低以减少冒进）
    double w_danger = 2.00;  // 我在对手火力通道内（下回合会被击杀）
    // 【0.6 = 实测最优】网格扫描 + 600 局验证（2026-09-19）：
    // 0.3 → vs hunter 0.703；0.6 → vs hunter 0.757（红 75.3%，最强红数据）。
    // 代价是对 baseline 0.753 -> 0.730（余量仍足）。取舍方向对准平台暴露的
    // hunter 型短板。
    //
    // 【定版 0.3】平台复测：0.6+400ms 预算的组合包排名掉一名——400ms 预算
    // 会撞爆"整局 30s 含对手思考"的子进程时限（判负）；0.6 单独则是镜像
    // 取舍（baseline +1.7 / hunter -2.8 @600 局）。0.3 的 hunter 分最高且
    // 是平台上拿到"略有提升"的那一版。
    // 【0.45 = 实测最优】中点+ready0.45 配置（2026-09-20 200 局验证）：
    // 4 象限全赢：baseline 红 49%/蓝 100%，hunter 红 75%/蓝 77%。
    // 双色双对手全过的唯一配置。
    double w_ready = 0.45;   // 我方火力就绪
    // 行动是有限资源（每回合 3 次）。这一项很小，只用来打破"浪费行动"
    // 与"不浪费"之间的平局——没有它，搜索会选出 TURN 来 TURN 去这类
    // 状态不变却照常消耗额度的方案（它们的评估值与什么都不做完全相同）。
    double w_waste = 0.02;

    // 「信念推测出的危险」相对「锚点处确定的危险」的折扣。
    // 【0.75 = 实测最优】蓝方负局回放诊断（24 次死亡，58% 死于视野外枪线）
    // 指向 0.25 的折扣过猛，网格扫描 + 600 局验证（2026-09-18）：
    //   0.25 → vs baseline 0.500（蓝 0%）、vs hunter 0.400
    //   0.75 → vs baseline 0.753（蓝 50.7%）、vs hunter 0.703
    // "看不见的危险"近乎全额计入后，走廊盲死循环消失。
    double w_uncertain = 0.75;

    // CD 时序不对称（#14 的机制）：引擎的 CD 递减只发生在蓝方阶段后，
    // 同样一枪红方要熬 3 个相位、蓝方 1 个；对手可乘虚窗口红 2 蓝 1。
    // 我方 fire_cd>0（打不了还手）时，火力通道内的代价按此比例放大。
    //
    // 【默认 1.0 = 关闭】上线实测（2026-09-16 打榜）：开启后红方表现明显
    // 变差；本地对照里 scale=100 时红方直接输给 det_ai（过度谨慎），
    // 且 2.0 的局部收益在噪声内（200 局 σ≈4.9，+7 局仅 1.4σ）。
    // 机制本身未被证伪，但作为默认值不成熟。实验用 ST_W_DANGER_RED=2.0。
    double w_danger_red_scale = 1.0;
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

// 基于敌方**可能位置集合**的危险/机会统计。
//
// 这里刻意把危险拆成两档，因为把它们混成一个"信念占比"会造成**不确定性瘫痪**：
// 看不见对手时信念会覆盖几十格，占比饱和成一个常值，于是连必须守的得分区
// 也会被判成危险区，AI 就再也不敢进场了（实测：对 baseline 从 0.525 掉到 0.21）。
//
//   danger_known   —— 最后已知位置（锚点）处就能打到我。这是"他就在那儿瞄着我"，
//                     近乎确定的击杀，给全权重。
//   danger_unknown —— 信念中其余位置可能打到我。这是推测，给 w_uncertain 折扣。
struct ThreatStats {
    double danger_known = 0.0;   // 锚点处的威胁：0 或 1
    double danger_unknown = 0.0; // 信念中除锚点外的格子能打到我的占比
    double threat_prob = 0.0;    // 信念中落在我方火力范围内的格子占比
};

ThreatStats threat_stats(const sim::State& s, const sim::Belief& belief);

// 局面评估，分数越高对我方越有利。belief 为敌方可能位置集合。
// acts_first_world：我方在世界坐标里是否先手（执红）。局部框架抹平了颜色，
// 但 CD 递减只发生在蓝方阶段后，火炮未就绪时的暴露代价红蓝不对称——
// 见 Weights::w_danger_red_scale。传 true 即旧行为（对称）。
double evaluate(const sim::State& s, const Weights& w, const sim::Belief& belief,
                bool acts_first_world);

// 从任意一方的视角评估（MCTS 的负极大值回传需要它：
// 树里既有我方走子的节点也有对手走子的节点，叶值必须按"轮到谁"取视角）。
// 我方先手 = 红 ⟹ 对手后手；反之亦然。
double evaluate_for(const sim::State& s, const Weights& w, const sim::Belief& belief,
                    char side, bool acts_first_world);

} // namespace brain

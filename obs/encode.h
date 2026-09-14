// obs/encode.h —— 观测编码（自对弈训练与真机部署共用同一份）
//
// 这是整个训练管线里**最容易出致命错误**的地方：如果自对弈记录的观测与
// 部署时喂给网络的观测有任何一位不一致，策略就是在错误的输入上训练的
// ——而且不会报错，只会表现为"训不出来"。所以编码只有这一份实现，
// 两侧都调它。
//
// 输入必须是**局部视角**状态（我方落在 red 槽位，见 sim::to_local）：
// 这样红蓝双方共用同一个观测布局，网络不需要学两套。
//
// 布局（行主序，索引 = plane * 49 + y * 7 + x）：
//   平面 0  障碍
//   平面 1  得分区
//   平面 2  我方位置
//   平面 3  敌方**可能位置集合**（多热；这是部分可观测下的关键特征）
//   平面 4  敌方当前直接可见位置
//   平面 5  敌方最后已知位置（信念锚点）
//   平面 6  我方当前火力覆盖格
//   平面 7  危险格：信念中能从那里打到我方的格子
// 标量 18 个：朝向 one-hot(4)、fire_cd、scan_cd、双方得分、回合、已用额度、
//             免费转向、敌方朝向 one-hot(5，含未知)、可见性、信念规模

#pragma once

#include "sim/belief.h"
#include "sim/rules.h"

namespace obs {

inline constexpr int kPlanes = 8;
inline constexpr int kCells = sim::kBoardSize * sim::kBoardSize; // 49
inline constexpr int kPlaneDim = kPlanes * kCells;               // 392
inline constexpr int kScalarDim = 18;
inline constexpr int kObsDim = kPlaneDim + kScalarDim;           // 410

inline int cell(int x, int y) { return y * sim::kBoardSize + x; }

// 写满 kObsDim 个 float。local_state 必须是局部视角（我方在 red 槽位）。
void encode(const sim::State& local_state, const sim::Belief& belief,
            bool enemy_visible, int used_by_now, bool free_turn, float* out);

} // namespace obs

// obs/encode_v3.h —— 阶段③ 观测编码（360 维）
//
// 与 obs/encode.h（410 维）**是两套**，不要混用：阶段①/② 继续用旧的，
// 阶段③ 的端到端策略用这一套。两套并存是刻意的——旧路径仍然可部署，
// 新路径要能在同口径下与它对比。
//
// 与旧编码的根本差别：**不含手工信念**。
// 旧编码里有三个平面是"我们推断出来的东西"（敌方可达集、信念锚点、危险格），
// 外加一个"信念规模"标量。阶段③ 把这些全部拿掉，交给 RNN 的 hidden state 去学。
// 所以这个函数的签名里**没有 sim::Belief 参数**——这不是省略，是设计。
//
// 输入必须是**局部视角**状态（我方落在 red 槽位，见 sim::to_local）：
// 引擎已对蓝方做 180° 镜像，所以红蓝双方共用同一个网络、同一份观测布局，
// 颜色对称性由这个约定保证，不需要网络学两套。
//
// 布局（行主序，索引 = plane * 49 + y * 7 + x）：
//   平面 0  障碍
//   平面 1  得分区
//   平面 2  我方位置
//   平面 3  我方火力覆盖格（由我方位置+朝向完全确定）
//   平面 4  我方视野覆盖格（同上，可直接观测的几何，不是记忆）
//   平面 5  敌方**当前可见**位置（不可见时全 0）
//   平面 6  敌方 last_known_pos（引擎提供，属规则信息）
//
// 标量 17 个：我方朝向 one-hot(4)、fire_cd、scan_cd、双方得分、回合号、
//             本回合已用额度、免费转向、敌方可见性、敌方朝向 one-hot(5，含未知)

#pragma once

#include "sim/rules.h"

namespace obs {

inline constexpr int kPlanesV3 = 7;
inline constexpr int kCellsV3 = sim::kBoardSize * sim::kBoardSize; // 49
inline constexpr int kPlaneDimV3 = kPlanesV3 * kCellsV3;           // 343
inline constexpr int kScalarDimV3 = 17;
inline constexpr int kObsDimV3 = kPlaneDimV3 + kScalarDimV3;       // 360

// 具名平面下标。任何消费方（轨迹写入、调试打印、可视化）都用这些常量，
// 不要写裸数字——平面顺序一旦调整，裸数字会静默错位。
inline constexpr int kPlaneObstacles = 0;
inline constexpr int kPlaneZones = 1;
inline constexpr int kPlaneMyPos = 2;
inline constexpr int kPlaneMyFire = 3;
inline constexpr int kPlaneMyVision = 4;
inline constexpr int kPlaneEnemyVisible = 5;
inline constexpr int kPlaneEnemyAnchor = 6;

inline int cell_v3(int x, int y) { return y * sim::kBoardSize + x; }

// 写满 kObsDimV3 个 float。local_state 必须是局部视角（我方在 red 槽位）。
//
// 【重要】调用方**不要**先把信念写回 local_state.blue.last_known_pos。
// 旧路径的 agent/act.cpp 会调 apply_belief() 用我们的推断覆盖引擎给的值，
// 那条路径要用旧编码。这里要的是引擎原样的 last_known_pos。
void encode_v3(const sim::State& local_state, bool enemy_visible,
               int used_by_now, bool free_turn, float* out);

} // namespace obs

// brain/opening_book.h —— 开局书查表（全局最优解的部署形态）
//
// solve_ab --opening-book 导出的紧凑二进制，编译进 .so。
// turn < N（导出时的 book_turns）时，双方位置公开（全知=部分信息），
// 书里的最优手在部署时是**精确**的——不存在信息缺口。
// turn ≥ N 或查不到时，回退 phase① 搜索（调用方负责）。
//
// 键 = (红姿态, 蓝姿态, 回合, 行动方, 已用额度, 免费旗标, 分差)，
// 与求解器 make_key 的字段语义一致（世界坐标，红蓝各自的真实姿态）。

#pragma once

#include "sentry_duel.h"

namespace brain {

bool book_available();
int book_turns(); // 覆盖到的最大回合数

// 查表。返回最优动作下标（brain::kActionDim 空间，7=收手），-1 = 未命中。
// freef：bit1 = 红方免费转向，bit0 = 蓝方免费转向。
int book_lookup(int pr, int pb, int turn, int side, int ac, int freef, int diff);

} // namespace brain

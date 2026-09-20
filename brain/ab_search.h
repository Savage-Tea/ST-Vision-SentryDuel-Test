// brain/ab_search.h —— αβ + 置换表搜索（brain::ab_search_turn）
//
// 替换 brain/search.cpp 的 dfs_our/dfs_opp 结构。
// 核心改进：
//   · 标准 αβ 剪枝：统一框架下我方 max / 对手 min
//   · 置换表：同一状态不重复展开（turn 0-2 状态高度重叠）
//   · 正确的 end_round 建模：CD 递减 + 回合推进
//   · 可调深度：ST_AB_DEPTH=2（默认）= 两回合
//
// 接口与 brain::search_turn 完全一致（TurnInput → Plan），可直接替换。

#pragma once

#include "brain/search.h"

namespace brain {

Plan ab_search_turn(const TurnInput& in, const Weights& w,
                    std::chrono::steady_clock::time_point deadline,
                    SearchStats* stats);

} // namespace brain

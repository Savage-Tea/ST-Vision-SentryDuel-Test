// sim/state.h —— 前向模型的状态定义
//
// 设计取舍：这里 #include "sentry_duel.h"（选手 ABI 头）而不是重定义 Pos/Sentry。
// 理由是让 sim 的字段布局与引擎 Board 逐字段同构，差分测试才能机械地逐步比对，
// 也杜绝两边字段悄悄漂移。注意这不引入对引擎**实现**的任何依赖——
// sentry_duel.h 只是数据结构声明，链接期不需要 libsentry_duel_engine。
//
// 命名沿用引擎约定：Sentry::last_known_pos / last_known_facing 在引擎内部的
// 真实世界棋盘上就是「真实位置 / 真实朝向」（见 engine/src/board.cpp:141）。
// 「最后已知」的语义只体现在传给选手的视图里，这里保持一致以免混淆。

#pragma once

#include "sentry_duel.h" // 只读引用：Pos / Sentry 的字段布局

#include <vector>

namespace sim {

inline constexpr int kBoardSize = 7;

// 一局内的行动编码，与引擎 engine_internal.h 的约定一致
enum Action { kMove = 0, kTurn = 1, kFire = 2, kScan = 3 };

// apply_action 的返回，对应引擎的 ActionOutcome
struct Outcome {
    bool success = false;
    bool hit = false;

    friend bool operator==(const Outcome& a, const Outcome& b) {
        return a.success == b.success && a.hit == b.hit;
    }
    friend bool operator!=(const Outcome& a, const Outcome& b) { return !(a == b); }
};

// 一方对敌方的最后已知情报（对应引擎 Match::Intel）
struct Intel {
    Pos pos{-1, -1};
    char facing = '?';
    bool known = false;
};

// 世界坐标下的完整对局状态（red/blue 的 pos/facing 为真实值）
struct State {
    Sentry red;
    Sentry blue;
    int turn = 0;
    int size = kBoardSize;
    std::vector<Pos> obstacles;
    std::vector<Pos> score_zones;

    Sentry& sentry_for(char side) { return side == 'R' ? red : blue; }
    const Sentry& sentry_for(char side) const { return side == 'R' ? red : blue; }
};

} // namespace sim

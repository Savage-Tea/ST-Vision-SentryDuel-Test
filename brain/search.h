// brain/search.h —— 阶段① 的本回合搜索
//
// 每回合最多 3 个行动且**立即结算**，所以这是一个有限深度、低分支的序列决策问题：
//   第一层：枚举我方 0..3 个行动（模拟引擎的额度与免费转向语义）
//   结算  ：我方行动阶段结束 → end_side_turn('R') → 本回合的占点分入账
//   第二层：枚举对手 0..3 个行动，取**对我最不利**的回应（minimax）
//   评估  ：brain::evaluate
//
// 第二层的作用是把"我把敌人打回出生点"和"我暴露在他的火力下"这类
// 跨回合后果算进来——只看自己一步的贪心会严重低估击杀的价值。

#pragma once

#include "brain/eval.h"

#include <chrono>

namespace brain {

inline constexpr int kMaxActions = 3;

// 对手回应细化的候选上限（按未细化评估排序后取前 K 个），用来界定最坏耗时
inline constexpr int kRefineTopK = 48;

struct Plan {
    int count = 0;
    int actions[kMaxActions] = {0, 0, 0};
    char args[kMaxActions] = {0, 0, 0};
    double value = 0.0;
    bool valid = false;
};

// 搜索输入。全部在「我方为 'R'、对手为 'B'」的局部坐标框架下。
struct TurnInput {
    static constexpr int kMaxBans = 6;

    sim::State state;
    int used_by_now = 0;             // 本回合已经消耗掉的额度（分叉重规划时 > 0）
    bool free_turn_available = false; // 出生点免费转向是否仍可能可用
    bool enemy_visible = false;      // 本回合是否真的看得见（含 SCAN 临时视野）

    // 本 act 内已被真机证明失败的行动。必须屏蔽掉，否则重规划会反复选中
    // 同一个动作（典型情形：移动目标格被我们看不到的对手占着）。
    int ban_actions[kMaxBans] = {0, 0, 0, 0, 0, 0};
    char ban_args[kMaxBans] = {0, 0, 0, 0, 0, 0};
    int ban_count = 0;

    void ban(int action, char arg) {
        if (ban_count >= kMaxBans) return;
        ban_actions[ban_count] = action;
        ban_args[ban_count] = arg;
        ++ban_count;
    }
    bool is_banned(int action, char arg) const {
        for (int i = 0; i < ban_count; ++i) {
            if (ban_actions[i] == action && ban_args[i] == arg) return true;
        }
        return false;
    }
};

struct SearchStats {
    long long our_nodes = 0;
    long long opp_nodes = 0;
    long long refined = 0;
    bool time_exhausted = false; // 超预算，部分叶子未做对手回应
};

using Deadline = std::chrono::steady_clock::time_point;

inline bool out_of_time(const Deadline& dl) {
    return std::chrono::steady_clock::now() >= dl;
}

// 返回的 Plan 中 turn 参数是**局部坐标**朝向，执行时需按颜色转成世界坐标。
Plan search_turn(const TurnInput& in, const Weights& w, const Deadline& deadline,
                 SearchStats* stats = nullptr);

} // namespace brain

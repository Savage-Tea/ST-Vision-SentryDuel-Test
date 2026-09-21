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

#include "brain/actions.h"
#include "brain/eval.h"

#include <chrono>

namespace brain {

inline constexpr int kMaxActions = 3;

// 叶评估噪声（σ）。0 = 关闭 —— 线上部署就是这一档，行为保持确定性。
//
// 存在的理由是**风格池需要随机化的对手**：当前池子里 10/13 是确定性的
// （每个对手只有 1~2 种对局），于是"2000 局胜率"是运气而不是统计量，
// 池子也就不可能测出"对多样化对手的稳定性"——而那恰恰是平台在考的。
// 叶值加噪声后，同一份策略在不同进程里会走出不同的棋，胜率重新变回统计量。
void set_leaf_noise(double sigma);

// 对手回应细化的候选上限（按未细化评估排序后取前 K 个），用来界定最坏耗时
inline constexpr int kRefineTopK = 48;
// 深模式（两回合前瞻）的单叶节点预算。超出即回退浅层值——部分探索的
// 极小值对对手乐观，会骗我方走险手，宁可不用。
inline constexpr long long kDeepNodeBudget = 150000;

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
    // 敌方可能位置集合。搜索会随我方行动后的视野证伪不断收缩它，
    // 所以"走到能看见更多地方的位置"本身就变得有价值。
    sim::Belief belief;
    int used_by_now = 0;             // 本回合已经消耗掉的额度（分叉重规划时 > 0）
    bool free_turn_available = false; // 出生点免费转向是否仍可能可用
    bool enemy_visible = false;      // 本回合是否真的看得见（含 SCAN 临时视野）
    // 我方在世界坐标里是否先手（执红）。局部框架抹平了颜色，但引擎的 CD
    // 递减只发生在蓝方阶段后，火炮未就绪时的暴露代价红蓝不对称（2 窗口 vs
    // 1 窗口）——评估需要知道这一项。见 eval.h 的 w_danger_red_scale。
    bool acts_first_world = true;
    // 值蒸馏网络（ST_VALUE_NET=1 启用）：叶评估替换为 V* 蒸馏网络的输出。
    // 网络不可用时自动回退手工评估。scale 把 [-1,1] 的值映射到与终局
    // 奖励可比的量级（终局 ±1e6，手工评估 ±10s）。
    bool use_value_net = false;
    double value_scale = 30.0;
    // 两回合前瞻（深模式）。0 = 关闭；1 = 在对手回应之后再看一轮我方+对手。
    int deep_turns = 0;
    // 对手建模开关：true = minimax（默认），false = 纯 eval 贪心（无对手回应层）
    bool do_opp_model = true;
    // 对手"刚刚开过火、正处在 CD 无力期"的推断：被击中时，打我们的人必然
    // 刚开火。引擎不暴露对手 CD（观测恒 -1，默认保守假设随时可开火），
    // 但这次击中本身泄露了信息。窗口按 CD 时序不对称折算：
    //   我方执蓝（打我们的是红方）：红开火后 2 个窗口无力 → 安全到 turn+1
    //   我方执红（打我们的是蓝方）：蓝开火后 1 个窗口无力 → 安全到 turn
    // 窗口内搜索会允许我们压近（对手在树里无法开火），过期后回退保守假设。
    int opp_defenseless_until = -1;

    // 本 act 内已被真机证明失败的行动。必须屏蔽掉，否则重规划会反复选中
    // 同一个动作（典型情形：移动目标格被我们看不到的对手占着）。
    Bans bans;

    void ban(int action, char arg) { bans.add(action, arg); }
    bool is_banned(int action, char arg) const { return bans.has(action, arg); }
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

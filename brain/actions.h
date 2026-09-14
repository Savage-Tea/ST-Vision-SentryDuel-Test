// brain/actions.h —— 共用的「行动模型」
//
// phase① 的本回合搜索与 phase② 的 MCTS 共用这一层，避免把额度语义、
// 免费转向、候选剪枝这些规则复制成两份（引擎里正是这么埋下漂移隐患的）。
//
// 全部在「我方为 'R'、对手为 'B'」的局部坐标框架下工作。

#pragma once

#include "brain/eval.h"
#include "sim/belief.h"
#include "sim/rules.h"

#include <vector>

namespace brain {

inline constexpr int kMaxActionsPerTurn = 3;
inline constexpr int kMaxBans = 6;

struct Cand {
    int action = sim::kMove;
    char arg = 0;
};

// 本 act 内已被真机证明失败的行动。必须屏蔽，否则重规划会反复选中同一个
// （典型情形：移动目标格被我们看不到的对手占着）。
struct Bans {
    int actions[kMaxBans] = {0, 0, 0, 0, 0, 0};
    char args[kMaxBans] = {0, 0, 0, 0, 0, 0};
    int count = 0;

    void add(int action, char arg) {
        if (count >= kMaxBans) return;
        actions[count] = action;
        args[count] = arg;
        ++count;
    }
    bool has(int action, char arg) const {
        for (int i = 0; i < count; ++i) {
            if (actions[i] == action && args[i] == arg) return true;
        }
        return false;
    }
};

// 生成某方的候选行动。剪掉的都是"引擎允许但纯属浪费"的分支：
//   · 转向到当前朝向且不免费 —— 成功、消耗额度、状态不变
//   · 火力/雷达未就绪 —— 必然失败
//   · 看不见对手时开火 —— 除非信念**全部**落在火力范围内（那这一枪必中）
//
// belief 只对"开火"判定有意义，给对手生成候选时传 nullptr。
void collect_candidates(const sim::State& s, char side, bool free_turn,
                        bool can_see_enemy, const Bans* bans,
                        const sim::Belief* belief, std::vector<Cand>& out);

// 应用一个行动，并复刻引擎 Match::do_action 的额度与免费转向语义。
// 失败返回 false（不消耗额度，也不会改动 s）。hit 非空时回填是否命中。
bool apply_step(sim::State& s, char side, const Cand& c, int& used, bool& free_turn,
                bool* hit = nullptr);

// 结束一方的行动阶段：先结算该方的占点分；
// 后手（'B'）的阶段结束即整回合结束，所以顺带推进 CD 与回合数。
void end_phase(sim::State& s, char side);

// 终局判定（与引擎一致）：
//   回合数 < 20         —— 未完
//   20 ≤ 回合数 < 25    —— 比分不等即终局（加时赛在每个回合末就决出）
//   回合数 ≥ 25         —— 终局（平局）
bool is_terminal(const sim::State& s);

// 终局分数（从 perspective 视角，胜 +kWinValue / 负 -kWinValue / 平 0）
inline constexpr double kWinValue = 1.0e6;
double terminal_value(const sim::State& s, char perspective);

} // namespace brain

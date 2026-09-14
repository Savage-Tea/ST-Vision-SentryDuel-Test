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

// 结束本方行动阶段的合成动作（"收手"）
inline constexpr int kStopAction = -1;

// —— 固定动作空间 ——
// 策略网络的输出维度。把 (动作, 转向参数) 压成一个定长索引，
// 这样策略头就是一个 8 维 softmax，训练侧不必处理变长动作。
//   0=move  1..4=turn N/E/S/W  5=fire  6=scan  7=收手
inline constexpr int kActionDim = 8;

inline int cand_to_index(const Cand& c) {
    switch (c.action) {
        case sim::kMove: return 0;
        case sim::kFire: return 5;
        case sim::kScan: return 6;
        case sim::kTurn:
            switch (c.arg) {
                case 'N': return 1;
                case 'E': return 2;
                case 'S': return 3;
                case 'W': return 4;
                default: return -1;
            }
        default: return 7; // kStopAction
    }
}

inline Cand index_to_cand(int i) {
    switch (i) {
        case 0: return {sim::kMove, 0};
        case 1: return {sim::kTurn, 'N'};
        case 2: return {sim::kTurn, 'E'};
        case 3: return {sim::kTurn, 'S'};
        case 4: return {sim::kTurn, 'W'};
        case 5: return {sim::kFire, 0};
        case 6: return {sim::kScan, 0};
        default: return {kStopAction, 0};
    }
}

// 局部坐标动作 → 世界坐标。**只在自己直接驱动 sim 时需要**。
// 驱动真实引擎时绝对不要调用：引擎的 turn() 收的就是局部朝向，
// 它会在 Match::do_action 内部自己做镜像。这里再镜像一次就会反向 180°
// ——这个坑已经踩过一次，整局报废。
Cand local_to_world(const Cand& local, char side);

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

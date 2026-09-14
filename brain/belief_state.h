// brain/belief_state.h —— 一方的敌方位置信念（跨回合维护）
//
// 抽成共用模块的原因：**信念是观测的一部分**。部署侧（agent/act.cpp）与
// 自对弈侧各自实现一份的话，一旦漂移，训练输入与部署输入就不一致了
// ——而且不会报错，只会表现为"训不出来"。这类静默错误在这个项目里
// 已经出现过（turn 参数被镜像两次），所以这里只留一份实现。
//
// 全部在该方的**局部框架**下维护（我方恒在 'R' 槽位，见 sim::to_local）。
//
// 生命周期：
//   ① 回合开始：扩张（对手上回合走完一个行动阶段，最多 3 格）
//               → 可见则塌缩 → 仍未知则落到敌方出生点 → 用我方视野证伪
//   ② 我方每次行动后：用新视野证伪（走到能看见更多地方 = 有信息价值）
//   ③ 特例：我方命中 → 对手瞬移回出生点（不属于 ≤3 格移动，扩张覆盖不到）
//           移动失败 → 目标格被对手占着（关于敌位最强的情报）

#pragma once

#include "sim/belief.h"
#include "sim/rules.h"

namespace brain {

struct SideBelief {
    sim::Belief set;                  // 敌方可能位置集合
    Pos anchor{-1, -1};               // 最后已知位置（信念锚点）
    char anchor_facing = '?';
    bool known = false;               // 是否已获得过任何情报
    bool certain = false;             // 是"知情"还是"假设"（决定要不要开雷达）

    void clear();

    // 塌缩到确定的一格
    void collapse(const Pos& p, char facing);

    // 回合开始
    void begin_turn(const sim::State& local, int turn, bool enemy_visible);

    // 我方一次成功行动之后
    void after_action(const sim::State& local);

    // 我方命中
    void on_our_hit(int size);

    // 由推断得到的确切位置（如移动失败）
    void infer_at(const Pos& p);

    // 把锚点写回局部状态，供搜索/推演使用
    void write_anchor_into(sim::State& local) const;
};

} // namespace brain

// brain/mcts.h —— MCTS / PUCT 搜索（跨回合前瞻）
//
// 为什么要有它：phase① 的本回合搜索穷举了我方 3 步 × 对手 3 步，本质只有
// **一个回合**的前瞻；而决定胜负的动力学——击杀 → 对手回出生点 → 被剥夺
// 2~3 个回合的占点分——横跨多个回合。MCTS 用时间预算换取跨回合的深度。
//
// 更重要的定位：这是通往自对弈的**策略改进算子**。先验 P 与价值 V 在这里是
// 可插拔的（现在用手工评估函数），阶段③ 换成神经网络即可，机器不用重写。
//
// 结构：
//   · 节点 = (状态, 敌方信念, 走子方, 已用额度, 是否仍有免费转向)
//   · 选择 = PUCT：Q + c_puct · P · √N / (1 + n)
//   · 扩展 = brain::collect_candidates + 一步估值做 softmax 先验
//   · 叶值 = evaluate_for(状态, 走子方)  —— **不做随机 rollout**：
//            本游戏完全确定性，随机走子到终局几乎没有信息量，直接用评估函数更省样本
//   · 回传 = 负极大值（树里红蓝节点交替，叶值必须按"轮到谁"取视角）
//   · 终局 = ±kWinValue 的大值，保证"赢"压倒任何启发式收益

#pragma once

#include "brain/search.h" // TurnInput / Plan / Deadline / out_of_time

#include <vector>

namespace brain {

struct MctsConfig {
    double c_puct = 1.5;            // 探索常数
    double prior_temperature = 0.5; // 先验 softmax 温度（越小越尖锐）
    int max_depth = 24;             // 最大动作步数（约 3 个回合）
    long long max_nodes = 300000;   // 节点上限（内存与时间双保险）

    // —— 自对弈专用 ——
    // 本游戏完全确定性：双方跑同一个确定性策略会每局下出同一盘棋，
    // 没有多样性就没有学习信号。所以自对弈必须开噪声 + 按分布采样。
    bool add_root_noise = false;    // 根部先验加 Dirichlet 噪声
    double dirichlet_alpha = 0.3;
    double dirichlet_eps = 0.25;
    double sample_temperature = 0.0; // >0 = 按访问次数^(1/T) 采样；0 = 取最大
    unsigned int seed = 12345;
};

struct MctsResult {
    Plan plan;                 // 根处选出的第一步
    double root_value = 0.0;   // 根值（我方视角）
    long long nodes = 0;
    bool time_exhausted = false;
    // 根节点的访问次数分布，映射到固定动作空间（brain::kActionDim）。
    // 这是自对弈训练的策略目标 π —— AlphaZero 配方的核心产物。
    float pi[kActionDim] = {0};
};

MctsResult mcts_search(const TurnInput& in, const Weights& w, const MctsConfig& cfg,
                       const Deadline& deadline);

} // namespace brain

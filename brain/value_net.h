// brain/value_net.h —— 值蒸馏网络：全信息叶评估（阶段① 搜索的可选叶评估器）
//
// 【它是什么】tools/solve_ab.cpp 精确求解出的 V*（完全信息博弈值，三值：
// 红胜/平/红负），被采样成 (全信息状态, 值) 数据集（--dump-values），由
// tools/train_value.py 蒸馏成这个小 MLP。查询发生在**搜索树的叶节点**——
// 那里的状态是全信息的（双方真实位置都在 sim 状态里），所以不存在
// BC 那个"全知教师 → 部分信息学生"的映射缺陷：这里是全知 → 全知。
//
// 【为什么值得】手工评估函数（eval.cpp）编码了"占中心+狙击"——正是颜色
// 不对称的载体。用精确博弈值替换/对照它，#14 与 #15 一次解决。
//
// 【特征契约】与 tools/value_net.py 的 build_features 严格一致（116 维），
// 两侧任何一侧改动都必须同步，train_value.py 会校验维度。
//   平面 0：红方位置 one-hot (49)
//   平面 1：蓝方位置 one-hot (49)
//   标量（18）：红朝向 one-hot(4)、蓝朝向 one-hot(4)、红 fcd/2、蓝 fcd/2、
//              红 scd/3、蓝 scd/3、turn/25、红方行动中(1)、ac/3、
//              free_red(1)、free_blue(1)、diff_red/51
// 输出：V_red ∈ [-1, 1]（红方视角的博弈值），线性输出。
//
// 权重由 tools/train_value.py 导出到 brain/value_weights.h（生成产物）。
// 缺失时 value_net_available() 返回 false，搜索回退手工评估。

#pragma once

#include "sim/rules.h"

namespace brain {

inline constexpr int kValueObsDim = 116;
inline constexpr int kValueH1 = 256;
inline constexpr int kValueH2 = 256;

bool value_net_available();

// 从局部帧叶状态构建特征。swap_colors：
//   false —— s.red 就是世界红方（我方执红）
//   true  —— s.blue 是世界红方（我方执蓝），内部做红蓝对调
// free_red/free_blue 是世界红/蓝双方的免费转向旗标。
void value_features(const sim::State& s, bool red_to_move, bool free_red,
                    bool free_blue, bool swap_colors, float* out);

// 前向：返回 V_red ∈ [-1,1]。不可用时返回 false。
bool value_eval(const float* features, float* v);

} // namespace brain

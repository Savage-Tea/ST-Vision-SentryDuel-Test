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
// 【特征契约 v2 —— 信徒相对 + 信念平面】与 tools/value_net.py 严格一致
// （110 维）。v1 教训：V* 是双方真实位置的阶跃函数，部署叶上对手位置只有
// 信念锚点（噪声输入），在阶梯函数上查噪声 ≈ 抛硬币（0/400）。
// v2 的标签改为 E[V* | 信念集]（dump 端对集合逐个查表求均值——连续函数），
// 输入改为部署时真正拥有的量：
//   平面 0：我方位置 one-hot (49)
//   平面 1：信念平面——我推断的对手可能位置集合 (49)
//   标量（12）：我方朝向 one-hot(4)、我方 fcd/2、我方 scd/3、turn/25、
//              我方行动中(1)、ac/3（恒 0，查询点都在相位边界）、
//              我方免费(1)、对手免费(1)、diff_mine/51
// 输出：我方视角的期望博弈值 ∈ [-1, 1]，线性输出。
// 网络是颜色无关的（训练数据同时覆盖两种持信念方），无需任何 swap/取反。
//
// 权重由 tools/train_value.py 导出到 brain/value_weights.h（生成产物）。
// 缺失时 value_net_available() 返回 false，搜索回退手工评估。

#pragma once

#include "sim/belief.h"
#include "sim/rules.h"

namespace brain {

inline constexpr int kValueObsDim = 110;
inline constexpr int kValueH1 = 256;
inline constexpr int kValueH2 = 256;

bool value_net_available();

// 从局部帧叶状态构建特征。s.red = 我方；belief = 我方对对手位置的信念
// （搜索树现成有）。me_to_move：查询点上是否轮到我方行动。
void value_features(const sim::State& s, const sim::Belief& belief,
                    bool me_to_move, bool my_free, bool opp_free, float* out);

// 前向：返回 V_red ∈ [-1,1]。不可用时返回 false。
bool value_eval(const float* features, float* v);

} // namespace brain

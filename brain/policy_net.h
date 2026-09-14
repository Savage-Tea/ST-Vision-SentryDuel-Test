// brain/policy_net.h —— 阶段③ 循环策略网络的前向推理（GRU）
//
// 权重由 tools/train_ppo.py 导出成 brain/policy_weights.h（C 数组，编译进 .so）。
// 与 brain/net.h 一样是生成产物：没有它也能编译，此时 policy_available() 返回
// false，上层应当回退——这样仓库在"还没训过"的状态下依然可以构建和跑。
//
// 观测布局见 obs/encode_v3.h。输入是**我方视角**的观测。
//
// 【为什么没有 torch 运行时也要手写 GRU】
// 部署的 .so 里没有 libtorch，只有编译进去的权重数组。GRU 的前向是十行数学，
// 手写比引入一个推理框架划算得多。数值一致性由测试保证（见 tests/difftest_policy.cpp）。

#pragma once

#include "obs/encode_v3.h"

namespace brain {

// 网络输入维度必须与观测编码一致——不一致就是静默训练在错误输入上，
// 所以这里用编译期断言钉死，而不是靠人记。
inline constexpr int kPolicyObsDim = 360;
inline constexpr int kPolicyEnc = 128;    // 观测编码器输出
inline constexpr int kPolicyHidden = 256; // GRU 隐状态维度
inline constexpr int kPolicyActDim = 8;   // 与 brain::kActionDim 一致

static_assert(kPolicyObsDim == obs::kObsDimV3,
              "policy_net 的输入维度与 obs::encode_v3 不一致");

// 是否编译进了训练好的权重
bool policy_available();

// 把隐状态清零。**新对局开始时必须调用**，否则上一局的记忆会污染这一局。
void policy_reset_hidden(float* hidden);

// 单步前向：读 obs 与当前 hidden，写回新 hidden、策略 logits、价值。
//
// hidden 是**调用方持有**的 kPolicyHidden 个 float（红蓝双方、以及同一进程里
// 跑的每一局，各自一份，绝不能共用）。
//
// value 是**线性输出，没有 tanh**。这是与 brain/net.h 的实质性差别：
// 阶段③ 的奖励是每回合 Δ 分差累积，回报量级可达 ±20，tanh 会长期饱和。
//
// 无权重时返回 false，且不写入任何输出。
bool policy_forward(const float* obs, float* hidden, float* policy_logits,
                    float* value);

} // namespace brain

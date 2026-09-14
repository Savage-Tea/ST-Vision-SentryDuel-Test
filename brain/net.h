// brain/net.h —— 策略/价值网络的前向推理
//
// 权重由 tools/train_mlp.py 导出成 brain/net_weights.h（C 数组，编译进 .so）。
// 该头文件是**生成产物**（见 .gitignore）：没有它也能编译，此时
// net_available() 返回 false，上层回退到手工评估函数——这样仓库在
// "还没训过"的状态下依然可以构建和跑。
//
// 观测布局见 obs/encode.h；网络输入是**我方视角**的观测
// （自对弈时每条样本都记录在产生它的那一方的视角下）。

#pragma once

namespace brain {

// 是否编译进了训练好的权重
bool net_available();

// obs[kObsDim] → 策略 logits[kActionDim] + 价值（标量，tanh 输出，范围 [-1,1]）
// 无权重时返回 false，且不写入输出
bool net_forward(const float* obs, float* policy_logits, float* value);

// 便捷封装：把 logits 过 softmax 得到先验（对数值稳定性做了 max 平移）
bool net_prior_value(const float* obs, float* prior, float* value);

} // namespace brain

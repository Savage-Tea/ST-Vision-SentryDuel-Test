#include "brain/policy_net.h"

#include <cmath>
#include <cstring>

#if __has_include("brain/policy_weights.h")
#include "brain/policy_weights.h"
#define SENTRY_HAS_POLICY 1
#else
#define SENTRY_HAS_POLICY 0
#endif

namespace brain {

bool policy_available() { return SENTRY_HAS_POLICY != 0; }

#if SENTRY_HAS_POLICY

// 导出头文件里的维度必须与这里一致。改了网络结构却忘了重新导出权重，
// 会在这里编译失败，而不是拿错位的权重跑出静默的错误结果。
static_assert(kPvObsIn == kPolicyObsDim, "policy_weights.h 的输入维度与观测不一致");
static_assert(kPvEnc == kPolicyEnc, "policy_weights.h 的编码器维度不一致");
static_assert(kPvHidden == kPolicyHidden, "policy_weights.h 的隐状态维度不一致");
static_assert(kPvAct == kPolicyActDim, "policy_weights.h 的动作维度不一致");

namespace {

inline float sigmoid(float v) { return 1.0f / (1.0f + std::exp(-v)); }

// y = relu(W x + b)，W 按行主序 [out][in]
void linear_relu(const float* w, const float* b, int out, int in, const float* x,
                 float* y) {
    for (int o = 0; o < out; ++o) {
        float acc = b[o];
        const float* row = w + static_cast<std::size_t>(o) * static_cast<std::size_t>(in);
        for (int i = 0; i < in; ++i) acc += row[i] * x[i];
        y[o] = acc > 0.0f ? acc : 0.0f;
    }
}

} // namespace

#endif // SENTRY_HAS_POLICY

void policy_reset_hidden(float* hidden) {
    std::memset(hidden, 0, sizeof(float) * kPolicyHidden);
}

bool policy_forward(const float* obs, float* hidden, float* policy_logits,
                    float* value) {
#if SENTRY_HAS_POLICY
    // thread_local：自对弈是多线程的，复用缓冲区避免每步分配。
    // hidden 本身是调用方传入的，不在这里——它必须跨步存活。
    thread_local float enc[kPolicyEnc];
    thread_local float r[kPolicyHidden];
    thread_local float z[kPolicyHidden];
    thread_local float n[kPolicyHidden];

    linear_relu(kPvEncW, kPvEncB, kPolicyEnc, kPolicyObsDim, obs, enc);

    const std::size_t stride = kPolicyHidden;
    // 权重按 PyTorch nn.GRU 的排布：[W_ir; W_iz; W_in] / [W_hr; W_hz; W_hn]，
    // 每段 kPolicyHidden 行。
    const float* wih_r = kPvGruWih;
    const float* wih_z = kPvGruWih + stride * kPolicyEnc;
    const float* wih_n = kPvGruWih + 2 * stride * kPolicyEnc;
    const float* whh_r = kPvGruWhh;
    const float* whh_z = kPvGruWhh + stride * kPolicyHidden;
    const float* whh_n = kPvGruWhh + 2 * stride * kPolicyHidden;
    const float* bih_r = kPvGruBih;
    const float* bih_z = kPvGruBih + stride;
    const float* bih_n = kPvGruBih + 2 * stride;
    const float* bhh_r = kPvGruBhh;
    const float* bhh_z = kPvGruBhh + stride;
    const float* bhh_n = kPvGruBhh + 2 * stride;

    // r/z 两门只依赖 x 与上一步的 h，先一起算。
    for (int j = 0; j < kPolicyHidden; ++j) {
        float ar = bih_r[j] + bhh_r[j];
        float az = bih_z[j] + bhh_z[j];
        const float* ir = wih_r + static_cast<std::size_t>(j) * kPolicyEnc;
        const float* iz = wih_z + static_cast<std::size_t>(j) * kPolicyEnc;
        const float* hr = whh_r + static_cast<std::size_t>(j) * kPolicyHidden;
        const float* hz = whh_z + static_cast<std::size_t>(j) * kPolicyHidden;
        for (int i = 0; i < kPolicyEnc; ++i) {
            ar += ir[i] * enc[i];
            az += iz[i] * enc[i];
        }
        for (int i = 0; i < kPolicyHidden; ++i) {
            ar += hr[i] * hidden[i];
            az += hz[i] * hidden[i];
        }
        r[j] = sigmoid(ar);
        z[j] = sigmoid(az);
    }

    // 候选隐状态 n：PyTorch 的 GRU 是 h' = (1-z)*n + z*h
    for (int j = 0; j < kPolicyHidden; ++j) {
        float an = bih_n[j];
        const float* inn = wih_n + static_cast<std::size_t>(j) * kPolicyEnc;
        const float* hn = whh_n + static_cast<std::size_t>(j) * kPolicyHidden;
        for (int i = 0; i < kPolicyEnc; ++i) an += inn[i] * enc[i];
        float hh = bhh_n[j];
        for (int i = 0; i < kPolicyHidden; ++i) hh += hn[i] * hidden[i];
        n[j] = std::tanh(an + r[j] * hh);
    }

    for (int j = 0; j < kPolicyHidden; ++j) {
        hidden[j] = (1.0f - z[j]) * n[j] + z[j] * hidden[j];
    }

    // 策略头
    for (int o = 0; o < kPolicyActDim; ++o) {
        float acc = kPvPolB[o];
        const float* row =
            kPvPolW + static_cast<std::size_t>(o) * static_cast<std::size_t>(kPolicyHidden);
        for (int i = 0; i < kPolicyHidden; ++i) acc += row[i] * hidden[i];
        policy_logits[o] = acc;
    }

    // 价值头：线性，**不加 tanh**（见 policy_net.h 的说明）
    float v = kPvValB[0];
    for (int i = 0; i < kPolicyHidden; ++i) v += kPvValW[i] * hidden[i];
    *value = v;
    return true;
#else
    (void)obs;
    (void)hidden;
    (void)policy_logits;
    (void)value;
    return false;
#endif
}

} // namespace brain

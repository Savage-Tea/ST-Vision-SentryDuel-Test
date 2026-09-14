#include "brain/net.h"

#include "brain/actions.h"
#include "obs/encode.h"

#include <cmath>

#if __has_include("brain/net_weights.h")
#include "brain/net_weights.h"
#define SENTRY_HAS_NET 1
#else
#define SENTRY_HAS_NET 0
#endif

namespace brain {

bool net_available() { return SENTRY_HAS_NET != 0; }

#if SENTRY_HAS_NET
namespace {

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
#endif

bool net_forward(const float* obs, float* policy_logits, float* value) {
#if SENTRY_HAS_NET
    // thread_local：自对弈是多线程的，复用缓冲区避免每次分配
    thread_local float h1[kNetH1];
    thread_local float h2[kNetH2];

    linear_relu(kW1, kB1, kNetH1, kNetIn, obs, h1);
    linear_relu(kW2, kB2, kNetH2, kNetH1, h1, h2);

    for (int o = 0; o < kNetAct; ++o) {
        float acc = kBp[o];
        const float* row =
            kWp + static_cast<std::size_t>(o) * static_cast<std::size_t>(kNetH2);
        for (int i = 0; i < kNetH2; ++i) acc += row[i] * h2[i];
        policy_logits[o] = acc;
    }

    float v = kBv[0];
    for (int i = 0; i < kNetH2; ++i) v += kWv[i] * h2[i];
    *value = std::tanh(v);
    return true;
#else
    (void)obs;
    (void)policy_logits;
    (void)value;
    return false;
#endif
}

bool net_prior_value(const float* obs, float* prior, float* value) {
#if SENTRY_HAS_NET
    float logits[kActionDim] = {0};
    if (!net_forward(obs, logits, value)) return false;
    float mx = logits[0];
    for (int i = 1; i < kActionDim; ++i) mx = std::max(mx, logits[i]);
    float sum = 0.0f;
    for (int i = 0; i < kActionDim; ++i) {
        prior[i] = std::exp(logits[i] - mx);
        sum += prior[i];
    }
    if (sum > 0.0f) {
        for (int i = 0; i < kActionDim; ++i) prior[i] /= sum;
    }
    return true;
#else
    (void)obs;
    (void)prior;
    (void)value;
    return false;
#endif
}

} // namespace brain

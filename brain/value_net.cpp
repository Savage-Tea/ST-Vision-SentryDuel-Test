#include "brain/value_net.h"

#include <cmath>
#include <cstring>

#if __has_include("brain/value_weights.h")
#include "brain/value_weights.h"
#define SENTRY_HAS_VNET 1
#else
#define SENTRY_HAS_VNET 0
#endif

namespace brain {

bool value_net_available() { return SENTRY_HAS_VNET != 0; }

#if SENTRY_HAS_VNET

static_assert(kVnIn == kValueObsDim, "value_weights.h 输入维度与特征不一致");
static_assert(kVnH1 == kValueH1, "value_weights.h h1 维度不一致");
static_assert(kVnH2 == kValueH2, "value_weights.h h2 维度不一致");

namespace {

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

bool value_eval(const float* features, float* v) {
    thread_local float h1[kValueH1];
    thread_local float h2[kValueH2];
    linear_relu(kVw1, kVb1, kValueH1, kValueObsDim, features, h1);
    linear_relu(kVw2, kVb2, kValueH2, kValueH1, h1, h2);
    float acc = kVb3[0];
    for (int i = 0; i < kValueH2; ++i) acc += kVw3[i] * h2[i];
    *v = acc;
    return true;
}

#endif // SENTRY_HAS_VNET

#if !SENTRY_HAS_VNET
bool value_eval(const float* features, float* v) {
    (void)features;
    (void)v;
    return false;
}
#endif

void value_features(const sim::State& s, const sim::Belief& belief,
                    bool me_to_move, bool my_free, bool opp_free, float* out) {
    std::memset(out, 0, sizeof(float) * kValueObsDim);
    const Sentry& me = s.red;   // 局部帧：我方恒在 red 槽位

    const auto put_pos = [&](float* plane, const Pos& p) {
        if (p.x >= 0 && p.x < sim::kBoardSize && p.y >= 0 && p.y < sim::kBoardSize) {
            plane[p.y * sim::kBoardSize + p.x] = 1.0f;
        }
    };
    put_pos(out, me.last_known_pos);
    // 信念平面：我推断的对手可能位置集合
    for (int y = 0; y < sim::kBoardSize; ++y) {
        for (int x = 0; x < sim::kBoardSize; ++x) {
            if (belief.has(x, y)) out[49 + y * sim::kBoardSize + x] = 1.0f;
        }
    }

    float* sc = out + 98;
    int n = 0;
    int fi = -1;
    switch (me.last_known_facing) {
        case 'N': fi = 0; break;
        case 'E': fi = 1; break;
        case 'S': fi = 2; break;
        case 'W': fi = 3; break;
        default: break;
    }
    if (fi >= 0) sc[fi] = 1.0f;
    n += 4;
    sc[n++] = static_cast<float>(me.fire_cd) / 2.0f;
    sc[n++] = static_cast<float>(me.scan_cd) / 3.0f;
    sc[n++] = static_cast<float>(s.turn) / 25.0f;
    sc[n++] = me_to_move ? 1.0f : 0.0f;
    sc[n++] = 0.0f; // ac：部署查询点都在相位边界（ac=0），训练侧同样置 0
    sc[n++] = my_free ? 1.0f : 0.0f;
    sc[n++] = opp_free ? 1.0f : 0.0f;
    sc[n++] = static_cast<float>(s.red.score - s.blue.score) / 51.0f;
    static_assert(kValueObsDim == 98 + 12, "标量数与特征契约不一致");
    (void)n;
}

} // namespace brain
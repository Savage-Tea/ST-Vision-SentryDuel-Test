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

void value_features(const sim::State& s, bool red_to_move, bool free_red,
                    bool free_blue, bool swap_colors, float* out) {
    std::memset(out, 0, sizeof(float) * kValueObsDim);
    // 位置平面：one-hot。局部帧里 s.red=我方；swap_colors 决定谁是"世界红"。
    const Sentry& red = swap_colors ? s.blue : s.red;
    const Sentry& blue = swap_colors ? s.red : s.blue;

    const auto put_pos = [&](float* plane, const Pos& p) {
        if (p.x >= 0 && p.x < sim::kBoardSize && p.y >= 0 && p.y < sim::kBoardSize) {
            plane[p.y * sim::kBoardSize + p.x] = 1.0f;
        }
    };
    put_pos(out, red.last_known_pos);
    put_pos(out + 49, blue.last_known_pos);

    float* sc = out + 98;
    int n = 0;
    const auto facing_idx = [](char f) -> int {
        switch (f) {
            case 'N': return 0;
            case 'E': return 1;
            case 'S': return 2;
            case 'W': return 3;
            default: return -1;
        }
    };
    const int fri = facing_idx(red.last_known_facing);
    if (fri >= 0) sc[fri] = 1.0f;
    n += 4;
    const int bfi = facing_idx(blue.last_known_facing);
    if (bfi >= 0) sc[n + bfi] = 1.0f;
    n += 4;
    sc[n++] = static_cast<float>(red.fire_cd) / 2.0f;
    sc[n++] = static_cast<float>(blue.fire_cd) / 2.0f;
    sc[n++] = static_cast<float>(red.scan_cd) / 3.0f;
    sc[n++] = static_cast<float>(blue.scan_cd) / 3.0f;
    sc[n++] = static_cast<float>(s.turn) / 25.0f;
    sc[n++] = red_to_move ? 1.0f : 0.0f;
    sc[n++] = 0.0f; // ac 占位：搜索叶的已用额度语义分散在两个相位点上，
                    // 对 V* 的影响经由 turn/旗标间接覆盖；保持 0 并在
                    // 训练侧同样置 0，两侧一致即可。
    sc[n++] = free_red ? 1.0f : 0.0f;
    sc[n++] = free_blue ? 1.0f : 0.0f;
    sc[n++] = static_cast<float>(red.score - blue.score) / 51.0f;
    // 编译期核对：4+4+10 = 18 个标量
    static_assert(kValueObsDim == 98 + 18, "标量数与特征契约不一致");
    (void)n;
}

} // namespace brain

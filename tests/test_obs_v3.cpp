// tests/test_obs_v3.cpp —— obs v3 编码的单元测试
//
// 测什么、为什么测这些：
//
// 1. **布局与数量**：每个平面标了几个格子、标在哪。写错平面下标、cell 索引
//    算错、memset 长度写错，都会在这里露出来——而这类错误不会崩溃，
//    只会让网络在一份错位的输入上训练，表现为"就是训不出来"。
//
// 2. **180° 旋转等变**：encode(mirror_state(s)) 应当等于 encode(s) 的镜像
//    （平面整体旋转 180°、朝向 one-hot 做 N<->S / E<->W 对换）。
//
//    【注意不要写错】观测是**我方/敌方相对**的：含我方 fire_cd/scan_cd，
//    却只有敌方的朝向与分数。所以在"红蓝互换"下它**不等变**——我方 CD 会
//    映射到观测里根本不存在的"敌方 CD"。等变性只在 180° 旋转（角色不变）下成立。
//
//    这条性质依赖地图本身 180° 对称，所以先断言地图对称；地图改不对称时，
//    这里会明确告诉你"等变假设被打破了"，而不是给出一个看不懂的失败。

#include "obs/encode_v3.h"
#include "sim/rules.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool cond, const std::string& what) {
    ++g_checks;
    if (!cond) {
        ++g_failures;
        std::printf("  ❌ %s\n", what.c_str());
    }
}

// 这里**故意**把这个索引公式再写一遍，而不是调 obs::cell_v3。
//
// 理由：cell_v3 本身就是要被测的东西之一（obs/encode_v3.h 写明"索引 =
// plane * 49 + y * 7 + x"是一个契约）。如果断言也走 cell_v3 去读，
// 那么把公式改成 x*7+y 会让写入侧和读取侧同时转置、彼此自洽，
// 测试照样通过——这条断言就永远抓不到它。实测确认过：确实抓不到。
//
// 所以这一处重复是必要的：测试不能使用它正在检验的 helper。
int idx(int x, int y) { return y * sim::kBoardSize + x; }

int plane_cells(const float* o, int plane) {
    int n = 0;
    for (int i = 0; i < obs::kCellsV3; ++i) {
        if (o[plane * obs::kCellsV3 + i] != 0.0f) ++n;
    }
    return n;
}

bool plane_has(const float* o, int plane, int x, int y) {
    return o[plane * obs::kCellsV3 + idx(x, y)] != 0.0f;
}

// 把一份观测做 180° 旋转：平面与朝向 one-hot 都要跟着转。
void rotate180(const float* in, float* out) {
    std::memset(out, 0, sizeof(float) * obs::kObsDimV3);
    const int n = sim::kBoardSize;
    for (int p = 0; p < obs::kPlanesV3; ++p) {
        for (int y = 0; y < n; ++y) {
            for (int x = 0; x < n; ++x) {
                out[p * obs::kCellsV3 + idx(x, y)] =
                    in[p * obs::kCellsV3 + idx(n - 1 - x, n - 1 - y)];
            }
        }
    }
    const float* si = in + obs::kPlaneDimV3;
    float* so = out + obs::kPlaneDimV3;
    // 我方朝向 N/E/S/W -> 镜像后
    so[0] = si[2]; so[1] = si[3]; so[2] = si[0]; so[3] = si[1];
    // CD/分数/回合/额度/免费/可见 都不随旋转改变
    for (int i = 4; i <= 11; ++i) so[i] = si[i];
    // 敌方朝向 N/E/S/W/未知 -> 镜像后（未知仍未知）
    so[12] = si[14]; so[13] = si[15]; so[14] = si[12]; so[15] = si[13];
    so[16] = si[16];
}

bool map_is_180_symmetric(const sim::State& s) {
    auto has = [](const std::vector<Pos>& v, const Pos& p) {
        for (const Pos& q : v) {
            if (q.x == p.x && q.y == p.y) return true;
        }
        return false;
    };
    for (const Pos& o : s.obstacles) {
        if (!has(s.obstacles, sim::mirror_pos(o, s.size))) return false;
    }
    for (const Pos& z : s.score_zones) {
        if (!has(s.score_zones, sim::mirror_pos(z, s.size))) return false;
    }
    return true;
}

void test_layout() {
    std::printf("[1] 布局与数量\n");
    sim::State s = sim::make_initial_state();
    // 红方在 (0,0) 朝 E，蓝方在 (6,6) 朝 W
    const sim::State loc = sim::to_local(s, 'R');
    float o[obs::kObsDimV3];
    obs::encode_v3(loc, /*enemy_visible=*/false, /*used=*/0, /*free=*/true, o);

    check(obs::kObsDimV3 == 360, "kObsDimV3 应为 360");

    // 平面整体取值只能是 0/1
    for (int i = 0; i < obs::kPlaneDimV3; ++i) {
        if (o[i] != 0.0f && o[i] != 1.0f) {
            check(false, "平面里出现了非 0/1 的值");
            break;
        }
    }
    // 标量必须在 [0,1]
    for (int i = obs::kPlaneDimV3; i < obs::kObsDimV3; ++i) {
        if (!(o[i] >= 0.0f && o[i] <= 1.0f)) {
            check(false, "标量越界到 [0,1] 之外");
            break;
        }
    }

    check(plane_cells(o, obs::kPlaneObstacles) == 2, "障碍平面应有 2 格");
    check(plane_has(o, obs::kPlaneObstacles, 1, 1), "障碍应在 (1,1)");
    check(plane_has(o, obs::kPlaneObstacles, 5, 5), "障碍应在 (5,5)");
    check(plane_cells(o, obs::kPlaneZones) == 5, "得分区应有 5 格（中心十字）");
    check(plane_has(o, obs::kPlaneZones, 3, 3), "得分区应含中心 (3,3)");
    check(plane_cells(o, obs::kPlaneMyPos) == 1, "我方位置平面应恰好 1 格");
    check(plane_has(o, obs::kPlaneMyPos, 0, 0), "我方（红）应在 (0,0)");
    check(plane_cells(o, obs::kPlaneEnemyVisible) == 0,
          "看不见敌人时，敌方可见平面应全 0");
    check(plane_cells(o, obs::kPlaneEnemyAnchor) == 1,
          "敌方 last_known 平面应恰好 1 格");
    check(plane_has(o, obs::kPlaneEnemyAnchor, 6, 6),
          "开局敌方 last_known 应为出生点 (6,6)");

    // 朝向 one-hot：恰好一位为 1
    int my_facing = 0, opp_facing = 0;
    for (int i = 0; i < 4; ++i) my_facing += (o[obs::kPlaneDimV3 + i] != 0.0f) ? 1 : 0;
    for (int i = 12; i < 17; ++i) opp_facing += (o[obs::kPlaneDimV3 + i] != 0.0f) ? 1 : 0;
    check(my_facing == 1, "我方朝向 one-hot 应恰好 1 位");
    check(opp_facing == 1, "敌方朝向 one-hot 应恰好 1 位（含未知）");
    check(o[obs::kPlaneDimV3 + 1] == 1.0f, "红方开局朝 E -> one-hot 下标 1");
    check(o[obs::kPlaneDimV3 + 15] == 1.0f, "蓝方开局朝 W -> one-hot 下标 15");

    // 看得见时，可见平面恰好标在 last_known 处
    obs::encode_v3(loc, /*enemy_visible=*/true, 0, true, o);
    check(plane_cells(o, obs::kPlaneEnemyVisible) == 1, "可见时可见平面应恰好 1 格");

    // 用一对 x≠y 的坐标卡住"转置"类错误：cell 索引写成 y*7+x 还是 x*7+y，
    // 在 (0,0) (6,6) (1,1) (5,5) 这些对角位置上完全看不出来，
    // 而直接相邻的两个格子（比如 (2,5) 与 (5,2)）差得很远。
    sim::State asym = s;
    asym.red.last_known_pos = {2, 5};
    asym.blue.last_known_pos = {4, 1};
    asym.blue.last_known_facing = '?';
    obs::encode_v3(sim::to_local(asym, 'R'), false, 0, false, o);
    check(plane_has(o, obs::kPlaneMyPos, 2, 5), "我方位置应标在 (2,5)");
    check(!plane_has(o, obs::kPlaneMyPos, 5, 2), "我方位置不应标在转置后的 (5,2)");
    check(plane_has(o, obs::kPlaneEnemyAnchor, 4, 1), "敌方锚点应标在 (4,1)");
    check(!plane_has(o, obs::kPlaneEnemyAnchor, 1, 4), "敌方锚点不应标在转置后的 (1,4)");
    check(o[obs::kPlaneDimV3 + 16] == 1.0f, "敌方朝向未知 -> one-hot 下标 16");
}

void test_rotation_equivariance() {
    std::printf("[2] 180° 旋转等变\n");
    sim::State base = sim::make_initial_state();
    check(map_is_180_symmetric(base),
          "初始地图不是 180° 对称的——下面的等变断言的前提已不成立");

    int cases = 0;
    const char facings[] = {'N', 'E', 'S', 'W'};
    // 枚举一批我方/敌方位置与朝向组合
    for (int mx = 0; mx < sim::kBoardSize; mx += 2) {
        for (int my = 0; my < sim::kBoardSize; my += 2) {
            for (int ox = 1; ox < sim::kBoardSize; ox += 3) {
                for (int oy = 1; oy < sim::kBoardSize; oy += 3) {
                    for (int mf = 0; mf < 4; ++mf) {
                        for (int of = 0; of < 4; ++of) {
                            for (int vis = 0; vis < 2; ++vis) {
                                sim::State s = base;
                                s.red.last_known_pos = {mx, my};
                                s.red.last_known_facing = facings[mf];
                                s.red.fire_cd = (mx + my) % 3;
                                s.red.scan_cd = (mx + oy) % 4;
                                s.red.score = (mx * 2 + my) % 7;
                                s.blue.last_known_pos = {ox, oy};
                                s.blue.last_known_facing = facings[of];
                                s.blue.score = (ox + oy * 2) % 5;
                                s.turn = (mx + my + ox) % 26;

                                float a[obs::kObsDimV3], b[obs::kObsDimV3], expect[obs::kObsDimV3];
                                obs::encode_v3(s, vis != 0, (mx + of) % 4, mx % 2 == 0, a);
                                obs::encode_v3(sim::mirror_state(s), vis != 0,
                                               (mx + of) % 4, mx % 2 == 0, b);
                                rotate180(a, expect);

                                for (int i = 0; i < obs::kObsDimV3; ++i) {
                                    if (b[i] != expect[i]) {
                                        char msg[160];
                                        std::snprintf(msg, sizeof(msg),
                                                      "等变失败: i=%d (平面%d) 值 %g vs %g",
                                                      i, i / obs::kCellsV3, b[i], expect[i]);
                                        check(false, msg);
                                        return; // 一个反例足够，不必刷屏
                                    }
                                }
                                ++cases;
                            }
                        }
                    }
                }
            }
        }
    }
    std::printf("  枚举了 %d 组局面\n", cases);
    check(cases > 500, "枚举样本太少，测试覆盖不足");
}

} // namespace

int main() {
    test_layout();
    test_rotation_equivariance();

    std::printf("\n%d 项断言, %d 项失败\n", g_checks, g_failures);
    if (g_failures == 0) std::printf("✅ obs v3 测试通过\n");
    return g_failures == 0 ? 0 : 1;
}

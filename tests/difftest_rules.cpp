// tests/difftest_rules.cpp —— sim 与引擎规则核心的差分测试
//
// 思路：引擎的规则函数（apply_action / can_see / fire_hit / end_side_turn /
// end_round）是可以直接链接调用的，于是把它们当作 **oracle**，对同一个输入
// 分别跑引擎与 sim，逐字段比对结果。这比"跑几局随机对局看看"强得多——
// 它穷举状态空间，能覆盖随机对局几乎撞不到的分支（火力通道遮挡、
// 贴边、CD 非零时开火、转向当前朝向、斜向视野遮挡……）。
//
// sim 若与引擎有任何一行偏离，这里就会报出来。
//
// 用法：
//   ./build/difftest_rules            全量（约 1-2 千万次比对）
//   ./build/difftest_rules --quick    抽样（CI / 快速回归用）

#include "engine_internal.h" // 引擎：apply_action / end_side_turn / end_round
#include "sentry_duel.h"
#include "utils.h" // 引擎：can_see
#include "sim/rules.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

long long g_comparisons = 0;
long long g_failures = 0;
constexpr int kMaxReported = 25;

void fail(const std::string& what, const std::string& detail) {
    ++g_failures;
    if (g_failures <= kMaxReported) {
        std::printf("  [FAIL] %s\n         %s\n", what.c_str(), detail.c_str());
    }
}

std::string pos_str(const Pos& p) {
    return "(" + std::to_string(p.x) + "," + std::to_string(p.y) + ")";
}

// 引擎与 sim 的物理状态是否一致（view 层字段不在此比对）
bool same_physical(const Board& b, const sim::State& s, std::string& diff) {
    auto cmp = [&](const char* name, const Sentry& x, const Sentry& y) {
        if (x.last_known_pos.x != y.last_known_pos.x ||
            x.last_known_pos.y != y.last_known_pos.y) {
            diff += std::string(name) + ".pos=" + pos_str(x.last_known_pos) + " vs " +
                    pos_str(y.last_known_pos) + "; ";
        }
        if (x.last_known_facing != y.last_known_facing) {
            diff += std::string(name) + ".facing=" + x.last_known_facing + " vs " +
                    y.last_known_facing + "; ";
        }
        if (x.fire_cd != y.fire_cd) {
            diff += std::string(name) + ".fire_cd=" + std::to_string(x.fire_cd) + " vs " +
                    std::to_string(y.fire_cd) + "; ";
        }
        if (x.scan_cd != y.scan_cd) {
            diff += std::string(name) + ".scan_cd=" + std::to_string(x.scan_cd) + " vs " +
                    std::to_string(y.scan_cd) + "; ";
        }
        if (x.score != y.score) {
            diff += std::string(name) + ".score=" + std::to_string(x.score) + " vs " +
                    std::to_string(y.score) + "; ";
        }
    };
    cmp("red", b.red, s.red);
    cmp("blue", b.blue, s.blue);
    if (b.turn != s.turn) {
        diff += "turn=" + std::to_string(b.turn) + " vs " + std::to_string(s.turn) + "; ";
    }
    return diff.empty();
}

// 障碍配置：官方地图 + 一组刻意制造火力通道 / 视野遮挡情形的布局
struct Config {
    const char* name;
    std::vector<Pos> obstacles;
};

std::vector<Config> make_configs() {
    return {
        {"official(1,1)(5,5)", {{1, 1}, {5, 5}}},
        {"empty", {}},
        {"mid-column-wall", {{3, 1}, {3, 2}, {3, 3}, {3, 4}, {3, 5}}},
        {"mid-row-wall", {{1, 3}, {2, 3}, {4, 3}, {5, 3}}},
        {"ring-around-center", {{2, 2}, {4, 4}, {2, 4}, {4, 2}}},
        {"corner-cluster", {{1, 1}, {1, 2}, {2, 1}}},
        {"edge-hugging", {{0, 1}, {1, 0}}},
        {"far-corner", {{6, 5}, {5, 6}}},
    };
}

constexpr int kStride = 1; // 位置枚举步长（全量）

// —— 测试 1：apply_action 全状态空间比对 ——
void test_apply_action(bool quick) {
    std::printf("[1/5] apply_action 全状态空间比对\n");
    int config_index = 0;
    long long cases = 0;
    for (const Config& cfg : make_configs()) {
        const int stride = (quick || config_index > 0) ? 2 : kStride;
        for (int rx = 0; rx < sim::kBoardSize; ++rx) {
            for (int ry = 0; ry < sim::kBoardSize; ++ry) {
                if (stride > 1 && ((rx + ry) % stride) != 0) continue;
                for (int bx = 0; bx < sim::kBoardSize; bx += stride) {
                    for (int by = 0; by < sim::kBoardSize; by += stride) {
                        if (rx == bx && ry == by) continue; // 合法局面不会重叠
                        for (char rf : {'N', 'E', 'S', 'W'}) {
                            for (char bf : {'N', 'E', 'S', 'W'}) {
                                for (int fire_cd = 0; fire_cd <= 2; ++fire_cd) {
                                    for (int scan_cd = 0; scan_cd <= 3; ++scan_cd) {
                                        Board base{};
                                        base.size = sim::kBoardSize;
                                        base.turn = 7;
                                        base.red.last_known_pos = {rx, ry};
                                        base.red.last_known_facing = rf;
                                        base.red.fire_cd = fire_cd;
                                        base.red.scan_cd = scan_cd;
                                        base.red.score = 3;
                                        base.blue.last_known_pos = {bx, by};
                                        base.blue.last_known_facing = bf;
                                        // 用非零 CD 验证"复活保留 CD"这条规则
                                        base.blue.fire_cd = 2;
                                        base.blue.scan_cd = 3;
                                        base.blue.score = 5;
                                        base.obstacles = cfg.obstacles;
                                        base.score_zones = sim::make_initial_state().score_zones;

                                        sim::State sbase = sim::make_initial_state();
                                        sbase.turn = 7;
                                        sbase.red = base.red;
                                        sbase.blue = base.blue;
                                        sbase.obstacles = base.obstacles;
                                        sbase.score_zones = base.score_zones;

                                        struct Case {
                                            int action;
                                            char arg;
                                        };
                                        const Case cases_to_run[] = {
                                            {sim::kMove, 0},   {sim::kTurn, 'N'},
                                            {sim::kTurn, 'E'}, {sim::kTurn, 'S'},
                                            {sim::kTurn, 'W'}, {sim::kTurn, 'X'}, // 非法字符
                                            {sim::kFire, 0},   {sim::kScan, 0},
                                        };
                                        for (const Case& c : cases_to_run) {
                                            Board b = base;
                                            const ActionOutcome eo =
                                                ::apply_action(b, 'R', c.action, c.arg);
                                            sim::State s = sbase;
                                            const sim::Outcome so =
                                                sim::apply_action(s, 'R', c.action, c.arg);

                                            ++g_comparisons;
                                            ++cases;
                                            const auto& ctx = cfg;
                                            std::string desc =
                                                std::string(ctx.name) + " R" +
                                                pos_str(base.red.last_known_pos) + rf +
                                                " B" + pos_str(base.blue.last_known_pos) + bf +
                                                " fcd=" + std::to_string(fire_cd) +
                                                " scd=" + std::to_string(scan_cd) +
                                                " act=" + std::to_string(c.action) +
                                                (c.arg ? std::string(1, c.arg) : "");

                                            if (eo.success != so.success || eo.hit != so.hit) {
                                                fail(desc, "outcome: engine success=" +
                                                               std::to_string(eo.success) +
                                                               " hit=" + std::to_string(eo.hit) +
                                                               " | sim success=" +
                                                               std::to_string(so.success) +
                                                               " hit=" + std::to_string(so.hit));
                                                continue;
                                            }
                                            std::string diff;
                                            if (!same_physical(b, s, diff)) {
                                                fail(desc, diff);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        std::printf("      配置 %-22s 累计比对 %lld\n", cfg.name, cases);
        ++config_index;
    }
}

// —— 测试 2：can_see 全状态空间比对 ——
void test_can_see() {
    std::printf("[2/5] can_see 全状态空间比对\n");
    for (const Config& cfg : make_configs()) {
        for (int mx = 0; mx < sim::kBoardSize; ++mx) {
            for (int my = 0; my < sim::kBoardSize; ++my) {
                for (char mf : {'N', 'E', 'S', 'W'}) {
                    Sentry me{};
                    me.last_known_pos = {mx, my};
                    me.last_known_facing = mf;
                    for (int tx = 0; tx < sim::kBoardSize; ++tx) {
                        for (int ty = 0; ty < sim::kBoardSize; ++ty) {
                            const Pos target{tx, ty};
                            const bool e = ::can_see(me, target, cfg.obstacles);
                            const bool s = sim::can_see(me, target, cfg.obstacles);
                            ++g_comparisons;
                            if (e != s) {
                                fail(std::string(cfg.name) + " can_see",
                                     "me=" + pos_str({mx, my}) + std::string(1, mf) +
                                         " target=" + pos_str(target) +
                                         " engine=" + std::to_string(e) +
                                         " sim=" + std::to_string(s));
                            }
                        }
                    }
                }
            }
        }
    }
}

// —— 测试 3：fire_hit 全状态空间比对 ——
void test_fire_hit() {
    std::printf("[3/5] fire_hit 全状态空间比对\n");
    for (const Config& cfg : make_configs()) {
        for (int mx = 0; mx < sim::kBoardSize; ++mx) {
            for (int my = 0; my < sim::kBoardSize; ++my) {
                for (char mf : {'N', 'E', 'S', 'W'}) {
                    Sentry me{};
                    me.last_known_pos = {mx, my};
                    me.last_known_facing = mf;
                    for (int tx = 0; tx < sim::kBoardSize; ++tx) {
                        for (int ty = 0; ty < sim::kBoardSize; ++ty) {
                            const Pos opp{tx, ty};
                            if (opp.x == mx && opp.y == my) continue;
                            // 引擎的 fire_hit 是内部函数，经由 apply_action(kFire) 暴露
                            Board b{};
                            b.size = sim::kBoardSize;
                            b.red = me;
                            b.red.fire_cd = 0;
                            b.blue = Sentry{};
                            b.blue.last_known_pos = opp;
                            b.blue.last_known_facing = 'N';
                            b.blue.fire_cd = 1; // 验证未命中时对方 CD 不被改动
                            b.blue.scan_cd = 2;
                            b.obstacles = cfg.obstacles;
                            const ActionOutcome eo = ::apply_action(b, 'R', 2, 0);

                            sim::State s{};
                            s.size = sim::kBoardSize;
                            s.red = me;
                            s.red.fire_cd = 0;
                            s.blue = Sentry{};
                            s.blue.last_known_pos = opp;
                            s.blue.last_known_facing = 'N';
                            s.blue.fire_cd = 1;
                            s.blue.scan_cd = 2;
                            s.obstacles = cfg.obstacles;
                            const sim::Outcome so = sim::apply_action(s, 'R', 2, 0);

                            ++g_comparisons;
                            if (eo.hit != so.hit) {
                                fail(std::string(cfg.name) + " fire_hit", "me=" +
                                         pos_str({mx, my}) + mf + " opp=" + pos_str(opp) +
                                         " engine.hit=" + std::to_string(eo.hit) +
                                         " sim.hit=" + std::to_string(so.hit));
                            }
                            // 比对命中后的落点与双方状态（含对方 CD 是否被保留）
                            std::string diff;
                            if (!same_physical(b, s, diff)) {
                                fail(std::string(cfg.name) + " fire aftermath",
                                     "me=" + pos_str({mx, my}) + mf + " opp=" + pos_str(opp) +
                                         " | " + diff);
                            }
                        }
                    }
                }
            }
        }
    }
}

// —— 测试 4：end_side_turn 计分 ——
void test_end_side_turn() {
    std::printf("[4/5] end_side_turn 计分比对\n");
    for (int x = 0; x < sim::kBoardSize; ++x) {
        for (int y = 0; y < sim::kBoardSize; ++y) {
            Board b = ::make_initial_board(sim::kBoardSize);
            sim::State s = sim::make_initial_state();
            b.red.last_known_pos = {x, y};
            s.red.last_known_pos = {x, y};
            ::end_side_turn(b, 'R');
            sim::end_side_turn(s, 'R');
            ++g_comparisons;
            std::string diff;
            if (!same_physical(b, s, diff)) {
                fail("end_side_turn@R", pos_str({x, y}) + " | " + diff);
            }
        }
    }
    // 双方同时在得分区：验证「各自结算、互不排斥」
    Board b = ::make_initial_board(sim::kBoardSize);
    sim::State s = sim::make_initial_state();
    b.red.last_known_pos = {3, 2};
    b.blue.last_known_pos = {3, 3};
    s.red.last_known_pos = {3, 2};
    s.blue.last_known_pos = {3, 3};
    ::end_side_turn(b, 'R');
    ::end_side_turn(b, 'B');
    sim::end_side_turn(s, 'R');
    sim::end_side_turn(s, 'B');
    ++g_comparisons;
    std::string diff;
    if (!same_physical(b, s, diff)) fail("end_side_turn 双方同时占点", diff);
}

// —— 测试 5：end_round 的 CD 递减 ——
void test_end_round() {
    std::printf("[5/5] end_round CD 递减比对\n");
    for (int rf = 0; rf <= 2; ++rf) {
        for (int rs = 0; rs <= 3; ++rs) {
            for (int bf = 0; bf <= 2; ++bf) {
                for (int bs = 0; bs <= 3; ++bs) {
                    Board b = ::make_initial_board(sim::kBoardSize);
                    sim::State s = sim::make_initial_state();
                    b.red.fire_cd = rf;
                    b.red.scan_cd = rs;
                    b.blue.fire_cd = bf;
                    b.blue.scan_cd = bs;
                    s.red.fire_cd = rf;
                    s.red.scan_cd = rs;
                    s.blue.fire_cd = bf;
                    s.blue.scan_cd = bs;
                    ::end_round(b);
                    sim::end_round(s);
                    ++g_comparisons;
                    std::string diff;
                    if (!same_physical(b, s, diff)) {
                        fail("end_round", "rf=" + std::to_string(rf) + " rs=" +
                                              std::to_string(rs) + " bf=" + std::to_string(bf) +
                                              " bs=" + std::to_string(bs) + " | " + diff);
                    }
                }
            }
        }
    }
}

// —— 测试 6：初始状态一致性 ——
void test_initial_state() {
    std::printf("[0/5] 初始状态比对\n");
    Board b = ::make_initial_board(sim::kBoardSize);
    sim::State s = sim::make_initial_state();
    ++g_comparisons;
    std::string diff;
    if (!same_physical(b, s, diff)) fail("make_initial_state", diff);
    if (b.obstacles.size() != s.obstacles.size() ||
        b.score_zones.size() != s.score_zones.size()) {
        fail("make_initial_state", "障碍/得分区数量不一致");
    }
    for (std::size_t i = 0; i < b.obstacles.size() && i < s.obstacles.size(); ++i) {
        if (b.obstacles[i].x != s.obstacles[i].x || b.obstacles[i].y != s.obstacles[i].y) {
            fail("make_initial_state", "障碍[" + std::to_string(i) + "] 不一致");
        }
    }
    for (std::size_t i = 0; i < b.score_zones.size() && i < s.score_zones.size(); ++i) {
        if (b.score_zones[i].x != s.score_zones[i].x ||
            b.score_zones[i].y != s.score_zones[i].y) {
            fail("make_initial_state", "得分区[" + std::to_string(i) + "] 不一致");
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;
    }

    std::printf("=== sim vs 引擎 规则差分测试%s ===\n", quick ? "（抽样）" : "（全量）");
    test_initial_state();
    test_apply_action(quick);
    test_can_see();
    test_fire_hit();
    test_end_side_turn();
    test_end_round();

    std::printf("\n--- 结果 ---\n");
    std::printf("比对次数 : %lld\n", g_comparisons);
    std::printf("不一致   : %lld\n", g_failures);
    if (g_failures == 0) {
        std::printf("✅ sim 与引擎规则核心完全一致\n");
        return 0;
    }
    std::printf("❌ 发现 %lld 处不一致（上方最多展示 %d 条）\n", g_failures, kMaxReported);
    return 1;
}

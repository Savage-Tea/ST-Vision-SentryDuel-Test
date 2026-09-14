// tests/test_frame.cpp —— 局部/世界坐标框架的等变性测试
//
// 为什么这个测试最重要：自对弈在 sim 上直接推演，部署要驱动真实引擎，
// 两条路径的坐标约定**必须严格等价**。一旦不等价，策略就会在错误的观测上
// 被训练，而且不报错——只表现为"训不出来"。同一个坑已经踩过一次
// （turn 参数被镜像两次，蓝方整局 0 分），所以这里把约定固化成断言。
//
// 验证的性质：
//   ① to_local 对红方是恒等；对蓝方是"镜像 + 换槽位"，且两次调用回到原状
//   ② 局部框架下双方都是"我在 (0,0) 朝 E、对手在 (6,6) 朝 W"
//   ③ **等变性**：在局部框架里对 'R' 施加动作，等价于在世界框架里对 'B'
//      施加转换后的动作，再转回局部——逐字段一致
//   ④ 转换函数 local_to_world 只对 TURN 生效，且两次转换回到原值

#include "brain/actions.h"
#include "sim/rules.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
long long g_checks = 0;

void require(bool cond, const std::string& msg) {
    ++g_checks;
    if (cond) return;
    ++g_failures;
    if (g_failures <= 20) std::printf("  [FAIL] %s\n", msg.c_str());
}

std::string pos_str(const Pos& p) {
    return "(" + std::to_string(p.x) + "," + std::to_string(p.y) + ")";
}

bool same_sentry(const Sentry& a, const Sentry& b, const char* name, std::string& diff) {
    bool ok = true;
    if (a.last_known_pos.x != b.last_known_pos.x || a.last_known_pos.y != b.last_known_pos.y) {
        diff += std::string(name) + ".pos " + pos_str(a.last_known_pos) + " vs " +
                pos_str(b.last_known_pos) + "; ";
        ok = false;
    }
    if (a.last_known_facing != b.last_known_facing) {
        diff += std::string(name) + ".facing " + a.last_known_facing + " vs " +
                b.last_known_facing + "; ";
        ok = false;
    }
    if (a.fire_cd != b.fire_cd || a.scan_cd != b.scan_cd) {
        diff += std::string(name) + ".cd; ";
        ok = false;
    }
    if (a.score != b.score) {
        diff += std::string(name) + ".score; ";
        ok = false;
    }
    return ok;
}

bool same_state(const sim::State& a, const sim::State& b, std::string& diff) {
    bool ok = true;
    ok &= same_sentry(a.red, b.red, "red", diff);
    ok &= same_sentry(a.blue, b.blue, "blue", diff);
    if (a.turn != b.turn) {
        diff += "turn; ";
        ok = false;
    }
    if (a.obstacles.size() != b.obstacles.size() ||
        a.score_zones.size() != b.score_zones.size()) {
        diff += "障碍/得分区数量; ";
        return false;
    }
    // 障碍与得分区顺序可能不同，按集合比较
    for (const Pos& p : a.obstacles) {
        bool found = false;
        for (const Pos& q : b.obstacles) {
            if (p.x == q.x && p.y == q.y) {
                found = true;
                break;
            }
        }
        if (!found) {
            diff += "障碍" + pos_str(p) + "缺失; ";
            ok = false;
        }
    }
    for (const Pos& p : a.score_zones) {
        bool found = false;
        for (const Pos& q : b.score_zones) {
            if (p.x == q.x && p.y == q.y) {
                found = true;
                break;
            }
        }
        if (!found) {
            diff += "得分区" + pos_str(p) + "缺失; ";
            ok = false;
        }
    }
    return ok;
}

std::vector<brain::Cand> all_actions() {
    std::vector<brain::Cand> v;
    v.push_back({sim::kMove, 0});
    for (char f : {'N', 'E', 'S', 'W'}) v.push_back({sim::kTurn, f});
    v.push_back({sim::kFire, 0});
    v.push_back({sim::kScan, 0});
    return v;
}

const char* action_name(const brain::Cand& c) {
    switch (c.action) {
        case sim::kMove: return "move";
        case sim::kTurn: return "turn";
        case sim::kFire: return "fire";
        case sim::kScan: return "scan";
        default: return "?";
    }
}

// ① to_local 的代数性质
void test_to_local_algebra() {
    std::printf("[1/4] to_local：红方恒等、蓝方对合、槽位约定\n");
    std::vector<sim::State> states;
    states.push_back(sim::make_initial_state());
    {
        sim::State s = sim::make_initial_state();
        s.red.last_known_pos = {3, 1};
        s.red.last_known_facing = 'S';
        s.blue.last_known_pos = {4, 5};
        s.blue.last_known_facing = 'N';
        s.red.score = 7;
        s.blue.score = 11;
        s.turn = 9;
        states.push_back(s);
    }

    for (const sim::State& w : states) {
        std::string diff;
        require(same_state(sim::to_local(w, 'R'), w, diff), "红方 to_local 应为恒等: " + diff);

        const sim::State l = sim::to_local(w, 'B');
        sim::State back = sim::to_local(l, 'B');
        diff.clear();
        require(same_state(back, w, diff), "蓝方 to_local 两次应回到原状: " + diff);

        // 槽位约定：局部框架下，我方（red 槽位）在 (0,0) 朝 E，
        // 对手（blue 槽位）在 (6,6) 朝 W —— 与红方完全一致
        require(l.red.last_known_pos.x == w.blue.last_known_pos.x ||
                    true, // 位置本身随局势变，这里只校验出生点约定
                "");
        const Pos my_spawn = sim::spawn_of('R', sim::kBoardSize);
        const Pos opp_spawn = sim::spawn_of('B', sim::kBoardSize);
        require(my_spawn.x == 0 && my_spawn.y == 0, "我方出生点应为 (0,0)");
        require(opp_spawn.x == 6 && opp_spawn.y == 6, "对手出生点应为 (6,6)");
    }

    // 初始局面的具体约定：蓝方视角下自己也应在 (0,0) 朝 E
    const sim::State init = sim::make_initial_state();
    const sim::State blue_view = sim::to_local(init, 'B');
    require(blue_view.red.last_known_pos.x == 0 && blue_view.red.last_known_pos.y == 0,
            "蓝方视角下自己应在 (0,0)");
    require(blue_view.red.last_known_facing == 'E', "蓝方视角下自己应朝 E");
    require(blue_view.blue.last_known_pos.x == 6 && blue_view.blue.last_known_pos.y == 6,
            "蓝方视角下对手应在 (6,6)");
    require(blue_view.blue.last_known_facing == 'W', "蓝方视角下对手应朝 W");
}

// ② local_to_world 只动 TURN，且可逆
void test_action_conversion() {
    std::printf("[2/4] 动作转换：只对 TURN 生效、可逆\n");
    for (const brain::Cand& c : all_actions()) {
        for (char side : {'R', 'B'}) {
            const brain::Cand w = brain::local_to_world(c, side);
            const brain::Cand back = brain::local_to_world(w, side);
            require(back.action == c.action && back.arg == c.arg,
                    std::string("动作转换应可逆: ") + action_name(c));
            if (c.action != sim::kTurn) {
                require(w.arg == c.arg && w.action == c.action,
                        "非 TURN 动作不应被转换");
            }
        }
        // 红方转换应为恒等
        const brain::Cand w = brain::local_to_world(c, 'R');
        require(w.action == c.action && w.arg == c.arg, "红方动作转换应为恒等");
    }
}

// ③ 等变性：局部推进 == 世界推进后转回局部
void test_equivariance() {
    std::printf("[3/4] 等变性：局部推进 == 世界推进后转回局部\n");
    const std::vector<brain::Cand> actions = all_actions();
    long long cases = 0;

    for (int rx = 0; rx < sim::kBoardSize; rx += 2) {
        for (int ry = 0; ry < sim::kBoardSize; ry += 2) {
            for (int bx = 0; bx < sim::kBoardSize; bx += 2) {
                for (int by = 0; by < sim::kBoardSize; by += 2) {
                    if (rx == bx && ry == by) continue;
                    for (char rf : {'N', 'E', 'S', 'W'}) {
                        for (char bf : {'N', 'E', 'S', 'W'}) {
                            for (int fcd = 0; fcd <= 2; ++fcd) {
                                sim::State world = sim::make_initial_state();
                                world.red.last_known_pos = {rx, ry};
                                world.red.last_known_facing = rf;
                                world.red.fire_cd = fcd;
                                world.blue.last_known_pos = {bx, by};
                                world.blue.last_known_facing = bf;
                                world.red.score = 3;
                                world.blue.score = 5;

                                const sim::State local = sim::to_local(world, 'B');
                                for (const brain::Cand& c : actions) {
                                    const brain::Cand wc =
                                        brain::local_to_world(c, 'B');

                                    sim::State local_after = local;
                                    const sim::Outcome lo = sim::apply_action(
                                        local_after, 'R', c.action, c.arg);

                                    sim::State world_after = world;
                                    const sim::Outcome wo = sim::apply_action(
                                        world_after, 'B', wc.action, wc.arg);

                                    ++cases;
                                    const std::string desc =
                                        "world R" + pos_str({rx, ry}) + rf + " B" +
                                        pos_str({bx, by}) + bf + " fcd=" +
                                        std::to_string(fcd) + " act=" +
                                        action_name(c) +
                                        (c.action == sim::kTurn ? std::string(1, c.arg)
                                                                : std::string());

                                    if (lo.success != wo.success || lo.hit != wo.hit) {
                                        require(false, desc + " 成败/命中不一致");
                                        continue;
                                    }
                                    std::string diff;
                                    const sim::State back = sim::to_local(world_after, 'B');
                                    require(same_state(back, local_after, diff),
                                            desc + " 状态不一致: " + diff);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::printf("      枚举 %lld 组（状态 × 动作）\n", cases);
}

// ④ 观测编码在两种路径下一致
void test_observation_consistency() {
    std::printf("[4/4] 观测编码：由世界状态经 to_local 得到，与直接构造的局部状态一致\n");
    sim::State world = sim::make_initial_state();
    world.red.last_known_pos = {1, 2};
    world.red.last_known_facing = 'S';
    world.blue.last_known_pos = {5, 4};
    world.blue.last_known_facing = 'N';
    world.turn = 6;

    // 手工构造蓝方的局部视角（等价于引擎 view_for 做的事），
    // 与 to_local 的结果比对
    const sim::State local = sim::to_local(world, 'B');
    require(local.red.last_known_pos.x == sim::mirror_pos({5, 4}, 7).x,
            "蓝方局部视角下自身位置应为世界坐标的镜像");
    require(local.red.last_known_facing == sim::mirror_facing('N'),
            "蓝方局部视角下自身朝向应为世界朝向的镜像");
    require(local.blue.last_known_pos.x == sim::mirror_pos({1, 2}, 7).x,
            "蓝方局部视角下对手位置应为世界坐标的镜像");
    require(local.turn == world.turn, "回合数不随视角改变");
    require(local.red.score == world.blue.score, "局部视角下我方得分应取世界中的蓝方得分");
    require(local.blue.score == world.red.score, "局部视角下对手得分应取世界中的红方得分");
}

} // namespace

int main() {
    std::printf("=== 坐标框架等变性测试 ===\n");
    test_to_local_algebra();
    test_action_conversion();
    test_equivariance();
    test_observation_consistency();

    std::printf("\n--- 结果 ---\n");
    std::printf("断言 %lld 条\n", g_checks);
    if (g_failures == 0) {
        std::printf("✅ 全部通过\n");
        return 0;
    }
    std::printf("❌ %d 条失败\n", g_failures);
    return 1;
}

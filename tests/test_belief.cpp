// tests/test_belief.cpp —— 敌方可达集信念的单元测试
//
// 信念是纯逻辑，可以直接断言，不需要跑对局。这里覆盖的四条规则
// 各对应一类会静默出错的写法：
//   ① 扩张必须绕障碍、且不含我方所在格
//   ② 视野证伪必须只剔除"当前看得见"的格子
//   ③ 塌缩（看见/扫描/击杀重生）必须精确到一格
//   ④ 火力覆盖判定必须与引擎的通道遮挡规则一致（复用 sim::fire_hit）

#include "sim/belief.h"
#include "sim/rules.h"

#include <cstdio>
#include <cstdlib>

namespace {

int g_failures = 0;

void require(bool cond, const char* msg) {
    if (cond) return;
    ++g_failures;
    std::printf("  [FAIL] %s\n", msg);
}

sim::State base_state() { return sim::make_initial_state(); }

// 用带障碍的官方地图：{1,1} 与 {5,5}
void test_dilate() {
    std::printf("[1/4] 扩张（BFS ≤3，绕障碍，不含我方格）\n");
    sim::State s = base_state();
    const Pos my_pos{0, 0};

    sim::Belief b;
    b.reset_to({6, 6}); // 敌方出生点（局部坐标）
    sim::dilate(b, 3, my_pos, s.obstacles);

    require(b.has(6, 6), "起点应仍在集合内");
    require(b.has(3, 6), "距起点 3 格应可达");
    require(b.has(6, 3), "距起点 3 格（纵向）应可达");
    require(!b.has(2, 6), "距起点 4 格不该可达");
    require(!b.has(6, 2), "距起点 4 格（纵向）不该可达");
    require(!b.has(5, 5), "障碍格不应在集合内");
    require(!b.has(0, 0), "我方所在格不应在集合内");

    // 扩张必须绕障碍：把起点放在 (4,5)，一步内不能穿过 (5,5) 到达 (6,5)
    sim::Belief c;
    c.reset_to({4, 5});
    sim::dilate(c, 1, my_pos, s.obstacles);
    require(c.has(4, 5) && c.has(3, 5) && c.has(4, 4) && c.has(4, 6),
            "一步扩张应覆盖四邻");
    require(!c.has(5, 5), "不能扩张到障碍格");

    // 起点只有一个格子时，扩张后大小应为该格 3 步内的可达格数（不含障碍与我方）
    std::printf("      起点 (6,6) 扩张 3 步后 |集合| = %d\n", b.count());
    require(b.count() > 1, "扩张后应多于一个格子");
}

void test_prune_by_vision() {
    std::printf("[2/4] 视野证伪（只剔除当前看得见的格子）\n");
    sim::State s = base_state();

    Sentry me{};
    me.last_known_pos = {0, 0};
    me.last_known_facing = 'E';
    // 朝 E 的 T 形视野：正前 1 格 (1,0)；距离 2 的横向 3 格 (2,-1)/(2,0)/(2,1)
    // 其中 (2,-1) 出界，所以可见格是 (1,0)、(2,0)、(2,1)

    sim::Belief b;
    b.set(1, 0); // 正前 1 格，无遮挡 → 可见
    b.set(2, 0); // 距离 2 中心格；中间格 (1,0) 无遮挡 → 可见
    b.set(2, 1); // 距离 2 下侧格；斜前中间格 (1,1) **是障碍** → 被遮挡，不可见
    b.set(4, 4); // 视野外
    b.set(0, 3); // 侧后方

    sim::prune_by_vision(b, me, s.obstacles);
    require(!b.has(1, 0), "正前方(距离1)可见格应被剔除");
    require(!b.has(2, 0), "距离2中心格(中间无障碍)应被剔除");
    // 这条是引擎遮挡规则里最容易搞错的一条：各条视线独立判定，
    // 距离 2 的侧格走的是斜前中间格，不是正前方那一格。
    require(b.has(2, 1), "距离2下侧格被障碍(1,1)遮挡，应保留在信念里");
    require(b.has(4, 4), "视野外的格子必须保留");
    require(b.has(0, 3), "侧后方格子必须保留（本游戏没有后向视野）");

    // 障碍遮挡：放一个障碍在正前方，则距离 2 的中心格不应被判为可见
    sim::State blocked = base_state();
    blocked.obstacles.push_back({1, 0}); // 正前方 1 格
    sim::Belief c;
    c.set(2, 0); // 距离 2 的中心格，被 (1,0) 挡住
    sim::prune_by_vision(c, me, blocked.obstacles);
    require(c.has(2, 0), "被障碍挡住的距离 2 中心格不该被判为可见");
}

void test_fire_lane() {
    std::printf("[3/4] 火力覆盖判定（与引擎通道遮挡一致）\n");
    sim::State s = base_state();

    Sentry me{};
    me.last_known_pos = {3, 1};
    me.last_known_facing = 'S'; // 朝下，火力覆盖 (2..4, 2..4)

    // 全部落在火力范围内 → 开火必中
    sim::Belief inside;
    inside.set(3, 2);
    inside.set(4, 3);
    require(sim::all_in_fire_lane(inside, me, s.obstacles),
            "两个格子都在 3x3 内应判为必中");
    require(sim::any_in_fire_lane(inside, me, s.obstacles), "任一在范围内应判为可打");

    // 有一个在范围外 → 不是必中
    sim::Belief mixed;
    mixed.set(3, 2);
    mixed.set(0, 0);
    require(!sim::all_in_fire_lane(mixed, me, s.obstacles),
            "含范围外格子时不应判为必中");
    require(sim::any_in_fire_lane(mixed, me, s.obstacles), "仍应判为可打");

    // 空集合：两个判定都应为假（不能因为"没有反例"就判真）
    const sim::Belief empty;
    require(!sim::all_in_fire_lane(empty, me, s.obstacles), "空集合不该判为必中");
    require(!sim::any_in_fire_lane(empty, me, s.obstacles), "空集合不该判为可打");

    // 超出距离 3 的格子不算
    sim::Belief far;
    far.set(3, 5); // 距离 4
    require(!sim::any_in_fire_lane(far, me, s.obstacles), "距离 4 的格子不在火力范围内");
}

void test_nearest_and_helpers() {
    std::printf("[4/4] 辅助：count / nearest_to / reset_to\n");
    sim::Belief b;
    require(b.empty(), "默认构造应为空");
    require(b.count() == 0, "默认构造 count 应为 0");

    b.set(1, 1);
    b.set(5, 5);
    require(b.count() == 2, "count 应为 2");
    require(b.has(Pos{1, 1}) && b.has(Pos{5, 5}), "has(Pos) 应与 set 一致");

    const Pos near = b.nearest_to({0, 0});
    require(near.x == 1 && near.y == 1, "nearest_to 应取曼哈顿距离最近者");
    const Pos near2 = b.nearest_to({6, 6});
    require(near2.x == 5 && near2.y == 5, "nearest_to 换一个观察点应换结果");

    // 越界坐标不得越界写入
    b.set(Pos{-1, 0});
    b.set(Pos{7, 7});
    require(b.count() == 2, "越界坐标应被忽略");

    b.reset_to({3, 3});
    require(b.count() == 1 && b.has(3, 3), "reset_to 应清空后只留一格");
}

} // namespace

int main() {
    std::printf("=== 敌方可达集信念 单元测试 ===\n");
    test_dilate();
    test_prune_by_vision();
    test_fire_lane();
    test_nearest_and_helpers();

    std::printf("\n--- 结果 ---\n");
    if (g_failures == 0) {
        std::printf("✅ 全部通过\n");
        return 0;
    }
    std::printf("❌ %d 项失败\n", g_failures);
    return 1;
}

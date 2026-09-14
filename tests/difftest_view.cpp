// tests/difftest_view.cpp —— 对局层差分测试：选手视图（Intel）必须与引擎一致
//
// 这是自对弈训练管线的**前置门槛**。
//
// 为什么非测不可：sim::State 里 last_known_pos 是**真实位置**，而引擎发给选手的
// 是 Intel（记忆，开局是 {-1,-1}）。自对弈若直接用世界状态构造观测，就是把真实
// 位置喂给网络——信息泄漏，而且**不会报错**，只表现为"训出来的策略在真机上莫名
// 其妙地差"。这类失败没法靠看训练曲线发现。
//
// 做法：让引擎驱动真实 Match 跑一局（策略由剧本驱动），同时在每个决策点用
// match->board() 的**世界真值**推进 sim::ViewMirror，然后逐字段比对：
//   · 阶段开头：引擎交给 act() 的那份 Board
//   · 每次行动后：ActionResult.observation
//
// 用引擎的世界真值来推进我们这边，是为了让两边严格同步——否则测出来的差异
// 分不清是 ViewMirror 错了还是我们的规则推演错了（规则推演另有 487 万次比对的
// 差分测试覆盖，见 tests/difftest_rules.cpp）。

#include "sim/rules.h"
#include "sim/view_mirror.h"

#include "match.h"        // 引擎（只读引用）
#include "sentry_duel.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;
int g_phase = 0;
int g_ok = 0;
int g_failed = 0;

void check(bool cond, const std::string& what) {
    ++g_checks;
    if (!cond) {
        ++g_failures;
        if (g_failures <= 20) std::printf("  ❌ %s\n", what.c_str());
    }
}

std::string fmt_pos(const Pos& p) {
    char b[32];
    std::snprintf(b, sizeof b, "(%d,%d)", p.x, p.y);
    return b;
}

bool same_pos(const Pos& a, const Pos& b) { return a.x == b.x && a.y == b.y; }

// —— 剧本 ——
struct Step {
    int action;
    char arg;
};

// 双方各一串动作，按顺序在自己的行动阶段内发出（每阶段最多 3 个）。
std::vector<Step> g_script[2];
size_t g_next[2] = {0, 0};

// 引擎世界真值 → sim::State（字段布局同构，见 sim/state.h）
sim::State to_world(const Board& b) {
    sim::State w;
    w.red = b.red;
    w.blue = b.blue;
    w.turn = b.turn;
    w.size = b.size;
    w.obstacles = b.obstacles;
    w.score_zones = b.score_zones;
    return w;
}

sim::ViewMirror g_mirror;
sentry::Match* g_match = nullptr;
sim::State g_world;

void cmp_view(const Board& v, char side, const char* when) {
    const sim::State ours = g_mirror.local_view(g_world, side);
    const std::string tag = std::string(when) + " side=" + side;

    // 【坐标约定差异】引擎的 view_for 只把**坐标**镜像，红蓝仍留在各自的槽位里；
    // sim::to_local 则会把红蓝槽位**交换**，让"我"恒在 red。
    // 所以这里必须按 side 取，不能一律读 .blue。
    const Sentry& eng_me = (side == 'R') ? v.red : v.blue;
    const Sentry& eng_enemy = (side == 'R') ? v.blue : v.red;

    check(same_pos(eng_enemy.last_known_pos, ours.blue.last_known_pos),
          tag + " 敌方 last_known_pos: 引擎" + fmt_pos(eng_enemy.last_known_pos) +
              " vs 我们" + fmt_pos(ours.blue.last_known_pos));
    check(eng_enemy.last_known_facing == ours.blue.last_known_facing,
          tag + " 敌方 last_known_facing 不一致");
    check(eng_enemy.visible == ours.blue.visible, tag + " 敌方 visible 不一致");
    check(eng_enemy.fire_cd == ours.blue.fire_cd, tag + " 敌方 fire_cd 不一致");
    check(same_pos(eng_me.last_known_pos, ours.red.last_known_pos),
          tag + " 我方位置不一致");
    check(eng_me.last_known_facing == ours.red.last_known_facing,
          tag + " 我方朝向不一致");
    check(v.turn == ours.turn, tag + " 回合号不一致");
}

void cmp_obs(const ActionObservation& o, char side, const char* when) {
    const sim::State ours = g_mirror.local_view(g_world, side);
    const std::string tag = std::string(when) + " side=" + side;

    check(same_pos(o.opp_last_known_pos, ours.blue.last_known_pos),
          tag + " obs.opp_last_known_pos: 引擎" + fmt_pos(o.opp_last_known_pos) +
              " vs 我们" + fmt_pos(ours.blue.last_known_pos));
    check(o.opp_last_known_facing == ours.blue.last_known_facing,
          tag + " obs.opp_last_known_facing 不一致");
    check(o.opp_visible == ours.blue.visible, tag + " obs.opp_visible 不一致");
    check(same_pos(o.my_pos, ours.red.last_known_pos), tag + " obs.my_pos 不一致");
}

int policy(const Board& view, char side) {
    const int si = (side == 'R') ? 0 : 1;
    ++g_phase;

    g_world = to_world(g_match->board());
    g_mirror.begin_phase(g_world, side);
    cmp_view(view, side, "阶段开头");

    for (int k = 0; k < 3; ++k) {
        if (g_next[si] >= g_script[si].size()) break;
        const Step st = g_script[si][g_next[si]++];

        ActionResult r{};
        bool is_scan = false;
        switch (st.action) {
            case sim::kMove: r = move(); break;
            case sim::kTurn: r = turn(st.arg); break;
            case sim::kFire: r = fire(); break;
            default: {
                const ScanResult s = scan();
                r.success = s.success;
                r.consumed = s.consumed;
                r.observation = s.observation;
                is_scan = true;
                break;
            }
        }
        (void)is_scan;
        if (!r.success) {
            // 剧本只是一串"试试这些动作"的建议：某个动作在当前局面下不可行
            // （撞障碍、CD 没好）是正常的，不是被测代码的问题。
            // 失败的行动不消耗额度，也**不更新情报**（引擎在失败分支直接返回，
            // 不走 refresh_current_vision），所以这里什么都不做，继续下一个。
            ++g_failed;
            continue;
        }
        ++g_ok;

        g_world = to_world(g_match->board());
        g_mirror.after_action(g_world, side, st.action);
        cmp_obs(r.observation, side, "行动后");
    }
    return 0;
}

} // namespace

int main() {
    // 剧本设计：刻意覆盖 Intel 的三种更新路径与一个**不该更新**的路径。
    //
    //   1. 开局未见过 -> known=false，敌方位置必须是 {-1,-1}（不是出生点！）
    //   2. SCAN 成功 -> 无条件 remember
    //   3. 走出视野后移动 -> Intel 保持不动（记忆，不是真值）
    //   4. FIRE 命中 -> **Intel 不重置**，仍停在开枪时的位置
    //      （引擎只置 respawn_turn_pending，那只影响对方下次的免费转向）
    const Step red[] = {
        {sim::kScan, 0},   // 1: 开局扫描，建立情报
        {sim::kMove, 0},   // 逼近
        {sim::kMove, 0},
        {sim::kTurn, 'S'}, // 转向得分区方向
        {sim::kMove, 0},
        {sim::kFire, 0},   // 尝试开火（可能命中，也可能不中）
        {sim::kMove, 0},
        {sim::kScan, 0},
        {sim::kMove, 0},
    };
    const Step blue[] = {
        {sim::kMove, 0},
        {sim::kTurn, 'N'},
        {sim::kMove, 0},
        {sim::kMove, 0},
        {sim::kScan, 0},
        {sim::kMove, 0},
        {sim::kMove, 0},
        {sim::kMove, 0},
        {sim::kMove, 0},
    };
    for (const Step& s : red) g_script[0].push_back(s);
    for (const Step& s : blue) g_script[1].push_back(s);

    g_mirror.reset();
    sentry::Match m(policy, policy, "ours", "ours", /*max_turns=*/6, /*game_id=*/0,
                    [](const std::string&) {}); // 吞掉 JSON 事件，别刷屏
    g_match = &m;
    m.run();

    std::printf("差分比对 %d 次, %d 处不一致；剧本动作 成功 %d / 失败 %d\n",
                g_checks, g_failures, g_ok, g_failed);
    // 成功动作太少说明剧本基本没跑起来，那样的"通过"没有意义
    check(g_ok >= 10, "成功执行的剧本动作太少，测试覆盖不足");
    if (g_failures == 0) std::printf("✅ 选手视图与引擎完全一致\n");
    return g_failures == 0 ? 0 : 1;
}

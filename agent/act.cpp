// agent/act.cpp —— 选手 .so 的 act() 入口（阶段①：搜索 + policy model）
//
// 一个 act() 内的流程：
//   ① 把引擎给的视图转成「我方为 'R'」的局部坐标状态（引擎已对蓝方镜像，
//      所以红蓝双方共用同一份策略代码，不需要任何按颜色的分支）
//   ② 逐步决策：每执行一个行动前都重新规划一次（搜索很快，重新规划最省心）
//      —— 这样"最后已知敌情"与真实的任何偏差都会被下一步立刻纠正，
//      而不是把一份基于错误前提的计划执行到底
//   ③ 每次行动后用真实观测校验 sim 的预测；不符则用真实观测重建状态
//   ④ 任何异常/超时都退回绝对安全的启发式
//
// 时间预算是硬约束：引擎给 1 秒，超时 = 本回合剩余行动作废 + 送对手 1 分。

#include "brain/belief_state.h"
#include "brain/eval.h"
#include "brain/mcts.h"
#include "brain/net.h"
#include "brain/opening_book.h"
#include "brain/policy_net.h"
#include "brain/search.h"
#include "obs/encode_v3.h"
#include "sim/belief.h"
#include "sim/rules.h"
#include "sentry_duel.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

using Clock = std::chrono::steady_clock;

// 引擎硬性 1 秒，留足余量。
//
// 这个值现在同时决定**部署时的对局耗时**：MCTS 会用满预算，
// 350ms/act ≈ 7 秒一局，而平台打榜要对榜内每个对手各打 20 局。
// 降到 150ms 后约 3 秒一局，搜索深度只浅一点——性价比更高。
// phase① 的穷举搜索只需几百微秒，根本用不到这个预算。
// 【必须 ≤150ms】平台的 timeout=30s 是**整个对局子进程**的时限，含对手
// 的思考时间。本地对局测不出这个坑：baseline/hunter 思考 ≈0s，但平台
// 对手是别人的 AI——400ms/act 时我方 ~24s + 对手思考时间会撞爆 30s，
// 子进程被杀 = 整局判负（实测：combo 包上线排名掉一名）。
// 150ms × 两 AI × 最坏 60 act ≈ 18s，留足余量。600 局验证的调优收益
// （w_ready=0.6）全部来自这个预算档，不要动。
constexpr int kDefaultBudgetMs = 150;
constexpr int kMaxFailedAttempts = 4;

int env_int(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return fallback;
    const int n = std::atoi(v);
    return n > 0 ? n : fallback;
}

bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && *v != '\0' && *v != '0';
}

const int g_budget_ms = env_int("ST_BUDGET_MS", kDefaultBudgetMs);
const bool g_debug = env_flag("ST_DEBUG");

// 策略切换：默认走 phase① 的本回合搜索；ST_MCTS=1 走 MCTS。
// 用开关而不是直接替换，是为了能同口径 A/B 对比，而不是假设"复杂方法一定更好"。
// 部署默认走 phase① 的穷举搜索。
//
// MCTS 会用满时间预算（150ms/act ≈ 3 秒一局，比 phase① 慢两个数量级），
// 而实测它的强度并没有兑现——本地对确定性对手 10:13，反而不如 phase① 的
// 21:10。所以默认不再启用，只保留 ST_MCTS=1 作为离线实验开关。
// 代价：网络目前只在 MCTS 路径上被读取，默认路径下不参与决策。
const bool g_use_mcts = env_flag("ST_MCTS");

// 阶段③：策略网络直接决策（不搜索）。
//
// 与 MCTS 路径的根本差别在**观测口径**：策略网络吃的是 obs v3，它不含手工
// 信念，敌方位置用的是**引擎给的 Intel**。而搜索路径会调 apply_belief() 用
// 我们自己的可达集推断覆盖掉那个值。两条路径的输入不同，绝不能混用——
// 混了就是训练/部署漂移，而且不会报错。
//
// 部署默认仍走阶段①。这条路径要先用分色评测证明更强，才能改默认。
const bool g_use_policy = env_flag("ST_POLICY");

// 被击中推断"对手刚开火、正无力"的窗口。默认关闭：检测条件（从非出生点
// 变为出生点）与"自己走回出生点"无法区分，假阳性会让我们误信对手 CD=2
// 而冒进。实验用 ST_OPP_WINDOW=1 开启（且仅执蓝生效，蓝方窗口收益大）。
const bool g_opp_window = env_flag("ST_OPP_WINDOW");

// 值蒸馏叶评估（阶段③ → phase① 的成果回接）。ST_VALUE_NET=1 启用。
double env_double(const char* name, double fallback); // 定义见下方
const bool g_use_value_net = env_flag("ST_VALUE_NET");
const double g_value_scale = env_double("ST_VALUE_SCALE", 30.0);
// 两回合前瞻（深模式）。ST_DEEP=1 启用，ST_DEEP_NODES 调预算。
const int g_deep_turns = env_int("ST_DEEP", 0);

double env_double(const char* name, double fallback); // 定义见下方

brain::MctsConfig load_mcts_config() {
    brain::MctsConfig c;
    c.c_puct = env_double("ST_MCTS_CPUCT", c.c_puct);
    c.max_depth = env_int("ST_MCTS_DEPTH", c.max_depth);
    return c;
}
const brain::MctsConfig g_mcts_cfg = load_mcts_config();

// 权重可通过环境变量覆盖，用于离线调参；未设置时用编译期默认值。
// 平台评测时环境里没有这些变量，所以线上行为完全由默认值决定。
double env_double(const char* name, double fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return fallback;
    char* end = nullptr;
    const double d = std::strtod(v, &end);
    return (end != nullptr && end != v) ? d : fallback;
}

brain::Weights load_weights() {
    brain::Weights w;
    w.w_diff = env_double("ST_W_DIFF", w.w_diff);
    w.w_zone = env_double("ST_W_ZONE", w.w_zone);
    w.w_dist = env_double("ST_W_DIST", w.w_dist);
    w.w_threat = env_double("ST_W_THREAT", w.w_threat);
    w.w_danger = env_double("ST_W_DANGER", w.w_danger);
    w.w_ready = env_double("ST_W_READY", w.w_ready);
    w.w_waste = env_double("ST_W_WASTE", w.w_waste);
    w.w_uncertain = env_double("ST_W_UNCERTAIN", w.w_uncertain);
    w.w_danger_red_scale = env_double("ST_W_DANGER_RED", w.w_danger_red_scale);
    return w;
}

brain::Weights g_weights = load_weights();

struct Memory {
    int last_turn = -1;
    Pos last_my_pos{-1, -1};
    // 敌方位置信念。实现在 brain/SideBelief，与自对弈侧共用一份——
    // 信念是观测的一部分，两份实现漂移会让训练输入与部署输入不一致。
    brain::SideBelief belief;

    // 被击中推断出的对手无力窗口（绝对回合号）。见 brain/search.h 的说明。
    int opp_defenseless_until = -1;

    // 阶段③ 策略网络的隐状态。必须**跨 act() 调用存活**——它就是网络对
    // 部分可观测历史的记忆。新对局时必须清零（见 run() 开头的重置）。
    float policy_hidden[brain::kPolicyHidden] = {};
};
Memory g_mem;

// 从引擎视图构造「我方为 'R'」的局部状态。
// 引擎已对蓝方做 180° 镜像，所以蓝方看到的 board.blue 就是局部坐标下的自己。
sim::State local_state(const Board& b, char side) {
    sim::State st = sim::make_initial_state(b.size);
    st.turn = b.turn;
    st.obstacles = b.obstacles;
    st.score_zones = b.score_zones;
    st.red = (side == 'R') ? b.red : b.blue;   // 我方
    st.blue = (side == 'R') ? b.blue : b.red;  // 对手
    // sim::make_initial_state 会写入默认地图，这里必须覆盖成真实地图
    return st;
}

// 把信念状态写回对局状态，供搜索使用。
//
// 引擎的情报只在"看得见"时更新；而我们击杀对手后能推算出他回了出生点。
// 所以我们的锚点往往比引擎的情报更准，只在从未得知时才用出生点兜底。
void apply_belief(sim::State& st, int turn) {
    // 搜索里用一个"代表位置"来推演：取锚点（最后已知）。
    // 不确定性由评估函数对信念取期望来体现，两者分工不同。
    g_mem.belief.write_anchor_into(st);
    // 对手的 CD 引擎不暴露（恒为 -1），搜索里按保守值处理；
    // 唯一例外：被击中推断出的无力窗口（对手刚开过火，是确定性情报）
    st.blue.fire_cd = (turn <= g_mem.opp_defenseless_until)
                          ? 2
                          : brain::kAssumedEnemyFireCd;
    st.blue.scan_cd = 0;
}

bool same_my_state(const Sentry& predicted, const ActionObservation& obs) {
    return predicted.last_known_pos.x == obs.my_pos.x &&
           predicted.last_known_pos.y == obs.my_pos.y &&
           predicted.last_known_facing == obs.my_facing &&
           predicted.fire_cd == obs.fire_cd && predicted.scan_cd == obs.scan_cd;
}

struct Executed {
    bool success = false;
    bool consumed = false;
    ActionObservation obs{};
};

Executed execute(int action, char world_arg) {
    Executed e;
    if (action == sim::kMove) {
        const ActionResult r = move();
        e.success = r.success;
        e.consumed = r.consumed;
        e.obs = r.observation;
    } else if (action == sim::kTurn) {
        const ActionResult r = turn(world_arg);
        e.success = r.success;
        e.consumed = r.consumed;
        e.obs = r.observation;
    } else if (action == sim::kFire) {
        const ActionResult r = fire();
        e.success = r.success;
        e.consumed = r.consumed;
        e.obs = r.observation;
    } else {
        const ScanResult r = scan();
        e.success = r.success;
        e.consumed = r.consumed;
        e.obs = r.observation;
    }
    return e;
}

// ── 开局书查表 ──
// 书键是世界坐标的求解器状态。部署时 Board 在局部帧（蓝方已镜像），
// 需要转回世界帧再编码查表。开局期 (turn<4) 双方位置公开，无信息缺口。
int book_action(const Board& board, char my_color, int used, bool free_turn) {
    if (!brain::book_available() || board.turn >= brain::book_turns()) return -1;
    const bool i_am_red = (my_color == 'R');

    // 局部帧 → 世界帧：蓝方镜像（mirror 自反）
    const auto to_world_pos = [&](const Pos& p) -> Pos {
        return i_am_red ? p : sim::mirror_pos(p, board.size);
    };
    const auto to_world_f = [&](char f) -> char {
        return i_am_red ? f : sim::mirror_facing(f);
    };

    const Sentry& my_s = i_am_red ? board.red : board.blue;
    const Sentry& opp_s = i_am_red ? board.blue : board.red;

    const auto encode = [&](const Pos& p, char f, int fcd, int scd) -> int {
        const int cell = p.y * 7 + p.x;
        int fi = 0;
        for (int i = 0; i < 4; ++i) { if ("NESW"[i] == f) fi = i; }
        return ((cell * 4 + fi) * 12) + (fcd * 4 + scd);
    };

    const int pr = encode(to_world_pos(my_s.last_known_pos),
                          to_world_f(my_s.last_known_facing),
                          my_s.fire_cd, my_s.scan_cd);
    const int pb = encode(to_world_pos(opp_s.last_known_pos),
                          to_world_f(opp_s.last_known_facing),
                          opp_s.fire_cd, opp_s.scan_cd);
    const int sidx = i_am_red ? 0 : 1;
    const int ff = (free_turn && i_am_red ? 2 : 0) | (free_turn && !i_am_red ? 1 : 0);
    const int diff = board.red.score - board.blue.score;

    const int best = brain::book_lookup(pr, pb, board.turn, sidx, used, ff, diff);
    if (best < 0) return -1;
    // best 是 solver 的动作下标（gen_moves 序：move/turn×3/fire/scan）
    // 或 254（收手）。转成 brain::kActionDim 空间。
    if (best == 254) return 7;
    // gen_moves 序: 0=move, 1..3=turn（跳过当前朝向）, 4=fire, 5=scan
    // brain 动作序: 0=move, 1..4=turn NESW, 5=fire, 6=scan, 7=stop
    // 需要从世界帧的动作反查局部帧的动作。
    // 因为 solver 用世界帧朝向而部署用局部帧朝向，转向的映射不同。
    // 简化：书条目只记录了下标——重新从 gen_moves 恢复动作。
    // 这里直接用 cand_to_index 反查不可靠（序不同），改为重新枚举。
    // gen_moves 序与 solve 的相同：按当前朝向剔除同向转向。
    // 行动方的朝向来自 Board（局部帧），但 solver 的 gen_moves 也是局部帧的
    // （Board 的朝向就是局部帧朝向），所以直接枚举即可。
    {
        // 行动方的朝向（局部帧）：找当前 acting side 的朝向
        const char acting_facing = i_am_red ? board.red.last_known_facing
                                            : board.blue.last_known_facing;
        std::vector<std::pair<int, char>> seq = {
            {0, 0}, {1, 'N'}, {1, 'E'}, {1, 'S'}, {1, 'W'}, {2, 0}, {3, 0}};
        int j = 0;
        for (int i = 0; i < static_cast<int>(seq.size()); ++i) {
            if (seq[i].first == 1 && seq[i].second == acting_facing) continue;
            if (j == best) {
                // solver 的动作 → brain 动作下标。
                // move/fire/scan 是同动作。turn 的 arg（世界帧方向）需要
                // 转回局部帧：蓝方时 N↔S、E↔W 对调。
                const brain::Cand wc{seq[i].first, seq[i].second};
                if (seq[i].first == sim::kTurn && !i_am_red) {
                    // 蓝方的局部帧朝向 = 世界帧朝向的镜像
                    char lf = seq[i].second;
                    if (lf == 'N') lf = 'S'; else if (lf == 'S') lf = 'N';
                    else if (lf == 'E') lf = 'W'; else if (lf == 'W') lf = 'E';
                    const brain::Cand lc{sim::kTurn, lf};
                    return brain::cand_to_index(lc);
                }
                return brain::cand_to_index(wc);
            }
            ++j;
        }
    }
    return -1;
}

char side_from_idx(int idx) { return idx == 0 ? 'R' : 'B'; }

void run(const Board& board, char my_color) {
    const Clock::time_point deadline = Clock::now() + std::chrono::milliseconds(g_budget_ms);

    // 新对局：重置跨回合记忆（同进程只跑一局，这里是防御性写法）
    if (board.turn == 0 || board.turn < g_mem.last_turn) {
        g_mem = Memory{};
        brain::policy_reset_hidden(g_mem.policy_hidden);
    }

    sim::State st = local_state(board, my_color);
    bool enemy_visible = st.blue.visible;

    // 免费转向估计：开局，或刚从别处被击回出生点。
    // 引擎不暴露该状态，但"位置突然变成出生点"是唯一能瞬移的情形，可据此判定。
    const Pos my_spawn = sim::spawn_of('R', st.size);
    const bool at_spawn =
        (st.red.last_known_pos.x == my_spawn.x && st.red.last_known_pos.y == my_spawn.y);
    const bool was_at_spawn =
        (g_mem.last_my_pos.x == my_spawn.x && g_mem.last_my_pos.y == my_spawn.y);
    const bool just_respawned = at_spawn && !was_at_spawn && board.turn > 0;
    bool free_turn = (board.turn == 0) || just_respawned;

    // 【被击中 = 对手刚开火】推断其无力窗口。我方执蓝时打我们的是红方
    // （开火后 2 个窗口无力），执红时是蓝方（1 个窗口）——不对称再次来自
    // CD 只在蓝方阶段后递减。窗口过期前搜索允许我们安全压近。
    if (just_respawned && g_opp_window && my_color == 'B') {
        g_mem.opp_defenseless_until = board.turn + 1;
    }

    // —— 敌方位置信念的维护（与自对弈侧共用 brain/SideBelief）——
    g_mem.belief.begin_turn(st, board.turn, enemy_visible);
    // 策略路径不覆盖敌方位置：obs v3 要的是引擎原样的 Intel，
    // 用我们的可达集推断覆盖它就是喂给网络一个训练时见不到的输入。
    if (!g_use_policy) apply_belief(st, board.turn);

    brain::TurnInput in;
    int used = 0;
    int failures = 0;

    // 信息优先：看不见对手且雷达就绪，就先扫。
    //
    // 为什么不让搜索自己决定？因为在"点为信念"下搜索**无法给 SCAN 定价**：
    // 模拟中扫描只改 scan_cd、不改变敌方位置，于是它在评估函数眼里是纯亏，
    // 永远不会被选中。而实战里不扫描就等于闭着眼走进对手的枪口。
    // 阶段③ 的 RNN 能从历史维持信念，届时这条规则会被真正的信息价值取代。
    // 只有在"位置不再确定"时才值得开雷达。开局时信念是一个格子，但那是
    // **假设**（对手在出生点）而不是知情，所以同样要扫。
    if (!enemy_visible && st.red.scan_cd == 0 && !g_mem.belief.certain) {
        const Executed e = execute(sim::kScan, 0);
        if (e.success) {
            if (e.consumed) ++used;
            st.red.scan_cd = e.obs.scan_cd;
            if (e.obs.opp_visible) {
                st.blue.last_known_pos = e.obs.opp_last_known_pos;
                st.blue.last_known_facing = e.obs.opp_last_known_facing;
                st.blue.visible = true;
                enemy_visible = true;
                g_mem.belief.collapse(e.obs.opp_last_known_pos,
                                      e.obs.opp_last_known_facing);
            }
        }
    }

    while (used < brain::kMaxActions && Clock::now() < deadline) {
        in.state = st;
        in.belief = g_mem.belief.set;
        in.used_by_now = used;
        in.free_turn_available = free_turn;
        in.enemy_visible = enemy_visible;
        // 【关键】评估的 CD 时序不对称项需要知道世界颜色。这是整个 AI 里
        // 唯一一处"按颜色分支"——它不是坐标分支（坐标已由镜像抹平），
        // 而是引擎结算顺序本身的不对称（CD 只在蓝方阶段后递减）。
        in.acts_first_world = (my_color == 'R');
        in.use_value_net = g_use_value_net;
        in.value_scale = g_value_scale;
        in.deep_turns = g_deep_turns;

        brain::SearchStats stats;
        brain::Plan plan;
        if (g_use_policy) {
            // 策略网络直接出动作：obs v3 → GRU → argmax
            float obs_v3[obs::kObsDimV3];
            obs::encode_v3(st, enemy_visible, used, free_turn, obs_v3);
            float logits[brain::kPolicyActDim];
            float value = 0.0f;
            if (brain::policy_forward(obs_v3, g_mem.policy_hidden, logits, &value)) {
                int best = 0;
                for (int i = 1; i < brain::kPolicyActDim; ++i) {
                    if (logits[i] > logits[best]) best = i;
                }
                // 部署取 argmax 而不是采样：单局要的是最可能的那手，
                // 不是从分布里抽一发（自对弈侧才需要噪声保证多样性）
                const brain::Cand c = brain::index_to_cand(best);
                if (c.action == brain::kStopAction) {
                    plan.valid = false;   // 收手 = 本阶段结束
                } else {
                    plan.valid = true;
                    plan.count = 1;
                    plan.actions[0] = c.action;
                    plan.args[0] = c.arg;
                    plan.value = value;
                }
            }
        } else if (g_use_mcts) {
            // 本 act 内每步都会重新规划，所以把剩余时间按剩余步数**均分**，
            // 否则第一步会把整个 act 的预算吃光。
            const int steps_left =
                brain::kMaxActions - used > 0 ? brain::kMaxActions - used : 1;
            const auto now = Clock::now();
            const auto span = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  deadline - now)
                                  .count();
            const auto share = std::chrono::milliseconds(
                std::max<long long>(1, span / steps_left));
            const brain::MctsResult r =
                brain::mcts_search(in, g_weights, g_mcts_cfg, now + share);
            plan = r.plan;
        } else {
            plan = brain::search_turn(in, g_weights, deadline, &stats);
        }
        if (g_debug) {
            const char* kNames[] = {"move", "turn", "fire", "scan"};
            const int a0 = plan.count > 0 ? plan.actions[0] : -1;
            std::fprintf(stderr,
                         "[ai] side=%c turn=%d used=%d vis=%d free=%d me=(%d,%d)%c "
                         "bel=(%d,%d)%c n=%d our=%lld opp=%lld refined=%lld exh=%d "
                         "value=%.3f plan=%d act=%s%c\n",
                         my_color, board.turn, used, static_cast<int>(enemy_visible),
                         static_cast<int>(free_turn), st.red.last_known_pos.x,
                         st.red.last_known_pos.y, st.red.last_known_facing,
                         st.blue.last_known_pos.x, st.blue.last_known_pos.y,
                         st.blue.last_known_facing, g_mem.belief.set.count(),
                         stats.our_nodes, stats.opp_nodes,
                         stats.refined, static_cast<int>(stats.time_exhausted), plan.value,
                         plan.count,
                         a0 >= 0 ? kNames[a0] : "none",
                         (a0 == sim::kTurn && plan.count > 0) ? plan.args[0] : ' ');
        }
        if (!plan.valid || plan.count == 0) break;

        const int action = plan.actions[0];
        const char local_arg = plan.args[0];
        // 【重要】turn() 接收的是**局部**朝向：引擎在 Match::do_action 内部
        // 自己会做 world_facing(local_arg, side) 的镜像转换（match.cpp:240）。
        // 这里若再镜像一次，蓝方就会被反向转 180°，整局报废。
        const char world_arg = local_arg;

        // 先用 sim 推演期望结果：这样分数、CD、以及"击杀后对手回出生点"
        // 都能被正确推进，比只看观测更完整
        sim::State predicted = st;
        const sim::Outcome planned = sim::apply_action(predicted, 'R', action, local_arg);

        const Executed e = execute(action, world_arg);

        if (!e.success) {
            // 失败不消耗额度。屏蔽掉这个动作，否则重规划会反复选中它。
            in.ban(action, local_arg);
            ++failures;

            // 移动失败 = 目标格被我们看不见的对手占着 —— 这是关于敌位最强的情报，
            // 直接把信念塌缩到那一格。
            if (action == sim::kMove) {
                int dx = 0;
                int dy = 0;
                sim::facing_delta(st.red.last_known_facing, dx, dy);
                // 目标格被对手占着 —— 位置确定，朝向仍未知
                g_mem.belief.infer_at({st.red.last_known_pos.x + dx,
                                       st.red.last_known_pos.y + dy});
                g_mem.belief.anchor_facing = '?';
                apply_belief(st, board.turn);
            }
            if (failures >= kMaxFailedAttempts) break;
            continue;
        }

        if (same_my_state(predicted.red, e.obs)) {
            st = predicted;
        } else {
            // 模型与真机不符（例如敌位判断有误）→ 用真实观测重建，下一轮重新规划
            st.red.last_known_pos = e.obs.my_pos;
            st.red.last_known_facing = e.obs.my_facing;
            st.red.fire_cd = e.obs.fire_cd;
            st.red.scan_cd = e.obs.scan_cd;
        }

        // 命中 → 对手回出生点。这是瞬移，不属于"3 格内的移动"，必须直接重置信念。
        if (planned.hit) {
            g_mem.belief.on_our_hit(st.size);
        }
        if (e.obs.opp_visible) {
            g_mem.belief.collapse(e.obs.opp_last_known_pos,
                                  e.obs.opp_last_known_facing);
        }
        if (g_use_policy) {
            // sim 推演出来的 st.blue.last_known_pos 是**真实位置**，
            // 而策略网络要的是引擎观测里的 Intel。必须覆盖回去。
            st.blue.last_known_pos = e.obs.opp_last_known_pos;
            st.blue.last_known_facing = e.obs.opp_last_known_facing;
        }
        // 我方这一步后视野变了 → 用新视野证伪
        g_mem.belief.after_action(st);
        apply_belief(st, board.turn);
        st.blue.visible = e.obs.opp_visible;
        enemy_visible = e.obs.opp_visible;

        if (e.consumed) ++used;
        // 免费转向资格的失效条件（对应引擎 Match::do_action）：
        //   · 用掉一次 TURN  → 资格被消费
        //   · 离开出生点     → 资格立即失效
        //   在出生点做 FIRE/SCAN/MOVE 中的前两者不会使其失效
        if (action == sim::kTurn) {
            free_turn = false;
        } else {
            free_turn = free_turn && (st.red.last_known_pos.x == my_spawn.x &&
                                      st.red.last_known_pos.y == my_spawn.y);
        }
        failures = 0;
    }

    // 保存跨回合记忆（信念与锚点在上面每一步都已就地维护）
    g_mem.last_turn = board.turn;
    g_mem.last_my_pos = st.red.last_known_pos;
}

// 绝对安全的兜底：只做不会失败的事，且最多一步
void fallback(const Board& board, char my_color) {
    const Sentry& me = (my_color == 'R') ? board.red : board.blue;
    const Sentry& opp = (my_color == 'R') ? board.blue : board.red;

    if (opp.visible && me.fire_cd == 0) {
        fire();
        return;
    }
    if (me.scan_cd == 0) {
        scan();
        return;
    }
    const Pos* target = nullptr;
    int best = 1 << 20;
    for (const Pos& z : board.score_zones) {
        const int d = std::abs(z.x - me.last_known_pos.x) +
                      std::abs(z.y - me.last_known_pos.y);
        if (d < best) {
            best = d;
            target = &z;
        }
    }
    if (target == nullptr) return;
    // 同样：turn() 要的是局部朝向，引擎自己处理蓝方镜像
    const char want = sim::best_turn_to_face(me.last_known_pos, *target);
    if (me.last_known_facing != want) {
        turn(want);
        return;
    }
    move();
}

} // namespace

extern "C" void act(const Board& board, char my_color) {
    try {
        run(board, my_color);
    } catch (...) {
        // 崩溃 = 整局判负，所以这里必须吞掉一切异常
        try {
            fallback(board, my_color);
        } catch (...) {
        }
    }
}

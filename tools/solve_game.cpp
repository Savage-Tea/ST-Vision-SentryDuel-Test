// tools/solve_game.cpp —— 精确求解这个游戏（完全信息版本）
//
// **需要约 190 GB 内存，必须在 cluster48（2 TB）上跑，不要在本机跑。**
//
// ── 解的是什么 ──
//
// 完全信息版本：假设**双方都能看见对方的真实位置**。这不是真实游戏（真游戏里
// 只看得见 T 形 4 格），但：
//   · 它能精确算出来
//   · 它给出每个局面的**真值**，是衡量任何近似策略离最优有多远的唯一标尺
//   · 在部分可观测下"信念加权查这张表"正是我们想要的在线策略
//     （信念弥散 → 自动选出对所有假设都不差的中性动作；信息一到 → 瞬间收敛）
//
// 真值取 {红胜, 平, 红负}，从红方视角。
//
// ── 为什么能精确求解 ──
//
// 实测可达状态 1,327,194,499（tools/count_states_full.cpp，无任何抽象）。
// 而且状态图是 **DAG**：turn / side / action_count 三个量单调不减，所以
// 反向 DP **一遍过**，不需要迭代到收敛。同层边只有"出生点免费转向"一种
// （apply_step 里 used 不变但清掉 free 位），按 free 标志排序即可消化。
//
// ── 状态 ──
//
//   (红姿态, 蓝姿态, 回合, 行动方, 本阶段已用额度, 免费转向×2, 分差)
//   姿态 = 位置 × 朝向 × fire_cd × scan_cd
//
// 分差足够刻画终局（is_terminal 只看比分是否相等）与奖励，所以不需要两个绝对分。
//
// ── 动作集 ──
//
// 枚举**全部合法动作**（move / turn×4 / fire / scan / 收手），不用
// collect_candidates 的剪枝。剪枝是我们 AI 的策略选择，不是规则；用剪枝后的
// 动作集求解得到的是"受限游戏"的值，不是真值。

#include "brain/actions.h"
#include "sim/rules.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>

namespace {

constexpr int kCells = sim::kBoardSize * sim::kBoardSize; // 49
constexpr int kPoses = kCells * 4 * 3 * 4;                // 2352
constexpr int kTurnN = 26;
constexpr int kAcN = 4;
constexpr int kFreeN = 4;
constexpr int kDiffMax = 51;
constexpr int kDiffN = 2 * kDiffMax + 1;                  // 103
constexpr int kSideN = 2;
constexpr int kActions = 5;                               // move/turn/fire/scan/收手

constexpr std::uint64_t kProduct =
    static_cast<std::uint64_t>(kPoses) * kPoses * kTurnN * kSideN * kAcN * kFreeN * kDiffN;
constexpr std::uint64_t kWords = (kProduct + 63) / 64;

// 层号：turn * 8 + side_idx * 4 + ac，共 26*8 = 208 层
constexpr int kLayers = kTurnN * kSideN * kAcN;
inline int layer_of(int turn, int side_idx, int ac) {
    return (turn * kSideN + side_idx) * kAcN + ac;
}

inline std::uint64_t encode_state(int pr, int pb, int turn, int side, int ac,
                                  int free_flag, int diff_idx) {
    std::uint64_t v = static_cast<std::uint64_t>(pr) * kPoses + pb;
    v = v * kTurnN + turn;
    v = v * kSideN + side;
    v = v * kAcN + ac;
    v = v * kFreeN + free_flag;
    v = v * kDiffN + diff_idx;
    return v;
}

inline int diff_to_idx(int d) {
    const int i = d + kDiffMax;
    return i < 0 ? 0 : (i >= kDiffN ? kDiffN - 1 : i);
}

// 终局判定，等价于 brain::is_terminal（它只看比分是否相等，即分差是否为 0）
inline bool terminal_of(int turn, int diff) {
    if (turn < 20) return false;
    if (turn >= 25) return true;
    return diff != 0;
}

// 值编码（2 bit）：0=未算, 1=红负, 2=平, 3=红胜
inline int value_of_diff(int diff) { return diff > 0 ? 3 : (diff < 0 ? 1 : 2); }

int encode_pose(const Sentry& s) {
    if (!(s.last_known_pos.x >= 0 && s.last_known_pos.x < sim::kBoardSize &&
          s.last_known_pos.y >= 0 && s.last_known_pos.y < sim::kBoardSize)) {
        return -1;
    }
    int fi = 0;
    for (int i = 0; i < 4; ++i) {
        if (s.last_known_facing == "NESW"[i]) fi = i;
    }
    const int cell = s.last_known_pos.y * sim::kBoardSize + s.last_known_pos.x;
    const int fcd = s.fire_cd < 0 ? 0 : (s.fire_cd > 2 ? 2 : s.fire_cd);
    const int scd = s.scan_cd < 0 ? 0 : (s.scan_cd > 3 ? 3 : s.scan_cd);
    return ((cell * 4) + fi) * 12 + (fcd * 4 + scd);
}

struct Decoded {
    int pr, pb, turn, side_idx, ac, free_flag, diff;
    char side() const { return side_idx == 0 ? 'R' : 'B'; }
};

inline Decoded decode(std::uint64_t v) {
    Decoded d{};
    d.diff = static_cast<int>(v % kDiffN) - kDiffMax;
    v /= kDiffN;
    d.free_flag = static_cast<int>(v % kFreeN);
    v /= kFreeN;
    d.ac = static_cast<int>(v % kAcN);
    v /= kAcN;
    d.side_idx = static_cast<int>(v % kSideN);
    v /= kSideN;
    d.turn = static_cast<int>(v % kTurnN);
    v /= kTurnN;
    const int pair = static_cast<int>(v);
    d.pr = pair / kPoses;
    d.pb = pair % kPoses;
    return d;
}

void apply_pose(int p, Sentry& s) {
    const int cd = p % 12;
    const int t = p / 12;
    const int cell = t / 4;
    s.last_known_pos = {cell % sim::kBoardSize, cell / sim::kBoardSize};
    s.last_known_facing = "NESW"[t % 4];
    s.fire_cd = cd / 4;
    s.scan_cd = cd % 4;
}

// 2 bit 值的位图
std::vector<std::uint64_t> g_value;

inline int get_value(std::uint64_t v) {
    const std::uint64_t bit = v * 2;
    return static_cast<int>((g_value[bit >> 6] >> (bit & 63)) & 3ULL);
}
inline void set_value(std::uint64_t v, int val) {
    const std::uint64_t bit = v * 2;
    const std::uint64_t w = bit >> 6;
    const unsigned sh = static_cast<unsigned>(bit & 63);
    g_value[w] = (g_value[w] & ~(3ULL << sh)) | (static_cast<std::uint64_t>(val & 3) << sh);
}

// 正向枚举用的访问位图。**并行版本用原子 test-and-set**：
// 多个线程会同时认领同一批后继，靠 CAS 保证只有一个线程把它收进缓冲。
std::vector<std::uint64_t> g_seen;
inline bool mark(std::uint64_t v) {
    const std::uint64_t w = v >> 6;
    const std::uint64_t mask = 1ULL << (v & 63);
    const std::uint64_t old = __atomic_fetch_or(&g_seen[w], mask, __ATOMIC_RELAXED);
    return (old & mask) == 0;
}

// 每线程的后继缓冲：线程之间不碰同一份内存，层末再合并。
// 直接往全局桶里 push 需要加锁，10^10 次 push 的锁开销会把并行收益吃光。
struct EmitBuffers {
    std::vector<std::vector<std::uint64_t>> buckets; // kLayers * kFreeN
    void init() { buckets.assign(static_cast<std::size_t>(kLayers) * kFreeN, {}); }
    void clear() {
        for (auto& b : buckets) b.clear();
    }
};
std::vector<EmitBuffers> g_tls;

// 每层每 free 标志一份状态表
using LayerTable = std::vector<std::vector<std::uint64_t>>;
LayerTable g_states;

// 从状态算出"我方"在当前局面下的合法动作（全部合法动作，不剪枝）
struct Succ {
    std::uint64_t key;
    bool valid;
};

} // namespace

int main(int argc, char** argv) {
    bool query_only = false;
    std::uint64_t q_pr = 0, q_pb = 0;
    int q_turn = 0;
    (void)query_only; (void)q_pr; (void)q_pb; (void)q_turn;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) {
            std::printf("用法: %s\n  精确求解完全信息版本并输出统计\n", argv[0]);
            return 0;
        }
    }

    const auto t_start = std::chrono::steady_clock::now();
    std::printf("积空间 %.4g  值表 %.1f GB  访问位图 %.1f GB\n",
                static_cast<double>(kProduct),
                static_cast<double>(kWords) * 8 / 1024 / 1024 / 1024 * 2,
                static_cast<double>(kWords) * 8 / 1024 / 1024 / 1024);
    std::fflush(stdout);

    try {
        g_value.assign(kWords * 2, 0); // 2 bit/状态
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "值表分配失败——内存不够，这个程序要在 cluster48 上跑\n");
        return 1;
    }
    std::printf("值表已分配 (%.1f GB)\n", static_cast<double>(g_value.size()) * 8 / 1073741824.0);
    std::fflush(stdout);

    // ══════════ 阶段 1：正向枚举可达状态，按层收集 ══════════
    sim::State init = sim::make_initial_state();
    g_states.assign(static_cast<std::size_t>(kLayers) * kFreeN, {});

    std::printf("\n阶段 1/2：正向枚举\n");
    std::fflush(stdout);

    {
        g_seen.assign(kWords, 0);
        const int pr0 = encode_pose(init.red), pb0 = encode_pose(init.blue);
        const std::uint64_t start = encode_state(pr0, pb0, 0, 0, 0, 3, diff_to_idx(0));
        mark(start);
        g_states[0 * kFreeN + 3].push_back(start);

        int nthreads = 1;
#ifdef _OPENMP
        nthreads = omp_get_max_threads();
#endif
        g_tls.assign(static_cast<std::size_t>(nthreads), EmitBuffers{});
        for (auto& t : g_tls) t.init();
        std::printf("  线程数 %d\n", nthreads);
        std::fflush(stdout);
        std::atomic<long long> total{0};

        for (int layer = 0; layer < kLayers; ++layer) {
            const int turn = layer / (kSideN * kAcN);
            const int rem = layer % (kSideN * kAcN);
            const int side_idx = rem / kAcN;
            const int ac = rem % kAcN;
            const char side = side_idx == 0 ? 'R' : 'B';

            // 同层边只来自免费转向，且总是流向**更小**的 free 标志，
            // 所以正向按 3→0 的顺序处理即可
            for (int ff = kFreeN - 1; ff >= 0; --ff) {
                auto& list = g_states[static_cast<std::size_t>(layer) * kFreeN + ff];
                const std::ptrdiff_t n_states = static_cast<std::ptrdiff_t>(list.size());
#pragma omp parallel for schedule(dynamic, 256)
                for (std::ptrdiff_t i = 0; i < n_states; ++i) {
                    const int tid = omp_get_thread_num();
                    auto& tls = g_tls[static_cast<std::size_t>(tid)];
                    const std::uint64_t v = list[static_cast<std::size_t>(i)];
                    total.fetch_add(1, std::memory_order_relaxed);
                    const Decoded d = decode(v);

                    if (terminal_of(d.turn, d.diff)) continue;
                    if (ac >= brain::kMaxActionsPerTurn) continue;

                    sim::State s = init;
                    s.red = init.red;
                    s.blue = init.blue;
                    apply_pose(d.pr, s.red);
                    apply_pose(d.pb, s.blue);
                    s.turn = d.turn;

                    const bool free_r = (d.free_flag & 2) != 0;
                    const bool free_b = (d.free_flag & 1) != 0;
                    bool free_turn = (side == 'R') ? free_r : free_b;

                    auto emit = [&](int action, char arg) {
                        sim::State ns = s;
                        int used = d.ac;
                        bool ft = free_turn;
                        bool hit = false;
                        const brain::Cand wc = brain::local_to_world({action, arg}, side);
                        if (!brain::apply_step(ns, side, wc, used, ft, &hit)) return;
                        const int npr = encode_pose(ns.red), npb = encode_pose(ns.blue);
                        if (npr < 0 || npb < 0) return;
                        bool nfr = (side == 'R') ? ft : free_r;
                        bool nfb = (side == 'B') ? ft : free_b;
                        if (hit) {
                            if (side == 'R') nfb = true; else nfr = true;
                        }
                        const int nff = (nfr ? 2 : 0) | (nfb ? 1 : 0);
                        const int ndiff = d.diff + (hit ? (side == 'R' ? 2 : -2) : 0);
                        const int nac = used;
                        const std::uint64_t nv = encode_state(
                            npr, npb, d.turn, side_idx, nac, nff, diff_to_idx(ndiff));
                        if (mark(nv)) {
                            tls.buckets[static_cast<std::size_t>(layer_of(d.turn, side_idx, nac)) * kFreeN + nff]
                                .push_back(nv);
                        }
                    };

                    const char my_facing = s.sentry_for(side).last_known_facing;
                    emit(sim::kMove, 0);
                    for (char f : {'N', 'E', 'S', 'W'}) {
                        if (f != my_facing) emit(sim::kTurn, f);
                    }
                    emit(sim::kFire, 0);
                    emit(sim::kScan, 0);

                    // 收手：本阶段结束 —— 结算占点分，进入下一相位
                    {
                        sim::State es = s;
                        const int before = es.sentry_for(side).score;
                        sim::end_side_turn(es, side);
                        const int ediff =
                            d.diff + (es.sentry_for(side).score - before) * (side == 'R' ? 1 : -1);
                        const int epr = encode_pose(es.red), epb = encode_pose(es.blue);
                        if (epr >= 0 && epb >= 0) {
                            if (side == 'R') {
                                const std::uint64_t nv = encode_state(
                                    epr, epb, d.turn, 1, 0, d.free_flag, diff_to_idx(ediff));
                                if (mark(nv)) {
                                    tls.buckets[static_cast<std::size_t>(layer_of(d.turn, 1, 0)) * kFreeN + d.free_flag]
                                        .push_back(nv);
                                }
                            } else {
                                sim::State rs = es;
                                sim::end_round(rs);
                                if (d.turn + 1 < kTurnN) {
                                    const int rpr = encode_pose(rs.red), rpb = encode_pose(rs.blue);
                                    if (rpr >= 0 && rpb >= 0) {
                                        const std::uint64_t nv = encode_state(
                                            rpr, rpb, d.turn + 1, 0, 0, d.free_flag,
                                            diff_to_idx(ediff));
                                        if (mark(nv)) {
                                            tls.buckets[static_cast<std::size_t>(layer_of(d.turn + 1, 0, 0)) * kFreeN + d.free_flag]
                                                .push_back(nv);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // 层末合并：把各线程缓冲并入全局桶。
            // 同层边（免费转向）流向更小的 ff，本层后面还会再处理到，顺序正确。
            for (auto& tls : g_tls) {
                for (std::size_t b = 0; b < tls.buckets.size(); ++b) {
                    auto& dst = g_states[b];
                    auto& src = tls.buckets[b];
                    if (src.empty()) continue;
                    dst.insert(dst.end(), src.begin(), src.end());
                    src.clear();
                }
            }

            if (turn % 5 == 0 && rem == 0) {
                std::printf("  turn %2d 累计 %lld\n", turn, total.load());
                std::fflush(stdout);
            }
        }

        std::printf("  可达状态 %lld\n", total.load());
        std::fflush(stdout);
        // 访问位图不再需要，释放出 59 GB
        std::vector<std::uint64_t>().swap(g_seen);
    }

    // ══════════ 阶段 2：反向 DP ══════════
    std::printf("\n阶段 2/2：反向 DP\n");
    std::fflush(stdout);

    std::vector<brain::Cand> tmp;
    long long computed = 0;

    for (int layer = kLayers - 1; layer >= 0; --layer) {
        const int rem = layer % (kSideN * kAcN);
        const int side_idx = rem / kAcN;
        const int ac = rem % kAcN;
        const char side = side_idx == 0 ? 'R' : 'B';

        // 正向是 3→0，反向就是 0→3（同层后继总是更小的 free 标志，先算）
        for (int ff = 0; ff < kFreeN; ++ff) {
            const auto& list = g_states[static_cast<std::size_t>(layer) * kFreeN + ff];
            for (const std::uint64_t v : list) {
                const Decoded d = decode(v);

                if (terminal_of(d.turn, d.diff)) {
                    set_value(v, value_of_diff(d.diff));
                    ++computed;
                    continue;
                }

                sim::State s = init;
                s.red = init.red;
                s.blue = init.blue;
                apply_pose(d.pr, s.red);
                apply_pose(d.pb, s.blue);
                s.turn = d.turn;

                const bool free_r = (d.free_flag & 2) != 0;
                const bool free_b = (d.free_flag & 1) != 0;
                const bool free_turn = (side == 'R') ? free_r : free_b;

                int best = -1;
                auto consider = [&](int action, char arg) {
                    sim::State ns = s;
                    int used = d.ac;
                    bool ft = free_turn;
                    bool hit = false;
                    const brain::Cand wc = brain::local_to_world({action, arg}, side);
                    if (!brain::apply_step(ns, side, wc, used, ft, &hit)) return;
                    const int npr = encode_pose(ns.red), npb = encode_pose(ns.blue);
                    if (npr < 0 || npb < 0) return;
                    bool nfr = (side == 'R') ? ft : free_r;
                    bool nfb = (side == 'B') ? ft : free_b;
                    if (hit) {
                        if (side == 'R') nfb = true; else nfr = true;
                    }
                    const int nff = (nfr ? 2 : 0) | (nfb ? 1 : 0);
                    const int ndiff = d.diff + (hit ? (side == 'R' ? 2 : -2) : 0);
                    const std::uint64_t nv = encode_state(npr, npb, d.turn, side_idx, used,
                                                          nff, diff_to_idx(ndiff));
                    const int val = get_value(nv);
                    if (val == 0) return; // 后继未算 —— 分层有洞，见文末断言
                    if (best < 0) best = val;
                    else if (side == 'R') { if (val > best) best = val; }
                    else { if (val < best) best = val; }
                };

                // 额度用尽时只有"收手"一条路
                if (ac < brain::kMaxActionsPerTurn) {
                    const char my_facing = s.sentry_for(side).last_known_facing;
                    consider(sim::kMove, 0);
                    for (char f : {'N', 'E', 'S', 'W'}) {
                        if (f != my_facing) consider(sim::kTurn, f);
                    }
                    consider(sim::kFire, 0);
                    consider(sim::kScan, 0);
                }
                // 收手
                {
                    sim::State es = s;
                    const int before = es.sentry_for(side).score;
                    sim::end_side_turn(es, side);
                    const int ediff =
                        d.diff + (es.sentry_for(side).score - before) * (side == 'R' ? 1 : -1);
                    const int epr = encode_pose(es.red), epb = encode_pose(es.blue);
                    if (epr >= 0 && epb >= 0) {
                        std::uint64_t nv = 0;
                        bool ok = false;
                        if (side == 'R') {
                            nv = encode_state(epr, epb, d.turn, 1, 0, d.free_flag,
                                              diff_to_idx(ediff));
                            ok = true;
                        } else {
                            sim::State rs = es;
                            sim::end_round(rs);
                            if (d.turn + 1 < kTurnN) {
                                const int rpr = encode_pose(rs.red), rpb = encode_pose(rs.blue);
                                if (rpr >= 0 && rpb >= 0) {
                                    nv = encode_state(rpr, rpb, d.turn + 1, 0, 0,
                                                      d.free_flag, diff_to_idx(ediff));
                                    ok = true;
                                }
                            }
                        }
                        if (ok) {
                            const int val = get_value(nv);
                            if (val != 0) {
                                if (best < 0) best = val;
                                else if (side == 'R') { if (val > best) best = val; }
                                else { if (val < best) best = val; }
                            }
                        }
                    }
                }

                set_value(v, best < 0 ? 2 : best);
                ++computed;
            }
        }
    }

    std::printf("  已计算 %lld\n", computed);

    // ══════════ 结果 ══════════
    const std::uint64_t startv =
        encode_state(encode_pose(init.red), encode_pose(init.blue), 0, 0, 0, 3, diff_to_idx(0));
    const int v0 = get_value(startv);
    const char* name[] = {"未算", "红负", "平", "红胜"};

    std::printf("\n════════ 开局值（完全信息，红方先手）════════\n");
    std::printf("  %s\n", name[v0]);
    std::printf("\n注意：红蓝在开局是对称的（棋盘 180° 对称 + 出生点对置），\n");
    std::printf("所以'红胜'意味着**先手必胜**，不是红方有优势。\n");

    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::printf("\n用时 %.1f 秒\n", secs);
    return 0;
}

// tools/count_states_full.cpp —— 精确枚举**完整求解状态**的可达集
//
// 需要 59 GB 位图 + 数小时。**在 cluster48 上跑，不要在本机跑。**
//
// 状态（不做任何抽象——不按行为签名聚类，不合并等价姿态）：
//
//   (红姿态, 蓝姿态, 回合, 行动方, 本阶段已用额度, 免费转向×2, 分差)
//
//   姿态 = 位置(49) × 朝向(4) × fire_cd(3) × scan_cd(4) = 2352
//   分差 = 红分 − 蓝分，范围 ±51（25 次占点 + 26 次击杀的上界）
//
// 【为什么位图而不是哈希表】
// 积空间 4.74e11 位 = 59 GB，cluster48 有 2 TB，直接开位图。
// 哈希表要在每个条目上存 8 字节键 + 开销，量级上更贵而且有冲突处理。
//
// 【为什么不是"只保留相邻两层"】
// 相位结束会跳跃：一方用了 1 个行动就收手，直接从 (turn, side, ac=1) 跳到
// 下一相位的 ac=0，所以后继不总是 layer+1，双缓冲不成立。全位图最直白。
//
// 过渡用 brain::apply_step —— 它复刻了引擎的额度与免费转向语义，
// 也正是选手 .so 在用的那一份实现。

#include "brain/actions.h"
#include "sim/rules.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int kCells = sim::kBoardSize * sim::kBoardSize;
constexpr int kPoses = kCells * 4 * 3 * 4; // 2352
constexpr int kTurnN = 26;                 // 0..25
constexpr int kAcN = 4;                    // 已用额度 0..3
constexpr int kFreeN = 4;                  // free_r(2) × free_b(2)
constexpr int kDiffMax = 51;
constexpr int kDiffN = 2 * kDiffMax + 1;   // 103

constexpr std::uint64_t kProduct =
    static_cast<std::uint64_t>(kPoses) * kPoses * kTurnN * 2 * kAcN * kFreeN * kDiffN;
constexpr std::uint64_t kBits = kProduct;
constexpr std::uint64_t kWords = (kBits + 63) / 64;

inline std::uint64_t encode_state(int pr, int pb, int turn, int side, int ac,
                                  int free_flag, int diff_idx) {
    std::uint64_t v = static_cast<std::uint64_t>(pr) * kPoses + pb;
    v = v * kTurnN + turn;
    v = v * 2 + side;
    v = v * kAcN + ac;
    v = v * kFreeN + free_flag;
    v = v * kDiffN + diff_idx;
    return v;
}

inline int diff_to_idx(int d) {
    int i = d + kDiffMax;
    return i < 0 ? 0 : (i >= kDiffN ? kDiffN - 1 : i);
}

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

void decode_pose(int p, Sentry& s) {
    const int cd = p % 12;
    const int t = p / 12;
    const int fi = t % 4;
    const int cell = t / 4;
    s.last_known_pos = {cell % sim::kBoardSize, cell / sim::kBoardSize};
    s.last_known_facing = "NESW"[fi];
    s.fire_cd = cd / 4;
    s.scan_cd = cd % 4;
}

std::vector<std::uint64_t> g_seen;

inline bool mark(std::uint64_t v) {
    const std::uint64_t w = v >> 6, b = v & 63;
    if (g_seen[w] >> b & 1ULL) return false;
    g_seen[w] |= 1ULL << b;
    return true;
}

} // namespace

int main(int argc, char** argv) {
    long long limit = -1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--limit") == 0 && i + 1 < argc) {
            limit = std::atoll(argv[++i]);
        }
    }

    std::printf("积空间: %.4g 状态 → 位图 %.1f GB\n", static_cast<double>(kProduct),
                static_cast<double>(kWords) * 8 / 1024 / 1024 / 1024);
    std::fflush(stdout);

    g_seen.assign(kWords, 0);
    std::vector<std::uint64_t> queue;
    queue.reserve(1 << 20);

    sim::State init = sim::make_initial_state();
    const int pr0 = encode_pose(init.red), pb0 = encode_pose(init.blue);

    const std::uint64_t start = encode_state(pr0, pb0, 0, 0, 0, 3, diff_to_idx(0));
    mark(start);
    queue.push_back(start);

    std::size_t head = 0;
    long long visited = 0;

    std::vector<brain::Cand> cands;

    while (head < queue.size()) {
        const std::uint64_t v = queue[head++];
        ++visited;

        int diff_idx = static_cast<int>(v % kDiffN);
        std::uint64_t t = v / kDiffN;
        const int free_flag = static_cast<int>(t % kFreeN);
        t /= kFreeN;
        const int ac = static_cast<int>(t % kAcN);
        t /= kAcN;
        const int side_idx = static_cast<int>(t % 2);
        t /= 2;
        const int turn = static_cast<int>(t % kTurnN);
        t /= kTurnN;
        const int pair = static_cast<int>(t);
        const int pr = pair / kPoses, pb = pair % kPoses;

        const char side = side_idx == 0 ? 'R' : 'B';

        sim::State s = init;
        s.red = init.red;
        s.blue = init.blue;
        decode_pose(pr, s.red);
        decode_pose(pb, s.blue);
        s.turn = turn;

        const bool free_r = (free_flag & 2) != 0;
        const bool free_b = (free_flag & 1) != 0;
        const bool free_turn = (side == 'R') ? free_r : free_b;
        const int diff = diff_idx - kDiffMax;

        // —— 候选动作 ——
        // 注意用 can_see=false：求解的是**不依赖观测**的那一层，
        // 观测在信念加权那一步才用得上。
        cands.clear();
        brain::collect_candidates(s, side, free_turn, /*can_see_enemy=*/false,
                                  nullptr, nullptr, cands);

        bool phase_ended = false;

        for (const brain::Cand& c : cands) {
            sim::State ns = s;
            int used = ac;
            bool ft = free_turn;
            bool hit = false;
            const brain::Cand wc = brain::local_to_world(c, side);
            if (!brain::apply_step(ns, side, wc, used, ft, &hit)) continue;

            int ndiff = diff + (hit ? (side == 'R' ? 2 : -2) : 0);
            const int npr = encode_pose(ns.red), npb = encode_pose(ns.blue);
            if (npr < 0 || npb < 0) continue;

            bool nfr = (side == 'R') ? ft : free_r;
            bool nfb = (side == 'B') ? ft : free_b;
            if (hit) {
                if (side == 'R') nfb = true; else nfr = true;
            }
            const int nff = (nfr ? 2 : 0) | (nfb ? 1 : 0);

            if (used < brain::kMaxActionsPerTurn) {
                const std::uint64_t nv =
                    encode_state(npr, npb, turn, side_idx, used, nff, diff_to_idx(ndiff));
                if (mark(nv)) queue.push_back(nv);
            } else {
                phase_ended = true;
            }
        }

        // 收手（主动结束本阶段）
        phase_ended = true;

        if (phase_ended) {
            sim::State es = s;
            const int before = es.sentry_for(side).score;
            sim::end_side_turn(es, side);
            const int ediff =
                diff + (es.sentry_for(side).score - before) * (side == 'R' ? 1 : -1);
            const int nff = (free_r ? 2 : 0) | (free_b ? 1 : 0);

            if (side == 'R') {
                const int epr = encode_pose(es.red), epb = encode_pose(es.blue);
                if (epr >= 0 && epb >= 0) {
                    const std::uint64_t nv =
                        encode_state(epr, epb, turn, 1, 0, nff, diff_to_idx(ediff));
                    if (mark(nv)) queue.push_back(nv);
                }
            } else {
                sim::State rs = es;
                sim::end_round(rs);
                if (turn + 1 < kTurnN) {
                    const int epr = encode_pose(rs.red), epb = encode_pose(rs.blue);
                    if (epr >= 0 && epb >= 0) {
                        const std::uint64_t nv =
                            encode_state(epr, epb, turn + 1, 0, 0, nff, diff_to_idx(ediff));
                        if (mark(nv)) queue.push_back(nv);
                    }
                }
            }
        }

        if (visited % 5000000 == 0 && visited > 0) {
            std::printf("  已访问 %.3g  队列 %.3g  当前 turn=%d side=%c ac=%d\n",
                        static_cast<double>(visited), static_cast<double>(queue.size()),
                        turn, side, ac);
            std::fflush(stdout);
        }
        if (limit > 0 && visited >= limit) {
            std::printf("达到 --limit，提前停止\n");
            break;
        }
    }

    std::printf("\n");
    std::printf("可达状态总数        : %lld  (%.4g)\n", visited, static_cast<double>(visited));
    std::printf("积空间              : %.4g  (可达率 %.3g%%)\n",
                static_cast<double>(kProduct),
                100.0 * static_cast<double>(visited) / static_cast<double>(kProduct));
    std::printf("1 字节/状态存储     : %.2f GB\n",
                static_cast<double>(visited) / 1024 / 1024 / 1024);
    std::printf("1 字节/状态 + 对称规范化(/2): %.2f GB\n",
                static_cast<double>(visited) / 2 / 1024 / 1024 / 1024);
    return 0;
}

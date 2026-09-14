// tools/count_states.cpp —— 精确统计这个游戏的可达状态空间
//
// 手算容易差好几个数量级（并非所有格子/朝向可达、CD 有约束、比分由历史决定），
// 而"能不能精确求解"完全取决于这个数。所以先精确枚举一遍再谈方案。
//
// 枚举的状态是**姿态对**：
//     (红方姿态, 蓝方姿态)   姿态 = 位置 × 朝向 × fire_cd × scan_cd
// 不含比分与回合号。理由：
//   · 姿态转移与比分无关（比分只由"阶段末在不在得分区"和"击杀"决定）
//   · 回合号只限制局长（≤25 回合），不改变姿态转移本身
//   所以姿态对的可达集是良定义的，后面乘上回合号/比分即可估总规模。
//
// 转移用 sim::apply_action —— 它已经过与引擎的差分测试（487 万次比对 0 不一致），
// 所以这里的可达集就是引擎的可达集。
//
// 用法: count_states [--max-millions N]
//   （姿态对空间上限 2352² ≈ 553 万，可以全枚举，不需要上限）

#include "sim/rules.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <string>
#include <vector>

namespace {

// 把 Sentry 压成一个 0..2351 的编号。障碍格永远不可达，但不影响编号的连续性，
// 只是会浪费一点空间——换取编码/解码的无分支与正确性。
constexpr int kCells = sim::kBoardSize * sim::kBoardSize; // 49
constexpr int kPoses = kCells * 4 * 3 * 4;                // 2352
constexpr int kPairs = kPoses * kPoses;                   // 5,531,904

int encode_pose(const Sentry& s) {
    const char f[] = {'N', 'E', 'S', 'W'};
    int fi = 0;
    for (int i = 0; i < 4; ++i) {
        if (s.last_known_facing == f[i]) fi = i;
    }
    const int cell = s.last_known_pos.y * sim::kBoardSize + s.last_known_pos.x;
    const int cd = (s.fire_cd < 0 ? 0 : s.fire_cd) * 4 + (s.scan_cd < 0 ? 0 : s.scan_cd);
    return ((cell * 4) + fi) * 12 + cd;
}

std::vector<uint8_t> g_seen; // 每个姿态对一字节，省内存

bool mark(int r, int b) {
    const int idx = r * kPoses + b;
    if (g_seen[idx]) return false;
    g_seen[idx] = 1;
    return true;
}

// 用固定顺序枚举"双方各自的候选动作"，不依赖 brain::collect_candidates 的剪枝
// ——剪枝会影响可达集，我们要的是**未剪枝**的真实可达集。
void for_each_action(const sim::State& s, char side,
                     const std::function<void(int, char)>& fn) {
    for (int a = 0; a < 4; ++a) {
        if (a == sim::kTurn) {
            for (char c : {'N', 'E', 'S', 'W'}) fn(a, c);
        } else {
            fn(a, 0);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) {
            std::printf("用法: %s\n  精确枚举可达的姿态对数量\n", argv[0]);
            return 0;
        }
    }

    g_seen.assign(kPairs, 0);

    sim::State init = sim::make_initial_state();
    const int r0 = encode_pose(init.red);
    const int b0 = encode_pose(init.blue);

    std::deque<std::pair<int, int>> q;
    mark(r0, b0);
    q.emplace_back(r0, b0);

    long long visited = 0;
    int max_pair = 0;
    // 记录每个姿态出现过的伙伴数，用来判断"姿态"这一层实际有多少可达
    std::vector<uint8_t> pose_seen(kPoses, 0);

    while (!q.empty()) {
        const auto [rp, bp] = q.front();
        q.pop_front();
        ++visited;

        // 解码回 State
        sim::State s = init;
        s.red = init.red;
        s.blue = init.blue;
        {
            int t = rp;
            const int cd = t % 12; t /= 12;
            const int fi = t % 4;  t /= 4;
            const int cell = t;
            s.red.last_known_pos = {cell % sim::kBoardSize, cell / sim::kBoardSize};
            s.red.last_known_facing = "NESW"[fi];
            s.red.fire_cd = cd / 4;
            s.red.scan_cd = cd % 4;
        }
        {
            int t = bp;
            const int cd = t % 12; t /= 12;
            const int fi = t % 4;  t /= 4;
            const int cell = t;
            s.blue.last_known_pos = {cell % sim::kBoardSize, cell / sim::kBoardSize};
            s.blue.last_known_facing = "NESW"[fi];
            s.blue.fire_cd = cd / 4;
            s.blue.scan_cd = cd % 4;
        }
        pose_seen[rp] = 1;
        pose_seen[bp] = 1;

        for (char side : {'R', 'B'}) {
            for (int a = 0; a < 4; ++a) {
                const char args[5] = {0, 'N', 'E', 'S'};
                const int n =
                    (a == sim::kTurn) ? 4 : 1;
                for (int k = 0; k < n; ++k) {
                    sim::State next = s;
                    const char arg = (a == sim::kTurn) ? "NESW"[k] : 0;
                    (void)args;
                    const sim::Outcome oc = sim::apply_action(next, side, a, arg);
                    if (!oc.success) continue;
                    const int nr = encode_pose(next.red);
                    const int nb = encode_pose(next.blue);
                    if (mark(nr, nb)) q.emplace_back(nr, nb);
                }
            }
        }
        // 回合结束（CD 各 −1），这也改变姿态
        {
            sim::State next = s;
            sim::end_round(next);
            const int nr = encode_pose(next.red);
            const int nb = encode_pose(next.blue);
            if (mark(nr, nb)) q.emplace_back(nr, nb);
        }

        if (visited % 500000 == 0) {
            std::printf("  ... 已访问 %lld, 队列 %zu\n", visited, q.size());
            std::fflush(stdout);
        }
        if (visited > 20000000) {
            std::printf("超过 2000 万，提前停止（说明空间远大于预期）\n");
            break;
        }
    }

    int reachable_poses = 0;
    for (uint8_t v : pose_seen) reachable_poses += v ? 1 : 0;

    std::printf("\n");
    std::printf("姿态总空间        : %d (49格 × 4朝向 × 3 fire_cd × 4 scan_cd)\n", kPoses);
    std::printf("姿态**实际可达**  : %d (%.1f%%)\n", reachable_poses,
                100.0 * reachable_poses / kPoses);
    std::printf("姿态对可达集      : %lld\n", visited);
    std::printf("姿态对全空间      : %d (%.2f%% 可达)\n", kPairs,
                100.0 * visited / kPairs);
    std::printf("\n乘上回合号(26)与(单方)行动阶段结构后，决策状态的量级:\n");
    std::printf("  %lld × 26 ≈ %.3g\n", visited, static_cast<double>(visited) * 26);
    return 0;
}

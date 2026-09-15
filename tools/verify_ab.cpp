// tools/verify_ab.cpp —— αβ 求解器的独立对拍器
//
// 与 tools/solve_ab.cpp 的差异（每一条都是为了**独立性**）：
//   · 裸 minimax：无 αβ 剪枝、无置换表、无 best-move 记录
//   · 无"同朝向转向"剪枝——枚举全部 7 个动作 + 收手。solve_ab 声称该动作
//     严格劣势、剪掉不改变值；这里不剪，若两边值相同，这条声称就被验证
//   · 备忘键 = 整个 sim::State 原始字节 + (side, ac, free_r, free_b)，
//     不用 make_key 的紧凑编码——键打包逻辑的 bug 不会两边共享
//   · 直接用真实比分 + brain::is_terminal 判终局，不用分差重推
//
// 用法: verify_ab --from-turn N
// 输出根值（红胜/平/红负），与 solve_ab --from-turn N 的值必须一致。

#include "brain/actions.h"
#include "sim/rules.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

namespace {

long long g_nodes = 0;
long long g_hits = 0;
std::unordered_map<std::string, std::int8_t> g_memo;

// 终局值（红方视角）
int terminal_value(const sim::State& s) {
    if (s.red.score > s.blue.score) return 1;
    if (s.red.score < s.blue.score) return -1;
    return 0;
}

std::string make_key(const sim::State& s, char side, int ac, bool fr, bool fb) {
    std::string k;
    k.reserve(sizeof(sim::State) + 16);
    // 只要会变的部分：双方姿态+比分、回合、地图不必（固定）
    k.append(reinterpret_cast<const char*>(&s.red), sizeof(s.red));
    k.append(reinterpret_cast<const char*>(&s.blue), sizeof(s.blue));
    k.append(reinterpret_cast<const char*>(&s.turn), sizeof(s.turn));
    k.push_back(side);
    k.push_back(static_cast<char>(ac));
    k.push_back(fr ? '1' : '0');
    k.push_back(fb ? '1' : '0');
    return k;
}

// side 视角的 negamax：返回值已按"红正蓝负"编码，取 max/min 由符号处理
int search(const sim::State& s, char side, int ac, bool fr, bool fb) {
    if (brain::is_terminal(s)) return terminal_value(s);

    const std::string key = make_key(s, side, ac, fr, fb);
    const auto it = g_memo.find(key);
    if (it != g_memo.end()) { ++g_hits; return it->second; }
    ++g_nodes;

    const bool my_free = (side == 'R') ? fr : fb;
    int best = (side == 'R') ? -2 : 2;

    // 全部 7 个动作，**不剪枝**
    for (int action = 0; action < 4; ++action) {
        const int nargs = (action == sim::kTurn) ? 4 : 1;
        for (int ai = 0; ai < nargs; ++ai) {
            const char arg = (action == sim::kTurn) ? "NESW"[ai] : 0;
            sim::State ns = s;
            int used = ac;
            bool ft = my_free;
            if (!brain::apply_step(ns, side, {action, arg}, used, ft, nullptr)) continue;
            const int v = search(ns, side, used, (side == 'R') ? ft : fr,
                                 (side == 'B') ? ft : fb);
            if (side == 'R') { if (v > best) best = v; }
            else { if (v < best) best = v; }
        }
    }

    // 收手：结束本阶段
    {
        sim::State es = s;
        sim::end_side_turn(es, side);
        int v;
        if (side == 'R') {
            v = search(es, 'B', 0, fr, fb);
        } else {
            sim::State rs = es;
            sim::end_round(rs);
            // is_terminal 在下一层入口判定；turn 越界时引擎早已终局
            v = search(rs, 'R', 0, fr, fb);
        }
        if (side == 'R') { if (v > best) best = v; }
        else { if (v < best) best = v; }
    }

    const std::int8_t stored = static_cast<std::int8_t>(best);
    g_memo.emplace(key, stored);
    return best;
}

} // namespace

int main(int argc, char** argv) {
    int from_turn = 24;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--from-turn") == 0 && i + 1 < argc) {
            from_turn = std::atoi(argv[++i]);
        }
    }

    sim::State s = sim::make_initial_state();
    s.turn = from_turn; // 人为 0:0 子博弈，与 solve_ab --from-turn 同口径

    const int v = search(s, 'R', 0, true, true);
    const char* name[] = {"红负", "平", "红胜"};
    std::printf("from-turn %d: %s   (节点 %lld, 备忘 %zu, 命中 %lld)\n",
                from_turn, name[v + 1], g_nodes, g_memo.size(), g_hits);
    return 0;
}

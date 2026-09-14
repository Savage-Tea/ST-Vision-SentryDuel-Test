// sim/rules.h —— 规则核心（世界坐标），逐行对应 engine/src/board.cpp 与 utils.cpp
//
// 全部是纯函数：给定状态与行动返回新状态，不做任何 I/O、不持有全局状态。
// 这是搜索（阶段①）、MCTS（阶段②）、训练环境（阶段③）共用的推演内核。

#pragma once

#include "sim/state.h"

namespace sim {

// —— 初始状态 ——
State make_initial_state(int size = kBoardSize);

// —— 坐标 / 朝向 / 出生点 ——
void facing_delta(char f, int& dx, int& dy);
Pos mirror_pos(Pos p, int size);
char mirror_facing(char f);
Pos spawn_of(char side, int size);
char spawn_facing(char side);
bool is_at_spawn(const State& s, char side);
bool obstacle_at(int x, int y, const std::vector<Pos>& obstacles);

// —— 规则核心 ——
// 应用一次行动。action 见 sim::Action；arg 仅 kTurn 使用（'N'/'E'/'S'/'W'）。
// 语义与引擎 apply_action 完全一致：不含行动额度与免费转向判定（那属于对局层）。
Outcome apply_action(State& s, char side, int action, char arg);

// 一方行动结束：该方位于得分区则 +1（双方各自结算，互不排斥）
void end_side_turn(State& s, char side);

// 双方行动结束：CD -1、回合数 +1
void end_round(State& s);

// 视野：T 形 4 格 + 逐条视线遮挡（对应 utils.cpp can_see）
bool can_see(const Sentry& me, const Pos& target, const std::vector<Pos>& obstacles);

// 火力：前向 3×3，同一火力通道上的障碍挡住其后的格子（对应 board.cpp fire_hit）
Pos fire_hit(const Sentry& me, const Pos& opp_pos, const std::vector<Pos>& obstacles);

// —— 工具（对应 utils.cpp）——
int manhattan_distance(const Pos& a, const Pos& b);
bool in_score_zone(const Pos& pos, const std::vector<Pos>& zones);
char best_turn_to_face(const Pos& from, const Pos& to);
bool can_move_forward(const Sentry& me, const Pos& opp_pos,
                      const std::vector<Pos>& obstacles, int board_size);

} // namespace sim

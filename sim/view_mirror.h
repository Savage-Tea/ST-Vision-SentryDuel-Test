// sim/view_mirror.h —— 复刻引擎 Match 的「敌方情报 + 选手视图」
//
// 【这个模块存在的唯一理由】
//
// sim::State 是世界状态：里面的 last_known_pos 是**真实位置**（见 sim/state.h
// 的说明）。而引擎发给选手的不是真实位置，是 Intel —— 一份**记忆**：
//
//   struct Intel { Pos pos{-1,-1}; char facing='?'; bool known=false; };
//
// 开局时 known == false，选手看到的敌方位置是 {-1,-1}。之后只在三处更新：
//   · view_for(side)            —— 每个行动阶段开头，若处于直接视野内
//   · SCAN 成功                  —— do_action 里无条件 remember
//   · refresh_current_vision()  —— 每次行动后，若处于直接视野内
//
// 自对弈如果图省事直接用世界状态构造观测，就是把**真实位置**喂给网络。
// 那是信息泄漏：网络会学到一个部署时根本拿不到的特征，而且**不会报错**，
// 只表现为"训练出来的策略在真机上莫名其妙地差"。
//
// 另有一个容易搞错的点：**击杀后 Intel 不重置**。引擎命中后只置
// respawn_turn_pending(对方)，那只影响对方下一次的免费转向资格；
// Intel 仍然停在"我们开枪时他在哪"。这里照抄，不自作聪明。
//
// 所有语义逐行对应 engine/src/match.cpp（remember_enemy / direct_vision /
// view_for / do_action 尾部 / act_phase 开头）。

#pragma once

#include "sim/rules.h"

namespace sim {

// Intel 定义在 sim/state.h（对应 engine/include/match.h 的 Match::Intel），
// 这里直接用，不再另立一份——两份定义迟早会漂移。

// 每一方一份。用 side 索引（'R'/'B'）。
class ViewMirror {
public:
    // 新对局：Intel 回到 unknown。**每局开始必须调用**。
    void reset();

    // 对应 Match::act_phase 开头：清 scanned_this_act_，再走一次 view_for。
    // 必须先于本阶段的任何 local_view() 调用。
    void begin_phase(const State& world, char side);

    // 对应 Match::do_action 尾部的一次成功行动。action 用 sim::Action。
    // 失败的行动不会走到这里（失败不消耗额度，也不更新情报）。
    void after_action(const State& world, char side, int action);

    // 该方此刻是否看得见对手（对应视图里的 enemy.visible）
    bool enemy_visible(char side) const { return visible_[idx(side)]; }

    const Intel& intel(char side) const { return intel_[idx(side)]; }

    // 该方此刻应当看到的局部视图：我方恒在 red 槽位，敌方槽位是 Intel 而非真值。
    // 可以直接喂给 obs::encode_v3。
    State local_view(const State& world, char side) const;

private:
    static int idx(char side) { return side == 'R' ? 0 : 1; }
    static Pos local_pos(Pos p, char side, int size);
    static char local_facing(char f, char side);

    bool direct_vision(const State& world, char side) const;
    void remember(const State& world, char side);

    Intel intel_[2];
    bool visible_[2] = {false, false};
    bool scanned_[2] = {false, false}; // 对应 Match::scanned_this_act_
};

} // namespace sim

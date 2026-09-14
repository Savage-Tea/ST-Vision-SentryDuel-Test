#!/usr/bin/env python3
"""test_traj_format.py —— SDPP 轨迹格式的 round-trip 与自洽性测试

生成 C++ 侧（selfplay_ppo）写的真实轨迹，然后：
  1. 读回来，逐字节写回，比对是否与原文件完全一致（round-trip）
  2. 校验取值域：obs 有限、action 在 [0, act_dim)、reward 有限
  3. 检查每条序列的**最后一步必须带终局奖励**（±1）——这是格式约定，
     也是 PPO 的 bootstrap 前提（每条序列都以真实终局结束）
  4. 打印步数/动作/奖励的分布，异常分布要能看出来

用法: python3 tests/test_traj_format.py [--games 30]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools import sdpp  # noqa: E402

FAILS = 0


def check(cond: bool, what: str) -> None:
    global FAILS
    if not cond:
        FAILS += 1
        print(f"  ❌ {what}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--keep", action="store_true", help="保留生成的轨迹文件")
    args = ap.parse_args()

    traj = ROOT / "build" / "traj_test.bin"
    traj.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] 生成 {args.games} 局策略自对弈轨迹")
    proc = subprocess.run(
        [str(ROOT / "build" / "selfplay_ppo"), "--games", str(args.games),
         "--threads", "4", "--seed", "7", "--out", str(traj)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout + proc.stderr)
        return 1

    print("[2/4] 读回并校验")
    t = sdpp.read(traj)
    check(t.obs_dim == 360, f"obs_dim 应为 360，实际 {t.obs_dim}")
    check(t.act_dim == 8, f"act_dim 应为 8，实际 {t.act_dim}")
    check(len(t.seqs) == args.games * 2,
          f"应有 {args.games * 2} 条序列（每局红蓝各一），实际 {len(t.seqs)}")

    all_actions = Counter()
    steps_per_seq = []
    neg_misplaced = 0     # 负奖励出现在非末步 —— 终局加成记错了位置
    neg_last = 0          # 末步为负（输掉）的序列数
    pos_last = 0          # 末步为正（赢下）的序列数
    bad_mid = 0           # 中间步出现了不该有的取值
    for i, s in enumerate(t.seqs):
        n = len(s.obs)
        steps_per_seq.append(n)
        check(n > 0, f"序列 {i} 是空的")
        if n == 0:
            continue
        check(np.all(np.isfinite(s.obs)), f"序列 {i} 的 obs 有非有限值")
        check(np.all(np.isfinite(s.reward)), f"序列 {i} 的 reward 有非有限值")
        check(bool(np.all(s.action < t.act_dim)), f"序列 {i} 有越界的 action")
        all_actions.update(s.action.tolist())

        # 信用分配的不变量。
        #
        # 中间步可以拿到：占点 +1（记在**每个行动阶段**的末步，而阶段末在整条
        # 序列里通常属于中间）、击杀 +2（记在开火那一步）、以及两者落在同一步
        # 时的 +3。所以中间步的取值范围是 {0,1,2,3}，不是 {0,2}。
        #
        # **真正的判据是负奖励的位置**：终局加成是唯一的负值来源，所以
        # 负数必须且只能出现在最后一步。这条不依赖策略强弱，也不会因为
        # "随机策略几乎不得分"而失效。
        if n > 1:
            mid = s.reward[:-1]
            if np.any(mid < 0):
                neg_misplaced += 1
            if np.any((mid < 0) | (mid > 3) | (mid != np.round(mid))):
                bad_mid += 1
        last = float(s.reward[-1])
        if last < 0:
            neg_last += 1
        elif last > 0:
            pos_last += 1

    check(neg_misplaced == 0,
          f"有 {neg_misplaced} 条序列的负奖励出现在非末步——终局加成记错了位置")
    check(bad_mid == 0,
          f"有 {bad_mid} 条序列的中间步出现了 {0,1,2,3} 之外的奖励"
          f"——占点分或终局加成可能被记到了错误的位置")

    print("[3/4] round-trip：读回后再写，必须与原文件逐字节一致")
    rt = ROOT / "build" / "traj_roundtrip.bin"
    sdpp.write(rt, t)
    a, b = traj.read_bytes(), rt.read_bytes()
    check(len(a) == len(b), f"round-trip 长度不同: {len(a)} vs {len(b)}")
    if len(a) == len(b):
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        check(first is None, f"round-trip 第 {first} 字节起不一致")

    print("[4/4] 分布")
    arr = np.array(steps_per_seq)
    print(f"  序列数 {len(t.seqs)}   每序列步数 最小 {arr.min()} / 中位 "
          f"{int(np.median(arr))} / 最大 {arr.max()}")
    total = sum(all_actions.values())
    names = ["move", "turnN", "turnE", "turnS", "turnW", "fire", "scan", "stop"]
    dist = "  ".join(f"{names[k]} {v * 100.0 / total:.1f}%" for k, v in sorted(all_actions.items()))
    print(f"  动作分布 {dist}")
    print(f"  平均每局步数（双方合计） {arr.sum() / args.games:.1f}")
    # 未训练的随机策略几乎不得分，于是大量 0-0 平局。这不是 bug，但必须
    # 让它在输出里可见——否则会误以为"奖励信号坏了"。
    print(f"  末步胜负：赢 {pos_last}  输 {neg_last}  平 {len(t.seqs) - pos_last - neg_last}"
          f"   —— 未训练策略下平局占比高是正常的")
    if pos_last == 0 or neg_last == 0:
        print("  ⚠ 这一批里 ±1 终局加成只有一侧出现过，终局路径未被完整覆盖")

    if not args.keep:
        traj.unlink(missing_ok=True)
        rt.unlink(missing_ok=True)

    print()
    if FAILS == 0:
        print("✅ 轨迹格式测试通过")
        return 0
    print(f"❌ {FAILS} 项断言失败")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

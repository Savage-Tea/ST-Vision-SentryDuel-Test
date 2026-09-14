#!/usr/bin/env python3
"""test_invariants.py —— 阶段① AI 的两条不变量回归测试

背景：第一版 AI 在 turn() 的参数上镜像了两次（引擎 turn() 收的是**局部**朝向，
它在 Match::do_action 内部自己做蓝方镜像）。结果是执红一切正常、执蓝整局报废
——20 回合 0 分。红蓝共用同一份策略代码，所以这类 bug 只会打崩一侧，
总分看起来"有一半是对的"，极易漏掉。

这两条不变量就是为它设的：

  ① 确定性   —— 对确定性对手重复跑，结果必须逐字节相同。
                否则说明有未初始化内存/时间依赖，所有评测数字都不可信。
  ② 颜色对称 —— 同一份策略执红与执蓝的表现不能出现巨大落差。
                镜像搞反、spawn/坐标框架弄错，都会在这里现形。

用引擎自带的 det_ai_a / det_ai_b 当对手：它们是确定性的，所以断言可以严格。

用法: python3 tests/test_invariants.py
"""
from __future__ import annotations

import collections
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "build" / "engine" / "runner"
LIBDIR = ROOT / "build" / "engine"
AI = ROOT / "build" / "my_ai.so"

# 颜色对称的判据。
#
# 阈值定在 10 分是有依据的：跑引擎自带 AI 做对照，hunter 对 baseline 的红蓝
# 净胜分落差约 7（红 +8.8 / 蓝 +1.9），baseline 自对弈近乎对称（-0.6）——
# 也就是游戏本身的先手优势就值这个量级。超出它就不是"游戏固有不对称"了。
#
# 教训：曾经因为"两侧都赢、只是分差大"把这个阈值放宽到 25，结果漏掉了
# 一次真实的颜色不对称回归（红 +34 / 蓝 -6）。两侧都赢 ≠ 对称。
MAX_COLOR_GAP = 10.0
MIN_MARGIN = 1  # 每一侧至少净胜这么多分

failures: list[str] = []


def play(red: Path, blue: Path, max_turns: int = 20) -> dict:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{LIBDIR}:{env.get('LD_LIBRARY_PATH', '')}"
    proc = subprocess.run(
        [str(ENGINE), "--red", str(red), "--blue", str(blue),
         "--game-id", "inv", "--max-turns", str(max_turns)],
        capture_output=True, text=True, env=env, timeout=120,
    )
    import json
    for line in proc.stdout.splitlines():
        if '"game_over"' in line:
            return json.loads(line)
    raise RuntimeError(f"没有 game_over 事件\n{proc.stdout[:400]}\n{proc.stderr[:400]}")


def check_determinism(opponent: Path, runs: int = 5) -> None:
    print(f"[1/2] 确定性：vs {opponent.name} 重复 {runs} 次")
    seen = collections.Counter()
    for _ in range(runs):
        g = play(AI, opponent)
        seen[(g["winner"], g["red_score"], g["blue_score"], g["turns"])] += 1
    if len(seen) != 1:
        failures.append(
            f"不确定性：{runs} 次运行出现 {len(seen)} 种结果 {dict(seen)}")
        print(f"      ❌ {len(seen)} 种结果：{dict(seen)}")
    else:
        only = next(iter(seen))
        print(f"      ✅ 全部一致：winner={only[0]} {only[1]}:{only[2]} turns={only[3]}")


def check_color_symmetry(opponent: Path) -> None:
    print(f"[2/2] 颜色对称：vs {opponent.name}")
    margins = {}
    for label, ai_is_red in (("执红", True), ("执蓝", False)):
        red, blue = (AI, opponent) if ai_is_red else (opponent, AI)
        g = play(red, blue)
        mine = g["red_score"] if ai_is_red else g["blue_score"]
        theirs = g["blue_score"] if ai_is_red else g["red_score"]
        margins[label] = mine - theirs
        print(f"      {label}：{mine}:{theirs}  净胜 {mine - theirs:+d}  ({g['winner']})")
    losing = [label for label, m in margins.items() if m < MIN_MARGIN]
    if losing:
        failures.append(
            f"颜色不对称：{'、'.join(losing)} 未能取胜"
            f"（执红 {margins['执红']:+d}，执蓝 {margins['执蓝']:+d}）"
            f" —— 蓝方镜像/坐标框架疑似有误")
        print(f"      ❌ {'、'.join(losing)} 未取胜")
    else:
        print(f"      ✅ 两侧均取胜")

    gap = abs(margins["执红"] - margins["执蓝"])
    if gap > MAX_COLOR_GAP:
        failures.append(
            f"颜色落差过大：执红 {margins['执红']:+d} vs 执蓝 {margins['执蓝']:+d}，"
            f"落差 {gap:.0f} > {MAX_COLOR_GAP:.0f}")
        print(f"      ❌ 落差 {gap:.0f}，超过 {MAX_COLOR_GAP:.0f}")
    else:
        print(f"      ✅ 落差 {gap:.0f}，在 {MAX_COLOR_GAP:.0f} 内")


def main() -> int:
    for required in (ENGINE, AI):
        if not required.exists():
            sys.exit(f"缺少 {required}，先跑 make ai engine")
    for name in ("det_ai_a.so", "det_ai_b.so"):
        if not (ROOT / "build" / "opponents" / name).exists():
            sys.exit(f"缺少 build/opponents/{name}，先跑 make opponents-det")

    print("=== 阶段① AI 不变量测试 ===\n")
    check_determinism(ROOT / "build" / "opponents" / "det_ai_a.so")
    check_determinism(ROOT / "build" / "opponents" / "det_ai_b.so")
    print()
    check_color_symmetry(ROOT / "build" / "opponents" / "det_ai_a.so")
    check_color_symmetry(ROOT / "build" / "opponents" / "det_ai_b.so")

    print("\n--- 结果 ---")
    if failures:
        for f in failures:
            print(f"❌ {f}")
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

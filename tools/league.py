#!/usr/bin/env python3
"""league.py —— 两方 AI 的平衡对局评测（三个阶段共用的评测工具）

红蓝对调各跑一半，从 A 的视角汇总。入围赛的硬指标是**胜场严格大于对手**
（`a_wins > b_wins`），所以胜场单独列出，不只看胜率。

用法:
  python3 tools/league.py --a build/my_ai.so --b build/opponents/baseline_ai.so --games 100
  python3 tools/league.py --a ... --b ... --games 8 --json out.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "build" / "engine" / "runner"
LIBDIR = ROOT / "build" / "engine"

TIMEOUT_RE = re.compile(r"\[engine\] (red|blue) AI 超时")


def play(red: Path, blue: Path, max_turns: int) -> dict:
    """跑一局，返回结构化结果。"""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{LIBDIR}:{env.get('LD_LIBRARY_PATH', '')}"
    proc = subprocess.run(
        [str(ENGINE), "--red", str(red), "--blue", str(blue),
         "--game-id", "league", "--max-turns", str(max_turns)],
        capture_output=True, text=True, env=env, timeout=120,
    )
    result = {
        "returncode": proc.returncode,
        "winner": None, "reason": None,
        "red_score": None, "blue_score": None, "turns": None,
        "red_timeouts": 0, "blue_timeouts": 0,
        "stderr": proc.stderr,
    }
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "game_over":
            result.update(
                winner=event["winner"], reason=event["reason"],
                red_score=event["red_score"], blue_score=event["blue_score"],
                turns=event["turns"],
            )
    for side in TIMEOUT_RE.findall(proc.stderr):
        result[f"{side}_timeouts"] += 1
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="两方 AI 平衡对局评测")
    ap.add_argument("--a", required=True, help="A 方 .so")
    ap.add_argument("--b", required=True, help="B 方 .so")
    ap.add_argument("--games", type=int, default=100, help="总对局数（红蓝各半）")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--json", help="把汇总结果写入该文件")
    args = ap.parse_args()

    a = (ROOT / args.a).resolve() if not Path(args.a).is_absolute() else Path(args.a)
    b = (ROOT / args.b).resolve() if not Path(args.b).is_absolute() else Path(args.b)
    for path in (ENGINE, a, b):
        if not path.exists():
            sys.exit(f"找不到: {path}")

    half = args.games // 2
    # (红方, 蓝方, A 是否执红)
    schedule = [(a, b, True)] * half + [(b, a, False)] * (args.games - half)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(play, r, bl, args.max_turns) for r, bl, _ in schedule]
        raw = [f.result() for f in futures]

    wins = draws = losses = 0
    crashes = a_timeouts = b_timeouts = 0
    score_margin = 0
    reasons: dict[str, int] = {}
    for (_, _, a_is_red), g in zip(schedule, raw):
        if g["returncode"] != 0 or g["winner"] is None:
            crashes += 1
            continue
        reasons[g["reason"]] = reasons.get(g["reason"], 0) + 1

        a_timeouts += g["red_timeouts"] if a_is_red else g["blue_timeouts"]
        b_timeouts += g["blue_timeouts"] if a_is_red else g["red_timeouts"]

        a_score = g["red_score"] if a_is_red else g["blue_score"]
        b_score = g["blue_score"] if a_is_red else g["red_score"]
        score_margin += a_score - b_score

        if g["winner"] == "D":
            draws += 1
        elif (g["winner"] == "R") == a_is_red:
            wins += 1
        else:
            losses += 1

    played = wins + draws + losses
    if played == 0:
        sys.exit("没有一局正常结束")

    summary = {
        "a": str(a), "b": str(b), "played": played,
        "wins": wins, "draws": draws, "losses": losses,
        "score_rate": (wins + 0.5 * draws) / played,
        "win_rate": wins / played,
        "avg_margin": score_margin / played,
        "a_timeouts": a_timeouts, "b_timeouts": b_timeouts, "crashes": crashes,
        "reasons": reasons,
    }

    print(f"A = {a.name}")
    print(f"B = {b.name}")
    print(f"对局 {played} 局（红蓝各半）")
    print(f"  胜 {wins}  平 {draws}  负 {losses}")
    print(f"  综合得分率 {(wins + 0.5 * draws) / played:.4f}"
          f"   胜率 {wins / played:.4f}   平均净胜分 {score_margin / played:+.2f}")
    print(f"  A 超时 {a_timeouts}   B 超时 {b_timeouts}   异常退出 {crashes}")
    print(f"  结束原因 {reasons}")

    verdict = "✅ 胜场严格大于对手（入围赛口径）" if wins > losses else "❌ 未达入围赛口径（需 a_wins > b_wins）"
    print(f"  {verdict}")

    if args.json:
        Path(args.json).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"  汇总写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

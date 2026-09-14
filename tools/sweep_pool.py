#!/usr/bin/env python3
"""sweep_pool.py —— 对**对手池**取最差值的参数扫描

教训：只对着 baseline 调参会得到 vs baseline 100%、vs hunter 4% 的配置。
入围赛要求对 baseline 和 hunter **都**胜场大于对手，所以目标函数必须是
"对池中每个对手的得分率的最小值"，而不是单一对手的平均。

用法:
  python3 tools/sweep_pool.py --games 150
"""
from __future__ import annotations

import argparse
import itertools
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEAGUE = ROOT / "tools" / "league.py"
POOL = ["baseline_ai.so", "hunter_ai.so"]


def score_rate(opp: str, over: dict[str, str], games: int) -> float:
    env = os.environ.copy()
    env.update(over)
    proc = subprocess.run(
        [sys.executable, str(LEAGUE), "--a", "build/my_ai.so",
         "--b", f"build/opponents/{opp}", "--games", str(games),
         "--jobs", str(os.cpu_count() or 4)],
        capture_output=True, text=True, env=env, cwd=str(ROOT),
    )
    m = re.search(r"综合得分率 ([\d.]+)", proc.stdout + proc.stderr)
    return float(m.group(1)) if m else -1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=150, help="每个对手的局数")
    args = ap.parse_args()

    grid = list(itertools.product(
        [1.0, 1.3, 1.6, 2.0],          # w_danger
        [0.4, 0.6, 0.9, 1.2],          # w_threat
        [0.25, 0.4],                   # w_dist
    ))

    print(f"对手池 {POOL}，每格每对手 {args.games} 局，共 {len(grid)} 格\n")
    print(f"{'danger':>7}{'threat':>8}{'dist':>6} | " +
          "".join(f"{o.replace('_ai.so',''):>10}" for o in POOL) + f"{'最差':>9}{'均值':>8}")
    print("-" * 62)

    rows = []
    for danger, threat, dist in grid:
        over = {"ST_W_DANGER": str(danger), "ST_W_THREAT": str(threat),
                "ST_W_DIST": str(dist)}
        rates = [score_rate(o, over, args.games) for o in POOL]
        worst, avg = min(rates), sum(rates) / len(rates)
        rows.append((worst, avg, danger, threat, dist, rates))
        print(f"{danger:>7}{threat:>8}{dist:>6} | " +
              "".join(f"{r:>10.4f}" for r in rates) + f"{worst:>9.4f}{avg:>8.4f}")

    rows.sort(reverse=True)
    print("\n按『最差值』排序前 5：")
    for worst, avg, d, t, di, rates in rows[:5]:
        print(f"  最差 {worst:.4f}  均值 {avg:.4f}   danger={d} threat={t} dist={di}"
              f"   ({', '.join(f'{r:.3f}' for r in rates)})")
    print("\n注：最差值 < 0.5 表示至少还有一个对手打不过，达不到入围赛口径。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

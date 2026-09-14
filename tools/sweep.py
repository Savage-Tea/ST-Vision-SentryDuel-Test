#!/usr/bin/env python3
"""sweep.py —— 权重参数扫描（用实测胜率挑参数，不靠感觉）

用法:
  python3 tools/sweep.py --opp build/opponents/baseline_ai.so --games 100
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEAGUE = ROOT / "tools" / "league.py"

# 每行一个配置：环境变量覆盖值。
# 注意官方对手（baseline/hunter）用 steady_clock 给 PRNG 播种，每次运行都不同，
# 所以胜率是采样估计——局数要够，别拿 100 局的差异当真。
CONFIGS: list[tuple[str, dict[str, str]]] = [
    ("当前默认", {}),
    ("危险0.3", {"ST_W_DANGER": "0.3"}),
    ("危险0.5", {"ST_W_DANGER": "0.5"}),
    ("危险0.8", {"ST_W_DANGER": "0.8"}),
    ("威胁1.5", {"ST_W_THREAT": "1.5"}),
    ("威胁2.5", {"ST_W_THREAT": "2.5"}),
    ("占点1.5", {"ST_W_ZONE": "1.5"}),
    ("距离0.5", {"ST_W_DIST": "0.5"}),
    ("危险0.5+威胁1.5", {"ST_W_DANGER": "0.5", "ST_W_THREAT": "1.5"}),
    ("危险0.5+威胁1.5+距离0.5", {"ST_W_DANGER": "0.5", "ST_W_THREAT": "1.5", "ST_W_DIST": "0.5"}),
    ("危险0.3+威胁2.5+占点1.5", {"ST_W_DANGER": "0.3", "ST_W_THREAT": "2.5", "ST_W_ZONE": "1.5"}),
    ("危险0.8+威胁1.5+占点1.5+距离0.5",
     {"ST_W_DANGER": "0.8", "ST_W_THREAT": "1.5", "ST_W_ZONE": "1.5", "ST_W_DIST": "0.5"}),
    ("危0.3+威2.5+占1.5+距0.5",
     {"ST_W_DANGER": "0.3", "ST_W_THREAT": "2.5", "ST_W_ZONE": "1.5", "ST_W_DIST": "0.5"}),
    ("危0.5+威4+占1.5+距0.5",
     {"ST_W_DANGER": "0.5", "ST_W_THREAT": "4.0", "ST_W_ZONE": "1.5", "ST_W_DIST": "0.5"}),
]


def run(name: str, env_over: dict[str, str], opp: str, games: int) -> tuple[str, int, int, float]:
    env = os.environ.copy()
    env.update(env_over)
    proc = subprocess.run(
        [sys.executable, str(LEAGUE), "--a", "build/my_ai.so", "--b", opp,
         "--games", str(games), "--jobs", str(os.cpu_count() or 4)],
        capture_output=True, text=True, env=env, cwd=str(ROOT),
    )
    text = proc.stdout + proc.stderr
    m = re.search(r"胜 (\d+)  平 (\d+)  负 (\d+)", text)
    r = re.search(r"综合得分率 ([\d.]+)", text)
    if not m:
        print(f"  {name}: 失败\n{text[-400:]}")
        return name, -1, -1, 0.0
    return name, int(m.group(1)), int(m.group(3)), float(r.group(1)) if r else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", default="build/opponents/baseline_ai.so")
    ap.add_argument("--games", type=int, default=100)
    args = ap.parse_args()

    print(f"对手 {args.opp}，每配置 {args.games} 局（红蓝各半）\n")
    print(f"{'配置':<26}{'胜':>5}{'负':>5}{'得分率':>9}")
    print("-" * 46)
    rows = []
    for name, env_over in CONFIGS:
        n, w, l, rate = run(name, env_over, args.opp, args.games)
        rows.append((rate, name, w, l))
        print(f"{name:<26}{w:>5}{l:>5}{rate:>9.4f}")
    print("-" * 46)
    rows.sort(reverse=True)
    print("\n按得分率排序：")
    for rate, name, w, l in rows[:5]:
        print(f"  {rate:.4f}  {name}  (胜 {w} 负 {l})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

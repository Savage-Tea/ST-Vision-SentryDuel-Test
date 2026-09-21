#!/usr/bin/env python3
"""weight_fit.py —— 在风格池上做评估权重的坐标扫描

为什么必须用池子：平台的综合得分率是"与榜上每个对手各打 20 局"的平均，
衡量的是**广度**。只对 baseline/hunter 调出来的权重，在池平均下没有意义。

为什么用 -D 烘进二进制、而不是 ST_W_* 环境变量：引擎把红蓝两个 AI
dlopen 在**同一个进程**里，ST_* 是进程级共享的，用它做对照会把对手的
权重也一起改掉，实验直接失效。所以每个候选要编译成独立的 .so。

怎么读结果：只看池平均会骗人 —— 池子饱和时一点小波动就能换出"提升"。
必须同时看**坏象限数**（任一颜色 < 0.5 的对手个数）与最差象限。

用法:
  python3 tools/weight_fit.py
  python3 tools/weight_fit.py --coord SD_W_DANGER=2,6,8 --games 200
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build" / "fit"
POOL_EVAL = ROOT / "tools" / "pool_eval.py"

SRC = [
    "agent/act.cpp", "brain/eval.cpp", "brain/actions.cpp", "brain/search.cpp",
    "brain/mcts.cpp", "brain/belief_state.cpp", "brain/net.cpp",
    "brain/policy_net.cpp", "brain/value_net.cpp", "brain/ab_search.cpp",
    "obs/encode.cpp", "obs/encode_v3.cpp", "sim/view_mirror.cpp",
    "sim/rules.cpp", "sim/belief.cpp",
]

# 默认坐标：覆盖战略上真正独立的那几个量。
# 不动 w_diff（量纲基准）与 w_waste（只用于打破平局）。
DEFAULT_COORDS = {
    "SD_W_DANGER": [2.0, 4.0, 6.0, 8.0, 12.0],
    "SD_W_THREAT": [0.2, 0.4, 0.8],
    "SD_W_UNCERTAIN": [0.5, 0.75, 1.0],
    "SD_W_ZONE": [0.4, 0.8, 1.6],
}


def compile_candidate(name: str, defines: list[str]) -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out = BUILD_DIR / f"{name}.so"
    cmd = ["g++", "-std=c++17", "-O2", "-fPIC", "-I.",
           "-I../sentry-duel/engine/include", *defines, *SRC,
           "-shared", "-Wl,-z,lazy", "-Wl,--allow-shlib-undefined",
           "-o", str(out)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0 or not out.is_file():
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"编译失败: {name}")
    return out


def evaluate(so: Path, games: int) -> dict:
    jf = BUILD_DIR / f"{so.stem}.json"
    subprocess.run([sys.executable, str(POOL_EVAL), str(so),
                    "--games", str(games), "--json", str(jf)],
                   check=True, capture_output=True, cwd=ROOT)
    d = json.loads(jf.read_text())
    bad = [r["opponent"] for r in d["rows"] if min(r["red"], r["blue"]) < 0.5]
    worst = min(min(r["red"], r["blue"]) for r in d["rows"])
    return {"pooled": d["pooled"], "bad": bad, "worst": worst, "rows": d["rows"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--coord", action="append", default=[],
                    help="形如 SD_W_DANGER=2,6,8，可重复")
    args = ap.parse_args()

    coords: dict[str, list[float]] = {}
    for spec in args.coord:
        macro, _, vals = spec.partition("=")
        coords[macro] = [float(v) for v in vals.split(",")]
    if not coords:
        coords = DEFAULT_COORDS

    base = evaluate(compile_candidate("base", []), args.games)
    print(f"基线（当前默认权重）: 池平均 {base['pooled']:.4f}  "
          f"最差象限 {base['worst']:.4f}  坏象限 {len(base['bad'])} 个\n")

    print(f"{'坐标':<16}{'取值':>7}{'池平均':>10}{'最差象限':>10}{'坏象限':>8}")
    print("-" * 54)
    for macro, values in coords.items():
        best = None
        for v in values:
            so = compile_candidate(f"{macro}_{v}", [f"-D{macro}={v}"])
            r = evaluate(so, args.games)
            mark = ""
            if best is None or r["pooled"] > best[1]["pooled"]:
                best = (v, r)
            print(f"{macro:<16}{v:>7}{r['pooled']:>10.4f}{r['worst']:>10.4f}"
                  f"{len(r['bad']):>8}{mark}")
        v, r = best
        delta = r["pooled"] - base["pooled"]
        print(f"{'  → 该坐标最优':<16}{v:>7}{r['pooled']:>10.4f}"
              f"{r['worst']:>10.4f}{len(r['bad']):>8}   Δ{delta:+.4f}"
              f"{'  ⚠ 坏象限变多' if len(r['bad']) > len(base['bad']) else ''}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

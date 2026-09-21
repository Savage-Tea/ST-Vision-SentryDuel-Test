#!/usr/bin/env python3
"""pool_eval.py —— 候选 AI 对整个风格池的评测

为什么要池子：平台的综合得分率是"与榜上**每一个**对手各打 20 局"的平均
（server/store.py:810），所以它衡量的是**广度**——在多少个不同对局里赢。
只对 baseline/hunter 调的参数，在池子平均下没有意义。

为什么要报"有效样本"：我们的 AI 是确定性的，确定性对局里跑 2000 局可能只是
同一局跑了 2000 遍（实测：vs det_ai_a 是 1 种对局，vs baseline 是 2 种）。
所以每个对手都并列输出**不同对局数**，得分率必须和它一起看。

用法:
  python3 tools/pool_eval.py build/my_ai.so
  python3 tools/pool_eval.py build/my_ai.so --games 400
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEAGUE = ROOT / "tools" / "league.py"


def pool_members(exclude: Path) -> list[Path]:
    """风格池 = 官方对手 + 引擎确定性测试 AI + 历史快照 + 风格变体。

    跳过与候选**逐字节相同**的成员：把自己当对手是重复计数。
    （注意是按内容判，不是按路径——池里的 var_current.so 在评测别的候选时
    是一个合法的对手。）
    """
    mine = hashlib.md5(exclude.read_bytes()).hexdigest()
    members: list[Path] = []
    for d in (ROOT / "build" / "opponents", ROOT / "pool"):
        if d.is_dir():
            members += sorted(p for p in d.glob("*.so"))
    return [m for m in members
            if m.resolve() != exclude.resolve()
            and hashlib.md5(m.read_bytes()).hexdigest() != mine]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", help="候选 .so 路径")
    ap.add_argument("--games", type=int, default=200, help="每个对手的局数（红蓝各半）")
    ap.add_argument("--jobs", type=int, default=100)
    ap.add_argument("--json", help="把汇总写到该文件")
    args = ap.parse_args()

    cand = Path(args.candidate)
    if not cand.is_absolute():
        cand = ROOT / cand
    if not cand.is_file():
        sys.exit(f"找不到候选: {cand}")

    members = pool_members(cand)
    if not members:
        sys.exit("风格池是空的（build/opponents/ 与 pool/ 下没有 .so）")

    rows = []
    for opp in members:
        out = ROOT / "build" / f"pool_{opp.stem}.json"
        subprocess.run([sys.executable, str(LEAGUE), "--a", str(cand),
                        "--b", str(opp), "--games", str(args.games),
                        "--jobs", str(args.jobs), "--json", str(out)],
                       check=True, capture_output=True)
        d = json.loads(out.read_text())
        rows.append({
            "opponent": opp.name,
            "score_rate": d["score_rate"],
            "red": d["by_color"]["R"]["score_rate"],
            "blue": d["by_color"]["B"]["score_rate"],
            "distinct_games": d["distinct_games"],
            "margin_values": d["margin_values"],
            "crashes": d["crashes"],
            "timeouts": d["a_timeouts"],
        })

    print(f"候选 = {cand.name}   风格池 {len(rows)} 个对手 × {args.games} 局\n")
    print(f"{'对手':<28}{'综合':>7}{'执红':>8}{'执蓝':>8}{'有效样本':>10}{'净胜分取值':>11}")
    print("-" * 74)
    for r in rows:
        flag = "  ⚠" if min(r["red"], r["blue"]) < 0.5 else ""
        print(f"{r['opponent']:<28}{r['score_rate']:>7.4f}{r['red']:>8.4f}"
              f"{r['blue']:>8.4f}{r['distinct_games']:>10}{r['margin_values']:>11}{flag}")

    # 池平均 = 每个对手等权（与平台口径一致：每个对手各 20 局）
    pooled = sum(r["score_rate"] for r in rows) / len(rows)
    worst = min(rows, key=lambda r: min(r["red"], r["blue"]))
    min_quad = min(min(r["red"], r["blue"]) for r in rows)
    total_distinct = sum(r["distinct_games"] for r in rows)
    print("-" * 74)
    print(f"{'池平均':<28}{pooled:>7.4f}")
    print(f"  最差象限 {min_quad:.4f}（{worst['opponent']}）")
    print(f"  有效样本合计 {total_distinct} 种不同对局"
          f"（{len(rows) * args.games} 局名义样本）")
    if any(r["crashes"] or r["timeouts"] for r in rows):
        print("  ❌ 存在崩溃或超时")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"candidate": str(cand), "pooled": pooled, "rows": rows},
            ensure_ascii=False, indent=2))
        print(f"  汇总写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

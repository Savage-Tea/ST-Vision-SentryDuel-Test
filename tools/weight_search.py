#!/usr/bin/env python3
"""weight_search.py —— 权重坐标下降自动调优（风格池评测防过拟合）

对每个可调权重，扫描 {×0.5, ×1.0(跳过), ×2.0}，选最差象限最高的值。
一轮扫完所有权重后可再来一轮（此时基准已更新）。

用法:
  python3 tools/weight_search.py --rounds 2 --games 100
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "build" / "engine" / "runner"
LIBDIR = ROOT / "build" / "engine"

# 可调权重（env 后缀, 当前值, 描述）
WEIGHTS = [
    ("THREAT", 0.40, "对手在我火力通道内"),
    ("ZONE", 0.80, "位于得分区"),
    ("DIST", 0.40, "距得分区距离优势"),
]

# 对手池
OPPONENTS = [
    ("baseline", ROOT / "build" / "opponents" / "baseline_ai.so"),
    ("hunter", ROOT / "build" / "opponents" / "hunter_ai.so"),
    ("det_ai_a", ROOT / "build" / "opponents" / "det_ai_a.so"),
]

GAMES = 100
JOBS = 100


def parse_game_over(output: str) -> dict | None:
    for line in output.splitlines():
        if '"game_over"' in line:
            import json as j
            return j.loads(line)
    return None


def run_pool_eval(env_overrides: dict[str, str], games: int) -> dict[str, dict]:
    """对所有池对手各打 games/2 局，返回 {对手名: {red_rate, blue_rate, rate}}"""
    results = {}
    env = os.environ.copy()
    env.update(env_overrides)
    env["LD_LIBRARY_PATH"] = f"{LIBDIR}:{env.get('LD_LIBRARY_PATH', '')}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=JOBS) as pool:
        futures = {}
        for opp_name, opp_path in OPPONENTS:
            half = games // 2
            futures[opp_name] = []
            # A 执红
            for _ in range(half):
                futures[opp_name].append(pool.submit(
                    subprocess.run,
                    [str(ENGINE), "--red", str(ROOT / "build" / "my_ai.so"),
                     "--blue", str(opp_path),
                     "--game-id", "ws", "--max-turns", "20"],
                    capture_output=True, text=True, env=env, timeout=120))
            # A 执蓝
            for _ in range(half):
                futures[opp_name].append(pool.submit(
                    subprocess.run,
                    [str(ENGINE), "--red", str(opp_path),
                     "--blue", str(ROOT / "build" / "my_ai.so"),
                     "--game-id", "ws", "--max-turns", "20"],
                    capture_output=True, text=True, env=env, timeout=120))

        for opp_name, futs in futures.items():
            red_wins = red_games = blue_wins = blue_games = 0
            for i, f in enumerate(futures[opp_name]):
                out = f.result().stdout
                go = parse_game_over(out)
                if go is None:
                    continue
                a_is_red = i < len(futures[opp_name]) // 2
                my_score = go["red_score"] if a_is_red else go["blue_score"]
                opp_score = go["blue_score"] if a_is_red else go["red_score"]
                winner = go["winner"]
                won = (winner == "R") == a_is_red
                if a_is_red:
                    red_games += 1
                    if won: red_wins += 1
                else:
                    blue_games += 1
                    if won: blue_wins += 1
            red_rate = red_wins / red_games if red_games else 0
            blue_rate = blue_wins / blue_games if blue_games else 0
            rate = (red_wins + blue_wins) / max(red_games + blue_games, 1)
            results[opp_name] = {
                "red_rate": red_rate, "blue_rate": blue_rate, "rate": rate,
                "red_games": red_games, "blue_games": blue_games,
            }
    return results


def worst_quadrant(results: dict) -> float:
    return min(min(r["red_rate"], r["blue_rate"]) for r in results.values())


def format_results(results: dict) -> str:
    lines = []
    for name, r in results.items():
        lines.append(f"    {name}: 综合 {r['rate']:.3f}  "
                     f"(红 {r['red_rate']:.2f} / 蓝 {r['blue_rate']:.2f})")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--games", type=int, default=100, help="每配对局数")
    ap.add_argument("--weights", default="", help="JSON 文件（上轮结果），空则从头")
    args = ap.parse_args()

    GAMES = args.games

    # 当前权重值
    current = {name: val for name, val, _ in WEIGHTS}
    weight_desc = {name: desc for name, val, desc in WEIGHTS}

    # 加载上轮结果
    if args.weights and Path(args.weights).exists():
        current.update(json.loads(Path(args.weights).read_text()))
        print(f"已加载权重: {current}")

    print(f"════════ 权重自动调优 ════════")
    print(f"轮数: {args.rounds}  每配对: {GAMES} 局  对手池: {[n for n, _ in OPPONENTS]}")

    best_config = dict(current)
    best_worst = -1.0

    for round_num in range(1, args.rounds + 1):
        print(f"\n{'='*20} 第 {round_num}/{args.rounds} 轮 {'='*20}")

        for wname, _, desc in WEIGHTS:
            cur_val = current[wname]
            print(f"\n--- {wname} ({desc}) 当前 {cur_val} ---")

            candidates = [round(cur_val * 0.5, 2), round(cur_val * 2.0, 2)]
            # 去重
            candidates = [c for c in candidates if abs(c - cur_val) > 0.01]
            if not candidates:
                continue

            results_by_val = {}

            # 先评当前值作为基准
            print(f"  基准 {wname}={cur_val}:")
            results = run_pool_eval({f"ST_W_{wname}": str(cur_val)}, GAMES)
            wq = worst_quadrant(results)
            mean_rate = sum(r["rate"] for r in results.values()) / len(results)
            print(format_results(results))
            print(f"  worst-quadrant {wq:.3f}  mean {mean_rate:.3f}")
            results_by_val[cur_val] = (wq, mean_rate, results)

            for val in candidates:
                print(f"\n  测试 {wname}={val}:")
                results = run_pool_eval({f"ST_W_{wname}": str(val)}, GAMES)
                wq = worst_quadrant(results)
                mean_rate = sum(r["rate"] for r in results.values()) / len(results)
                print(format_results(results))
                print(f"  worst-quadrant {wq:.3f}  mean {mean_rate:.3f}")
                results_by_val[val] = (wq, mean_rate, results)

            # 选 worst-quadrant 最高的
            best_val = max(results_by_val, key=lambda v: results_by_val[v][0])
            best_wq = results_by_val[best_val][0]
            current[wname] = best_val
            print(f"\n  ★ {wname} 定为 {best_val} (worst {best_wq:.3f})")

            # 跟踪全局最优
            if best_wq > best_worst:
                best_worst = best_wq
                best_config = dict(current)

    # 输出最终结果
    print(f"\n{'='*20} 最终最优配置 {'='*20}")
    print(json.dumps(best_config, indent=2))
    print(f"最差象限: {best_worst:.3f}")

    out_path = ROOT / "build" / "weight_search_best.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(best_config, indent=2))
    print(f"\n最优配置写入 {out_path}")
    print("用 ST_W_* 环境变量或修改 eval.h 默认值来应用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

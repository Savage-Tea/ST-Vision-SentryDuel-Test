#!/usr/bin/env python3
"""value_m1.py —— M1：学出来的 V 是不是比手写 evaluate() 更会排序？

【要回答的问题】把叶评估从手写线性函数换成学出来的值网络，值不值得？
先用一个**可证伪、且不需要接线**的判据回答：在同一批决策状态上，谁的
取值更会排序真实结局。

【为什么不用 MSE/MAE】手写 evaluate() 不是为"预测分差"标定的（它是启发
式，量纲只是碰巧接近分）。用回归误差比它不公平。**成对排序准确率**
（pairwise ranking accuracy）是尺度无关的：随机取两条来自**不同对局**的
样本，谁的值高、谁的真实分差就更高——对了算 1，平局不计。两边同一个判据。

【数据】ST_DUMP_SAMPLES=1 时 act.cpp 在每个决策点打出
（手写评估值, 110 维特征）。与引擎 game_over 的最终分差拼接。
标签 = **我方视角**的最终分差（side=R 取红−蓝，side=B 取蓝−红）。

【划分】**按对局划分**，不是按样本。同一局的 20 个样本高度相关，
按样本划分会把测试集泄漏进训练集。

【判据】留出对局上，V 的排序准确率若不能明显超过 vhand，就**不接线**。

用法:
  python3 tools/value_m1.py --games 400
  python3 tools/value_m1.py --games 400 --data build/m1_samples.npz   # 复用已有数据
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.value_net import IN_DIM, ValueNet, export_weights  # noqa: E402

ENGINE = ROOT / "build" / "engine" / "runner"
LIBDIR = ROOT / "build" / "engine"
SAMPLE_RE = re.compile(r"\[sample\] turn=(\d+) side=(\w) vhand=([-\d.]+) f=([\d.,\-]+)")

# 对手混池：确定性对手给不出多样状态，必须掺随机化的（var_noise_*），
# 否则样本几乎全同，排序准确率没有意义。
OPPONENTS = ["var_noise_lo.so", "var_noise_mid.so", "hunter_ai.so", "var_rush.so"]


def generate(games: int, me: Path) -> dict:
    """跑对局、抓 stderr 的样本、拼上 game_over 的最终分差。"""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{LIBDIR}:{env.get('LD_LIBRARY_PATH','')}"
    env["ST_DUMP_SAMPLES"] = "1"
    X, y, vhand, gid, turn, side = [], [], [], [], [], []
    for i in range(games):
        opp = ROOT / "pool" / OPPONENTS[i % len(OPPONENTS)]
        if not opp.is_file():
            opp = ROOT / "build" / "opponents" / "hunter_ai.so"
        me_red = (i % 2 == 0)
        red, blue = (me, opp) if me_red else (opp, me)
        p = subprocess.run([str(ENGINE), "--red", str(red), "--blue", str(blue),
                            "--game-id", "m1", "--max-turns", "20"],
                           capture_output=True, text=True, env=env, timeout=180)
        over = None
        for line in p.stdout.splitlines():
            if '"game_over"' in line:
                over = json.loads(line)
        if over is None:
            continue
        margin_mine = ((over["red_score"] - over["blue_score"]) if me_red
                       else (over["blue_score"] - over["red_score"]))
        for line in p.stderr.splitlines():
            m = SAMPLE_RE.search(line)
            if not m:
                continue
            t, s, vh, feats = int(m.group(1)), m.group(2), float(m.group(3)), m.group(4)
            f = np.fromstring(feats, sep=",", dtype=np.float32)
            if f.shape[0] != IN_DIM:
                continue
            X.append(f); vhand.append(vh); y.append(margin_mine)
            gid.append(i); turn.append(t); side.append(s)
    return {"X": np.array(X, dtype=np.float32), "y": np.array(y, dtype=np.float32),
            "vhand": np.array(vhand, dtype=np.float32), "gid": np.array(gid),
            "turn": np.array(turn), "side": np.array(side)}


def rank_acc(v: np.ndarray, y: np.ndarray, mask: np.ndarray, n_pairs: int,
             rng: np.random.Generator) -> float:
    """成对排序准确率。只在 mask 内取样本，且只比较**标签不同**的成对。"""
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        return float("nan")
    a = rng.choice(idx, size=n_pairs)
    b = rng.choice(idx, size=n_pairs)
    ok = (y[a] != y[b]) & (a != b)
    if not ok.any():
        return float("nan")
    return float((np.sign(v[a][ok] - v[b][ok]) == np.sign(y[a][ok] - y[b][ok])).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--data", help="已有样本 npz（跳过生成）")
    ap.add_argument("--out", default="build/m1_samples.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export", action="store_true",
                    help="把训好的网络导出到 brain/value_weights.h"
                         "（先把输出层重标定成**分差单位**）")
    args = ap.parse_args()

    data_path = ROOT / args.out
    if args.data:
        d = dict(np.load(ROOT / args.data if not Path(args.data).is_absolute()
                         else args.data))
    else:
        print(f"生成样本：{args.games} 局 ...")
        d = generate(args.games, ROOT / "build" / "my_ai.so")
        np.savez_compressed(data_path, **d)
        print(f"  {len(d['y'])} 个样本 / {len(set(d['gid']))} 局 → {data_path}")

    X, y, vh, gid, turn = d["X"], d["y"], d["vhand"], d["gid"], d["turn"]
    print(f"样本 {len(y)}，对局 {len(set(gid.tolist()))}，"
          f"分差取值 {len(set(y.tolist()))} 种，"
          f"手写评估取值 {len(set(np.round(vh, 4).tolist()))} 种")

    # 按对局划分
    rng = np.random.default_rng(args.seed)
    games = np.array(sorted(set(gid.tolist())))
    rng.shuffle(games)
    n_val = max(1, int(len(games) * 0.25))
    val_games = set(games[:n_val].tolist())
    val = np.array([g in val_games for g in gid])
    tr = ~val

    # 标准化：只按训练集统计，避免泄漏
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd
    ymu, ysd = y[tr].mean(), y[tr].std() + 1e-6
    yn = (y - ymu) / ysd

    torch.manual_seed(args.seed)
    net = ValueNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    # ValueNet.forward 已经把最后一维 squeeze 掉，输出是 (N,)——
    # 标签**不要** unsqueeze：否则 MSE 会广播成 (N,N) 矩阵，训练结果是垃圾
    # （实测踩过：loss 打印 1.04，看着正常，其实是全错）。
    Xt = torch.from_numpy(Xn[tr]); yt = torch.from_numpy(yn[tr])
    Xv = torch.from_numpy(Xn[val]); yv = torch.from_numpy(yn[val])
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(Xt.shape[0])
        for i in range(0, Xt.shape[0], 4096):
            b = perm[i:i + 4096]
            opt.zero_grad()
            loss = lossf(net(Xt[b]), yt[b])
            loss.backward()
            opt.step()
        if (ep + 1) % 10 == 0:
            net.eval()
            with torch.no_grad():
                vl = lossf(net(Xv), yv).item()
            print(f"  epoch {ep+1:3d}  train {loss.item():.4f}  val {vl:.4f}")

    net.eval()
    with torch.no_grad():
        v_net = net(torch.from_numpy(Xn)).numpy()

    print(f"\n留出 {len(val_games)} 局 / {int(val.sum())} 样本\n")
    print(f"{'子集':<22}{'V(网络)':>10}{'手写 eval':>11}{'样本数':>8}")
    print("-" * 52)
    subsets = [("全部", val),
               ("前半局 turn<=10", val & (turn <= 10)),
               ("后半局 turn>10", val & (turn > 10)),
               ("执红", val & (d["side"] == "R")),
               ("执蓝", val & (d["side"] == "B"))]
    rng2 = np.random.default_rng(args.seed + 1)
    for name, m in subsets:
        a_net = rank_acc(v_net, y, m, 200000, rng2)
        a_h = rank_acc(vh, y, m, 200000, rng2)
        print(f"{name:<22}{a_net:>10.4f}{a_h:>11.4f}{int(m.sum()):>8}")
    if args.export:
        # 重标定：训练时标签标准化过（yn=(y-ymu)/ysd），把 ysd/ymu 折进输出层，
        # 网络就直接输出"分差（点）"。这样 C++ 侧不需要额外常数，
        # 而且与 evaluate() 处在同一量纲上（w_diff=1.0 ≈ 1 分），
        # 叶评估里那个 w_waste 平局项才有意义。配 SD_VALUE_SCALE=1.0。
        with torch.no_grad():
            net.fc3.weight.mul_(float(ysd))
            net.fc3.bias.mul_(float(ysd)).add_(float(ymu))
        export_weights(net, ROOT / "brain" / "value_weights.h")

    print("\n（0.5 = 抛硬币。V 若不能明显超过手写 eval，就不该接线。）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""train_value.py —— 值蒸馏训练器

输入：solve_ab --dump-values 的 10 字节记录（状态字段 + 真值 V* ∈ {-1,0,1}）
输出：brain/value_weights.h（部署进 phase① 叶评估，ST_VALUE_NET=1 启用）

验收：三分类精度（把回归输出按最近邻归到 {-1,0,1}）与 MAE——
部署强度仍要分色对局才算数，这里只回答"值函数学没学会"。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.value_net import IN_DIM, ValueNet, build_features, export_weights  # noqa: E402

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

WEIGHTS_HEADER = ROOT / "brain" / "value_weights.h"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--out", default="build/value.npz")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu", choices=["cpu", "npu", "cuda"])
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    data = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    raw = np.fromfile(data, dtype=np.uint8)
    assert raw.size % 10 == 0, "记录长度必须是 10 字节"
    rec = raw.reshape(-1, 10)
    print(f"读取值样本 {data}  ({len(rec):,} 条)")
    val = rec[:, 9].astype(np.int16) - 1.0  # {-1,0,1}
    X = torch.from_numpy(build_features(rec))
    Y = torch.from_numpy(val.astype(np.float32))
    print(f"  值分布: 胜 {int((val>0).sum()):,}  平 {int((val==0).sum()):,}  "
          f"负 {int((val<0).sum()):,}")

    if args.device == "cuda":
        DEV = torch.device("cuda:0")
        print(f"  设备 {torch.cuda.get_device_name(0)}")
    else:
        DEV = torch.device("cpu")

    X, Y = X.to(DEV), Y.to(DEV)
    net = ValueNet().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"  参数量 {sum(p.numel() for p in net.parameters()):,}")

    n = len(rec)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = max(1000, int(n * args.val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    def run(idx, train: bool) -> tuple[float, float, float]:
        net.train(train)
        tot_l = tot_hit = tot_n = 0.0
        order = rng.permutation(idx) if train else idx
        for s in range(0, len(order), args.batch):
            b = torch.from_numpy(np.sort(order[s : s + args.batch])).to(DEV)
            v = net(X[b]).squeeze(-1)
            loss = ((v - Y[b]) ** 2).mean()
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            with torch.no_grad():
                pred = torch.round(v.clamp(-1, 1))
                hit = (pred == Y[b]).float().sum()
            tot_l += float(loss) * len(b)
            tot_hit += float(hit)
            tot_n += len(b)
        return tot_l / tot_n, tot_hit / tot_n, 0.0

    print("\n  epoch |   训练MSE   三类精度 |   验证MSE   三类精度")
    best = (-1.0, None)
    for ep in range(1, args.epochs + 1):
        tr_l, tr_acc, _ = run(tr_idx, True)
        va_l, va_acc, _ = run(val_idx, False)
        marker = ""
        if va_acc > best[0]:
            best = (va_acc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
            marker = " *"
        print(f"  {ep:5d} | {tr_l:10.4f} {tr_acc*100:9.2f}% | {va_l:10.4f} {va_acc*100:9.2f}%{marker}")

    assert best[1] is not None
    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best[1], out_path)
    print(f"\n最佳验证三类精度 {best[0]*100:.2f}%")
    net.load_state_dict(best[1])
    export_weights(net, WEIGHTS_HEADER)
    print("\n下一步：make ai 后 ST_VALUE_NET=1 分色评测（强度才算数）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

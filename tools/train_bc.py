#!/usr/bin/env python3
"""train_bc.py —— 用求解器教师数据做行为克隆（阶段③ 的"老师"路线）

输入：tools/solve_ab.cpp --teacher 模式产出的 SDPP 轨迹
      （obs = 部分信息观测 v3，action = 全知教师的最优手，reward 恒 0）
输出：brain/policy_weights.h（与 PPO 同一套网络与导出，可直接 ST_POLICY=1 部署）

与 train_ppo.py 的区别：损失只有交叉熵——模仿教师的动作分布。
没有优势、没有裁剪、没有价值目标（reward 恒 0，价值头不参与部署逻辑）。

验收指标是**教师命中率**（argmax 与教师动作一致的比例），以及最终的
分色对局评测——命中率只说明"学没学会模仿"，强度仍要真打出来才算数。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import sys as _sys
_USE_NPU = "--device" in _sys.argv and "npu" in _sys.argv
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1" if _USE_NPU else "0")

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.policy_net import HIDDEN, Net, export_weights  # noqa: E402
from tools import sdpp  # noqa: E402

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

WEIGHTS_HEADER = ROOT / "brain" / "policy_weights.h"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="教师轨迹（SDPP）")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64, help="每条 minibatch 的序列数")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--out", default="build/bc.npz")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu", choices=["cpu", "npu", "cuda"])
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    data = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    print(f"读取教师轨迹 {data}")
    t = sdpp.read(data)
    n_total = sum(len(s.obs) for s in t.seqs)
    print(f"  序列 {len(t.seqs)} 条，总步数 {n_total:,}")

    # ---- 补齐到等长（与 train_ppo 相同的约定：padding 只加末尾）----
    lengths = np.array([len(s.obs) for s in t.seqs])
    T = int(lengths.max())
    B = len(t.seqs)
    obs = np.zeros((B, T, t.obs_dim), dtype=np.float32)
    act = np.zeros((B, T), dtype=np.int64)
    mask = np.zeros((B, T), dtype=np.float32)
    for i, s in enumerate(t.seqs):
        n = len(s.obs)
        obs[i, :n] = s.obs
        act[i, :n] = s.action
        mask[i, :n] = 1.0

    if args.device == "cuda":
        DEV = torch.device("cuda:0")
        print(f"  设备 {torch.cuda.get_device_name(0)}")
    elif args.device == "npu":
        DEV = torch.device("npu:0")
        import torch_npu  # noqa: F401
    else:
        DEV = torch.device("cpu")

    net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"  参数量 {sum(p.numel() for p in net.parameters()):,}")

    O = torch.from_numpy(obs).to(DEV)
    A = torch.from_numpy(act).to(DEV)
    M = torch.from_numpy(mask).to(DEV)

    rng = np.random.default_rng(0)
    n = B
    perm = rng.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    def run_epoch(idx, train: bool):
        net.train(train)
        tot_ce = tot_hit = tot_n = 0.0
        order = rng.permutation(idx) if train else idx
        for s in range(0, len(order), args.batch):
            b = torch.from_numpy(np.sort(order[s : s + args.batch]))
            bo, ba, bm = O[b], A[b], M[b]
            enc = torch.relu(net.enc(bo))
            h = torch.zeros(1, len(b), HIDDEN, device=DEV)
            out, _ = net.gru(enc, h)
            logits = net.pol(out)
            logp = torch.log_softmax(logits, dim=-1)
            ce = -(logp.gather(-1, ba.unsqueeze(-1)).squeeze(-1) * bm).sum() / bm.sum()
            if train:
                opt.zero_grad()
                ce.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            with torch.no_grad():
                hit = ((logits.argmax(-1) == ba).float() * bm).sum()
            tot_ce += float(ce) * float(bm.sum())
            tot_hit += float(hit)
            tot_n += float(bm.sum())
        return tot_ce / tot_n, tot_hit / tot_n

    print("\n  epoch |   训练CE   教师命中 |   验证CE   教师命中")
    best_acc = -1.0
    best_state = None
    for ep in range(1, args.epochs + 1):
        tr_ce, tr_hit = run_epoch(tr_idx, True)
        va_ce, va_hit = run_epoch(val_idx, False)
        marker = ""
        if va_hit > best_acc:
            best_acc = va_hit
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            marker = " *"
        print(f"  {ep:5d} | {tr_ce:9.4f} {tr_hit*100:8.2f}% | {va_ce:9.4f} {va_hit*100:8.2f}%{marker}")

    assert best_state is not None
    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_path)
    print(f"\n最佳验证教师命中 {best_acc*100:.2f}%")
    net.load_state_dict(best_state)
    export_weights(net, WEIGHTS_HEADER)
    print("\n下一步：make ai 后用 ST_POLICY=1 做分色评测（强度才算数）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

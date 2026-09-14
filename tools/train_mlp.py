#!/usr/bin/env python3
"""train_mlp.py —— 用自对弈样本训练策略/价值网络，并导出成 C 数组

输入：selfplay 生成的二进制样本（格式见 selfplay/selfplay.cpp）
输出：
  · brain/net_weights.h —— C++ 直接 #include 的权重数组（部署与自对弈共用）
  · <out>.npz            —— 训练检查点，便于续训

网络很小（410 → 256 → 256 → 策略头 8 + 价值头 1），CPU 就够。
真正的算力瓶颈在自对弈生成，不在这里。

用法:
  python3 tools/train_mlp.py --data data.bin --epochs 20 --out build/net.npz
"""
from __future__ import annotations

import argparse
import ctypes
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_HEADER = ROOT / "brain" / "net_weights.h"

MAGIC = b"SDSP"


class Sample(ctypes.Structure):
    """必须与 C++ 的 Sample 结构体逐字段对应（含对齐）。"""

    _fields_ = [
        ("obs", ctypes.c_float * 410),
        ("pi", ctypes.c_float * 8),
        ("z", ctypes.c_float),
        ("side", ctypes.c_char),
    ]


def load_samples(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    if raw[:4] != MAGIC:
        sys.exit(f"{path} 不是 SDSP 样本文件")
    version, obs_dim, act_dim = struct.unpack("<III", raw[4:16])
    if version != 1:
        sys.exit(f"不支持的样本版本 {version}")
    if (obs_dim, act_dim) != (410, 8):
        sys.exit(f"观测/动作维度不符: {obs_dim}/{act_dim}")

    sz = ctypes.sizeof(Sample)
    body = raw[16:]
    n = len(body) // sz
    if n == 0:
        sys.exit("样本为空")
    arr = (Sample * n).from_buffer_copy(body[: n * sz])

    obs = np.frombuffer(
        b"".join(bytes(a.obs) for a in arr), dtype=np.float32
    ).reshape(n, obs_dim) if False else np.array([list(a.obs) for a in arr], dtype=np.float32)
    pi = np.array([list(a.pi) for a in arr], dtype=np.float32)
    z = np.array([a.z for a in arr], dtype=np.float32)
    return obs, pi, z


class Net(nn.Module):
    def __init__(self, obs_dim: int = 410, act_dim: int = 8, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.pi = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)
        for layer in (self.fc1, self.fc2):
            nn.init.orthogonal_(layer.weight, gain=2 ** 0.5)
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, gain=1.0)
        nn.init.zeros_(self.v.bias)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.pi(h), torch.tanh(self.v(h)).squeeze(-1)


def export_header(net: Net, path: Path) -> None:
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    lines = [
        "// 自动生成，请勿手工修改 —— 由 tools/train_mlp.py 导出",
        "#pragma once",
        "",
        f"static const int kNetIn = {sd['fc1.weight'].shape[1]};",
        f"static const int kNetH1 = {sd['fc1.weight'].shape[0]};",
        f"static const int kNetH2 = {sd['fc2.weight'].shape[0]};",
        f"static const int kNetAct = {sd['pi.weight'].shape[0]};",
        "",
    ]

    def emit(name: str, arr: np.ndarray) -> None:
        flat = arr.reshape(-1).astype(np.float32)
        lines.append(f"static const float {name}[{flat.size}] = {{")
        chunk = []
        for i, v in enumerate(flat):
            chunk.append(f"{v:.9g}f")
            if len(chunk) == 8:
                lines.append("    " + ",".join(chunk) + ",")
                chunk = []
        if chunk:
            lines.append("    " + ",".join(chunk) + ",")
        lines.append("};")
        lines.append("")

    emit("kW1", sd["fc1.weight"])
    emit("kB1", sd["fc1.bias"])
    emit("kW2", sd["fc2.weight"])
    emit("kB2", sd["fc2.bias"])
    emit("kWp", sd["pi.weight"])
    emit("kBp", sd["pi.bias"])
    emit("kWv", sd["v.weight"])
    emit("kBv", sd["v.bias"])

    path.write_text("\n".join(lines))
    print(f"  权重头文件 {path.relative_to(ROOT)}  ({path.stat().st_size/1024:.0f} KB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="selfplay 生成的 .bin")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--out", default="build/net.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="0 = 用满 CPU")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    data_path = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    print(f"读取样本 {data_path}")
    obs, pi, z = load_samples(data_path)
    n = len(obs)
    print(f"  样本 {n} 条  (obs {obs.shape[1]} 维)")

    # 只用有策略信号的样本（π 全零的已被生成器丢弃，这里再兜一层）
    mask = pi.sum(axis=1) > 1e-6
    obs, pi, z = obs[mask], pi[mask], z[mask]
    print(f"  有效 {len(obs)} 条")
    if len(obs) < 100:
        sys.exit("样本太少，先多生成一些")

    perm = np.random.default_rng(args.seed).permutation(len(obs))
    n_val = max(1, int(len(obs) * args.val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X = torch.from_numpy(obs)
    P = torch.from_numpy(pi)
    Z = torch.from_numpy(z)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"  网络参数量 {sum(p.numel() for p in net.parameters()):,}")

    def epoch_loss(idx, train: bool):
        net.train(train)
        total_ce = total_v = total_n = 0.0
        order = idx if train else idx
        for s in range(0, len(order), args.batch):
            b = torch.from_numpy(order[s : s + args.batch])
            logits, v = net(X[b])
            logp = torch.log_softmax(logits, dim=-1)
            ce = -(P[b] * logp).sum(dim=-1).mean()
            ve = ((v - Z[b]) ** 2).mean()
            loss = ce + args.vf_coef * ve
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            total_ce += float(ce) * len(b)
            total_v += float(ve) * len(b)
            total_n += len(b)
        return total_ce / total_n, total_v / total_n

    print("\n  epoch |  训练 CE   训练 VE |  验证 CE   验证 VE | 策略命中率")
    best = None
    for ep in range(1, args.epochs + 1):
        tr_ce, tr_ve = epoch_loss(tr_idx, True)
        va_ce, va_ve = epoch_loss(val_idx, False)
        with torch.no_grad():
            logits, _ = net(X[torch.from_numpy(val_idx)])
            pred = logits.argmax(dim=-1).numpy()
            target = pi[val_idx].argmax(axis=1)
            acc = float((pred == target).mean())
        print(f"  {ep:5d} | {tr_ce:9.4f} {tr_ve:9.4f} | {va_ce:9.4f} {va_ve:9.4f} | {acc:9.3f}")
        best = net

    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_path)
    print(f"\n检查点 -> {out_path.relative_to(ROOT)}")
    export_header(net, WEIGHTS_HEADER)
    print("\n下一步：重新编译（make ai / make selfplay）即可让 MCTS 用上这个网络。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

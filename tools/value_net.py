#!/usr/bin/env python3
"""value_net.py —— 值蒸馏网络的定义与导出（与 tools/policy_net.py 平行）

特征契约 v2（110 维，信徒相对 + 信念平面）必须与 brain/value_net.h 严格一致：
  平面 0：我方位置 one-hot (49)
  平面 1：信念平面 (49)
  标量（12）：我方朝向 one-hot(4)、我方 fcd/2、我方 scd/3、turn/25、
              我方行动中(1)、ac/3（恒 0）、我方免费(1)、对手免费(1)、
              diff_mine/51
输出：我方视角期望博弈值 ∈ [-1,1]，线性。
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch
import torch.nn as nn

IN_DIM = 110
H1 = 256
H2 = 256

# (导出符号名, state_dict 键, 形状)
WEIGHTS: list[tuple[str, str, tuple[int, ...]]] = [
    ("kVw1", "fc1.weight", (H1, IN_DIM)),
    ("kVb1", "fc1.bias", (H1,)),
    ("kVw2", "fc2.weight", (H2, H1)),
    ("kVb2", "fc2.bias", (H2,)),
    ("kVw3", "fc3.weight", (1, H2)),
    ("kVb3", "fc3.bias", (1,)),
]


class ValueNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(IN_DIM, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, 1)
        for layer in (self.fc1, self.fc2):
            nn.init.orthogonal_(layer.weight, gain=2 ** 0.5)
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.fc3(h).squeeze(-1)


def build_features(rec: np.ndarray) -> np.ndarray:
    """把 --dump-values v2 的 20 字节记录 (N,20) 转成特征 (N,110)。

    布局必须与 brain/value_net.cpp 的 value_features 一致（v2 契约）。
    """
    N = rec.shape[0]
    pose = rec[:, 0].astype(np.int64) | (rec[:, 1].astype(np.int64) << 8)
    turn = rec[:, 2].astype(np.float32)
    me_move = rec[:, 3].astype(np.float32)
    ac = rec[:, 4].astype(np.float32)
    f_b = rec[:, 5].astype(np.float32)
    f_o = rec[:, 6].astype(np.float32)
    diff = rec[:, 7].astype(np.float32) - 51.0

    feats = np.zeros((N, IN_DIM), dtype=np.float32)
    cell = pose // 48
    fi = (pose // 12) % 4
    cd = pose % 12
    np.add.at(feats[:, 0:49], (np.arange(N), cell), 1.0)
    for byte in range(7):
        for bit in range(8):
            c = byte * 8 + bit
            if c < 49:
                feats[:, 49 + c] = ((rec[:, 8 + byte] >> bit) & 1).astype(np.float32)

    sc = feats[:, 98:]
    for i in range(4):
        sc[:, i] = (fi == i).astype(np.float32)
    sc[:, 4] = (cd // 4).astype(np.float32) / 2.0
    sc[:, 5] = (cd % 4).astype(np.float32) / 3.0
    sc[:, 6] = turn / 25.0
    sc[:, 7] = me_move
    sc[:, 8] = ac / 3.0
    sc[:, 9] = f_b
    sc[:, 10] = f_o
    sc[:, 11] = diff / 51.0
    return feats


def literal(v: float) -> str:
    if not np.isfinite(v):
        literal.bad += 1
        return "0.0f"
    s = f"{v:.9g}"
    if not any(ch in s for ch in ".eE"):
        s += ".0"
    return s + "f"


def export_weights(net: ValueNet, path: Path) -> None:
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    lines = [
        "// 自动生成，请勿手工修改 —— 由 tools/value_net.py 导出",
        "#pragma once",
        "",
        f"static const int kVnIn = {IN_DIM};",
        f"static const int kVnH1 = {H1};",
        f"static const int kVnH2 = {H2};",
        "",
    ]
    literal.bad = 0
    for sym, key, shape in WEIGHTS:
        arr = sd[key]
        if tuple(arr.shape) != shape:
            raise SystemExit(f"{key} 形状 {tuple(arr.shape)} 与预期 {shape} 不符")
        flat = arr.reshape(-1).astype(np.float32)
        lines.append(f"static const float {sym}[{flat.size}] = {{")
        chunk: list[str] = []
        for v in flat:
            chunk.append(literal(float(v)))
            if len(chunk) == 8:
                lines.append("    " + ",".join(chunk) + ",")
                chunk = []
        if chunk:
            lines.append("    " + ",".join(chunk) + ",")
        lines.append("};")
        lines.append("")
    if literal.bad:
        print(f"  ⚠ 有 {literal.bad} 个非有限权值，已写成 0")
    path.write_text("\n".join(lines))
    print(f"  权重头文件 {path}  ({path.stat().st_size / 1024:.0f} KB)")

    bin_path = path.with_suffix(".bin")
    with bin_path.open("wb") as f:
        f.write(struct.pack("<III", IN_DIM, H1, H2))
        for _sym, key, shape in WEIGHTS:
            f.write(np.ascontiguousarray(sd[key], dtype="<f4").tobytes())
    print(f"  权重二进制 {bin_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="导出随机值网络权重（供差分/冒烟）")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="brain/value_weights.h")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    net = ValueNet()
    print(f"参数量 {sum(p.numel() for p in net.parameters()):,}")
    export_weights(net, Path(args.out))

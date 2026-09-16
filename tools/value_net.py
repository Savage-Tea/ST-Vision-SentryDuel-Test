#!/usr/bin/env python3
"""value_net.py —— 值蒸馏网络的定义与导出（与 tools/policy_net.py 平行）

特征契约（116 维）必须与 brain/value_net.h 的注释及 value_features 实现
严格一致：
  平面 0：红方位置 one-hot (49)
  平面 1：蓝方位置 one-hot (49)
  标量（18）：红朝向 one-hot(4)、蓝朝向 one-hot(4)、红 fcd/2、蓝 fcd/2、
              红 scd/3、蓝 scd/3、turn/25、红方行动中(1)、ac/3（恒 0，
              部署查询点都在相位边界上）、free_red、free_blue、diff_red/51
输出：V_red ∈ [-1,1]，线性。
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch
import torch.nn as nn

IN_DIM = 116
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
    """把 solve_ab --dump-values 的原始记录 (N,10) 转成特征矩阵 (N,116)。

    布局必须与 brain/value_net.cpp 的 value_features 一致。
    """
    N = rec.shape[0]
    pr = rec[:, 0].astype(np.int64) | (rec[:, 1].astype(np.int64) << 8)
    pb = rec[:, 2].astype(np.int64) | (rec[:, 3].astype(np.int64) << 8)
    turn = rec[:, 4].astype(np.float32)
    side = rec[:, 5].astype(np.float32)
    ac = rec[:, 6].astype(np.float32)
    ff = rec[:, 7]
    diff = rec[:, 8].astype(np.float32) - 51.0

    feats = np.zeros((N, IN_DIM), dtype=np.float32)

    def put_pos(plane: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cd = pose % 12
        t = pose // 12
        fi = t % 4
        cell = t // 4
        np.add.at(plane, (np.arange(N), cell), 1.0)
        return fi, cd

    fi_r, cd_r = put_pos(feats[:, 0:49], pr)
    fi_b, cd_b = put_pos(feats[:, 49:98], pb)

    sc = feats[:, 98:]
    for i in range(4):
        sc[:, i] = (fi_r == i).astype(np.float32)
        sc[:, 4 + i] = (fi_b == i).astype(np.float32)
    sc[:, 8] = (cd_r // 4).astype(np.float32) / 2.0
    sc[:, 9] = (cd_b // 4).astype(np.float32) / 2.0
    sc[:, 10] = (cd_r % 4).astype(np.float32) / 3.0
    sc[:, 11] = (cd_b % 4).astype(np.float32) / 3.0
    sc[:, 12] = turn / 25.0
    sc[:, 13] = side
    sc[:, 14] = ac / 3.0
    sc[:, 15] = ((ff >> 1) & 1).astype(np.float32)
    sc[:, 16] = (ff & 1).astype(np.float32)
    sc[:, 17] = diff / 51.0
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

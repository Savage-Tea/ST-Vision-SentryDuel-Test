#!/usr/bin/env python3
"""policy_net.py —— 阶段③ 策略网络的定义与权重导出

被两处共用：
  · tools/difftest_policy.py —— 生成随机权重做 C++/torch 差分测试
  · tools/train_ppo.py       —— 训练后导出权重

**只此一份定义。** 导出格式与 brain/policy_net.cpp 的手写前向必须严格对应，
所以权重名、排布顺序都写在下面的 WEIGHTS 表里，改一处就得同时改两边——
policy_net.cpp 里有 static_assert 会把维度不符挡在编译期。

网络结构：
  obs(360) → Linear(360→128) → ReLU → GRU(128→256)
                                     ├─→ Linear(256→8)  策略 logits
                                     └─→ Linear(256→1)  价值（线性，无 tanh）
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

# 同 train_mlp.py：必须在 import torch 之前设置。昇腾节点上装了 torch_npu，
# torch 会在导入时自动加载设备后端，没 source CANN 就会直接 import 失败。
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch
import torch.nn as nn

OBS_DIM = 360
ENC_DIM = 128
HIDDEN = 256
ACT_DIM = 8

# (导出符号名, state_dict 键, 形状)
# nn.GRU 把三个门的权重纵向拼在一起，顺序是 [r; z; n]（PyTorch 的约定）。
# policy_net.cpp 按同样的顺序切段，这里改了那边也要改。
WEIGHTS: list[tuple[str, str, tuple[int, ...]]] = [
    ("kPvEncW", "enc.weight", (ENC_DIM, OBS_DIM)),
    ("kPvEncB", "enc.bias", (ENC_DIM,)),
    ("kPvGruWih", "gru.weight_ih_l0", (3 * HIDDEN, ENC_DIM)),
    ("kPvGruWhh", "gru.weight_hh_l0", (3 * HIDDEN, HIDDEN)),
    ("kPvGruBih", "gru.bias_ih_l0", (3 * HIDDEN,)),
    ("kPvGruBhh", "gru.bias_hh_l0", (3 * HIDDEN,)),
    ("kPvPolW", "pol.weight", (ACT_DIM, HIDDEN)),
    ("kPvPolB", "pol.bias", (ACT_DIM,)),
    ("kPvValW", "val.weight", (1, HIDDEN)),
    ("kPvValB", "val.bias", (1,)),
]


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Linear(OBS_DIM, ENC_DIM)
        self.gru = nn.GRU(ENC_DIM, HIDDEN, batch_first=True)
        self.pol = nn.Linear(HIDDEN, ACT_DIM)
        self.val = nn.Linear(HIDDEN, 1)
        for layer in (self.enc,):
            nn.init.orthogonal_(layer.weight, gain=2 ** 0.5)
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.pol.weight, gain=0.01)
        nn.init.zeros_(self.pol.bias)
        nn.init.orthogonal_(self.val.weight, gain=1.0)
        nn.init.zeros_(self.val.bias)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor):
        """obs: (B, T, OBS_DIM)  hidden: (B, HIDDEN) → logits (B,T,ACT), value (B,T), hidden'"""
        h = torch.relu(self.enc(obs))
        h, hidden = self.gru(h, hidden.unsqueeze(0))
        hidden = hidden.squeeze(0)
        # 价值头是线性的：奖励是 Δ 分差累积，回报量级可达 ±20，tanh 会长期饱和。
        return self.pol(h), self.val(h).squeeze(-1), hidden


def literal(v: float) -> str:
    """C++ 里 0f / 1f 不是合法字面量，必须有小数点或指数，所以补 .0。
    非有限权值会让编译直接失败，这里挡掉并计数。"""
    if not np.isfinite(v):
        literal.bad += 1
        return "0.0f"
    s = f"{v:.9g}"
    if not any(ch in s for ch in ".eE"):
        s += ".0"
    return s + "f"


def export_weights(net: Net, path: Path) -> None:
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    lines = [
        "// 自动生成，请勿手工修改 —— 由 tools/policy_net.py 导出",
        "#pragma once",
        "",
        f"static const int kPvObsIn = {OBS_DIM};",
        f"static const int kPvEnc = {ENC_DIM};",
        f"static const int kPvHidden = {HIDDEN};",
        f"static const int kPvAct = {ACT_DIM};",
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
        print(f"  ⚠ 有 {literal.bad} 个非有限权值，已写成 0（训练可能发散了）")
    path.write_text("\n".join(lines))
    print(f"  权重头文件 {path}  ({path.stat().st_size / 1024:.0f} KB)")

    # 同时导出一份二进制，给自对弈生成器做**运行时**加载（对手池用）。
    # 部署路径仍然只用头文件（编译进 .so，平台没有运行时文件可读）。
    # 布局：4×u32 维度，然后按 WEIGHTS 表顺序的原始 f32——
    # brain/policy_net.cpp 的 policy_load_bin 按同一张表读，两处必须同步。
    bin_path = path.with_suffix(".bin")
    with bin_path.open("wb") as f:
        f.write(struct.pack("<IIII", OBS_DIM, ENC_DIM, HIDDEN, ACT_DIM))
        for _sym, key, shape in WEIGHTS:
            f.write(np.ascontiguousarray(sd[key], dtype="<f4").tobytes())
    print(f"  权重二进制 {bin_path}  ({bin_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="用固定随机种子导出一份权重头文件（供差分测试）")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="brain/policy_weights.h")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    net = Net()
    n = sum(p.numel() for p in net.parameters())
    print(f"参数量 {n:,}")
    export_weights(net, Path(args.out))

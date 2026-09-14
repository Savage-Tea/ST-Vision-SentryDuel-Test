#!/usr/bin/env python3
"""difftest_policy.py —— GRU 前向的差分测试（Python 侧，驱动）

手写 GRU（brain/policy_net.cpp）必须与 torch 的 nn.GRU 给出同样的结果。
差异不会让程序崩溃，只会让"C++ 里部署的策略"和"torch 里训练的策略"悄悄
变成两个不同的东西——本项目已经踩过两次这类坑，所以这条测试是阶段③的
**前置门槛**：不通过就不许开训。

每次运行都重新导出随机权重，所以"头文件是旧的"这种失败模式不存在。

用法：python3 tools/difftest_policy.py [--seqs 24] [--steps 12] [--seed 1234]
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.policy_net import ACT_DIM, HIDDEN, OBS_DIM, Net, export_weights  # noqa: E402

import torch  # noqa: E402

# 容差：C++ 用 -O2（可能生成 FMA）与逐行累加，torch 走 BLAS，求和顺序不同。
# 本机实测干净时最大相对误差 4.5e-07；而注入的突变（GRU 更新式写反、
# 门偏移读错、价值头加 tanh）产生的最小相对误差是 3.8e-03——中间有 4 个
# 数量级的空档。取 1e-5 是站在干净侧的 22 倍余量上，同时仍能在突变小 380 倍
# 时就报警。
#
# 若换机器（不同 BLAS / CPU）后这个测试失败，先看报告的最大误差量级：
# 若是各处均匀的 ~1e-6 漂移，属正常，放宽阈值即可；若是某一段突然 O(1)，
# 那是真 bug，不要靠放宽阈值糊过去。
ATOL = 1e-5
RTOL = 1e-5


def build_cases(seqs: int, steps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """返回 (init_hidden (seqs,HIDDEN), obs (seqs,steps,OBS))"""
    rng = np.random.default_rng(seed)
    # 隐状态取自 N(0,0.5)：tanh/sigmoid 都在工作区，不会一上来就饱和
    init_hidden = (rng.standard_normal((seqs, HIDDEN)) * 0.5).astype(np.float32)
    obs = rng.standard_normal((seqs, steps, OBS_DIM)).astype(np.float32)
    return init_hidden, obs


def write_cases(path: Path, init_hidden: np.ndarray, obs: np.ndarray) -> None:
    seqs, steps, _ = obs.shape
    with path.open("wb") as f:
        f.write(b"SDPT")
        f.write(struct.pack("<III", 1, seqs, steps))
        for s in range(seqs):
            f.write(init_hidden[s].tobytes())
            f.write(obs[s].tobytes())


def torch_reference(net: Net, init_hidden: np.ndarray, obs: np.ndarray):
    """逐序列、逐步跑，与 C++ 的调用方式严格一致（每步一次 policy_forward）。"""
    seqs, steps, _ = obs.shape
    refs = np.empty((seqs, steps, ACT_DIM + 1 + HIDDEN), dtype=np.float32)
    with torch.no_grad():
        for s in range(seqs):
            h = torch.from_numpy(init_hidden[s]).unsqueeze(0)  # (1,HIDDEN)
            for t in range(steps):
                o = torch.from_numpy(obs[s, t]).view(1, 1, OBS_DIM)
                logits, value, h = net(o, h)
                refs[s, t, :ACT_DIM] = logits[0, 0].numpy()
                refs[s, t, ACT_DIM] = value[0, 0].item()
                refs[s, t, ACT_DIM + 1:] = h[0].numpy()
    return refs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=24)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1234)
    # 注意别和二进制 build/difftest_policy 同名——那是个文件，不是目录
    ap.add_argument("--out-dir", default="build/policy_cases")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 随机权重 → 头文件。必须在编译之前，否则编译进去的是上一轮的权重。
    torch.manual_seed(args.seed)
    net = Net().eval()
    print(f"[1/5] 生成随机权重 (seed={args.seed})")
    export_weights(net, ROOT / "brain" / "policy_weights.h")

    # 2) 输入序列
    init_hidden, obs = build_cases(args.seqs, args.steps, args.seed + 1)
    cases_path = out_dir / "cases.bin"
    write_cases(cases_path, init_hidden, obs)
    print(f"[2/5] 写出 {args.seqs} 条序列 × {args.steps} 步 → {cases_path.relative_to(ROOT)}")

    # 3) 编译
    #
    # 必须先删掉旧二进制：policy_weights.h 是**生成产物**，不是 make 的先决条件
    # （写进去会让 make 认为"产物比源码新"而跳过重编），于是刚导出的新权重
    # 根本不会被编译进去，测试就变成了拿旧权重自己跟自己比——永远通过。
    print("[3/5] 编译 build/difftest_policy")
    bin_path = ROOT / "build" / "difftest_policy"
    if bin_path.exists():
        bin_path.unlink()
    proc = subprocess.run(["make", "-s", "build/difftest_policy"], cwd=ROOT,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        return 1

    # 4) 跑 C++ 侧
    out_path = out_dir / "cpp_out.bin"
    proc = subprocess.run([str(ROOT / "build" / "difftest_policy"),
                           str(cases_path), str(out_path)],
                          capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return 1

    # 5) 比对
    raw = out_path.read_bytes()
    if raw[:4] != b"SDPO":
        print("❌ 输出文件 magic 不对", file=sys.stderr)
        return 1
    cpp = np.frombuffer(raw[8:], dtype=np.float32).reshape(args.seqs, args.steps, -1)

    print("[4/5] 跑 torch 参考前向")
    ref = torch_reference(net, init_hidden, obs)

    if cpp.shape != ref.shape:
        print(f"❌ 形状不符: cpp {cpp.shape} vs torch {ref.shape}", file=sys.stderr)
        return 1

    print("[5/5] 逐元素比对")
    diff = np.abs(cpp - ref)
    denom = np.maximum(np.abs(ref), 1.0)
    rel = diff / denom

    # 分段报告：策略/价值/隐状态各自的最大误差。混在一起报会掩盖某一头的偏差。
    parts = {"logits": slice(0, ACT_DIM), "value": slice(ACT_DIM, ACT_DIM + 1),
             "hidden": slice(ACT_DIM + 1, None)}
    worst = 0.0
    for name, sl in parts.items():
        d = diff[:, :, sl]
        r = rel[:, :, sl]
        print(f"  {name:7s} 最大绝对误差 {d.max():.3e}   最大相对误差 {r.max():.3e}")
        worst = max(worst, float(r.max()))

    ok = bool(np.allclose(cpp, ref, rtol=RTOL, atol=ATOL))
    if not ok:
        bad = np.argwhere(rel > RTOL)
        s, t, c = bad[0]
        print(f"\n❌ 差分测试失败：{len(bad)} 个元素超差，"
              f"首个在 序列{s} 第{t}步 通道{c}", file=sys.stderr)
        print(f"   cpp={cpp[s, t, c]:.9g}  torch={ref[s, t, c]:.9g}", file=sys.stderr)
        return 1

    print(f"\n✅ GRU 前向与 torch 一致（rtol={RTOL}, atol={ATOL}，最差 {worst:.2e}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

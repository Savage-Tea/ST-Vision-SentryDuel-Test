#!/usr/bin/env python3
"""train_ppo.py —— 阶段③ 的 PPO 训练

读 selfplay_ppo 产出的 SDPP 轨迹 → PPO 更新 → 导出 brain/policy_weights.h。

三处刻意的设计：

1. **old_logprob 在这里重算，不用 C++ 侧的值。**
   轨迹里根本不存 logprob（见 sdpp.py 的说明）。更新开始前先用当前权重跑一次
   前向拿到 old_logprob 与 old_value，然后才动梯度。这样 C++ 手写 GRU 与 torch
   之间的数值差异不会进入重要性比率——两份实现的漂移不会报错，只会让训练静默变质。

2. **按整条序列做 minibatch，不跨序列拼接。**
   每条序列是一个 (局, 一方) 的完整历史，hidden 从零开始。跨序列拼接会让
   hidden 串台，等于把上一局的记忆灌进下一局。

3. **价值头是线性的（不加 tanh）。**
   奖励是每回合 Δ 分差累积，终局回报量级可达 ±20，tanh 会长期饱和。
   这一点在 brain/policy_net.h 里也写了。

用法:
  python3 tools/train_ppo.py --data traj.bin --epochs 4 --out build/policy.npz
  python3 tools/train_ppo.py --export-only --out build/policy.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 必须在 import torch 之前。
# 默认关掉设备后端自动加载——昇腾节点上没 source CANN 时它会让 import 直接
# 失败，而我们之前一直跑 CPU，不该被这个连累。要用 NPU 时显式加 --device npu，
# 那一支会在 import 前把开关打开。
import sys as _sys
_USE_NPU = "--device" in _sys.argv and "npu" in _sys.argv
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1" if _USE_NPU else "0")

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.policy_net import ACT_DIM, HIDDEN, OBS_DIM, Net, export_weights  # noqa: E402
from tools import sdpp  # noqa: E402

WEIGHTS_HEADER = ROOT / "brain" / "policy_weights.h"

# 输出改为行缓冲。
#
# 默认情况下 stdout 重定向到文件时是**块缓冲**，而这个脚本每轮只打印不到
# 十行——缓冲区永远填不满，于是整个训练过程在日志里是**完全看不见的**，
# 要等进程退出才一次性刷出来。监控长跑时这等于瞎着眼睛等，也分不清
# "在算"和"卡住了"。同一个坑在 train_mlp.py 上已经踩过一次。
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover - 极老的 Python
    pass


def pad_sequences(seqs, max_len=None):
    """把变长序列补到等长，返回 (obs, action, reward, mask)。

    补齐一律加在**末尾**。GRU 是因果的，所以后面的 padding 不会影响前面
    有效步的输出——这是能这么补的前提。
    """
    lengths = np.array([len(s.obs) for s in seqs])
    T = int(lengths.max()) if max_len is None else max_len
    B = len(seqs)
    obs = np.zeros((B, T, OBS_DIM), dtype=np.float32)
    act = np.zeros((B, T), dtype=np.int64)
    rew = np.zeros((B, T), dtype=np.float32)
    mask = np.zeros((B, T), dtype=np.float32)
    for i, s in enumerate(seqs):
        n = len(s.obs)
        obs[i, :n] = s.obs
        act[i, :n] = s.action
        rew[i, :n] = s.reward
        mask[i, :n] = 1.0
    return obs, act, rew, mask, lengths


def forward(net, obs, lengths):
    """按序列前向。hidden 每条序列从零开始。返回 logits / value。"""
    B, T, _ = obs.shape
    h = torch.zeros(1, B, HIDDEN, device=obs.device)
    enc = torch.relu(net.enc(obs))
    out, _ = net.gru(enc, h)          # padding 在末尾，不影响有效步
    return net.pol(out), net.val(out).squeeze(-1)


def compute_gae(rew, value, mask, gamma, lam):
    """按序列算 GAE。每条序列都以真实终局结束，所以 bootstrap 恒为 0。"""
    B, T = rew.shape
    adv = np.zeros((B, T), dtype=np.float32)
    last = np.zeros(B, dtype=np.float32)
    v = value.detach().numpy()
    for t in range(T - 1, -1, -1):
        next_v = v[:, t + 1] if t + 1 < T else np.zeros(B, dtype=np.float32)
        # padding 之后的位置 mask=0，delta 必须为 0，否则会把 padding 的价值
        # 算进有效步的回报里
        delta = (rew[:, t] + gamma * next_v - v[:, t]) * mask[:, t]
        last = delta + gamma * lam * last * mask[:, t]
        adv[:, t] = last
    ret = adv + v
    return adv, ret


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="", help="SDPP 轨迹文件")
    ap.add_argument("--epochs", type=int, default=4, help="每批数据的复用轮数")
    ap.add_argument("--batch", type=int, default=32, help="每条 minibatch 的序列数")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--out", default="build/policy.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu", choices=["cpu", "npu"],
                    help="npu 需要先 source CANN 环境")
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()

    if args.export_only:
        ckpt = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
        if not ckpt.exists():
            sys.exit(f"检查点不存在: {ckpt}")
        net = Net()
        net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"从检查点导出 {ckpt}")
        export_weights(net, WEIGHTS_HEADER)
        return 0

    if not args.data:
        sys.exit("训练需要 --data（或用 --export-only）")

    # 同 train_mlp.py：这个网络很小，用满所有核只会被同步开销淹没
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    data = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    print(f"读取轨迹 {data}")
    t = sdpp.read(data)
    print(f"  序列 {len(t.seqs)} 条（{t.declared_games} 局 × 红蓝各一）")
    if len(t.seqs) < 8:
        sys.exit("序列太少，先多生成一些")

    obs, act, rew, mask, lengths = pad_sequences(t.seqs)
    print(f"  步数 中位 {int(np.median(lengths))} / 最大 {int(lengths.max())}")
    print(f"  奖励 非零步占比 {(rew != 0).mean() * 100:.2f}%   总和 {rew.sum():+.1f}")

    net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"  参数量 {sum(p.numel() for p in net.parameters()):,}")

    # 设备。大 batch 下 NPU 的算子派发开销能被摊薄，小 batch 反而更慢，
    # 所以这里的 batch 与 device 要一起调。
    DEV = torch.device(args.device) if args.device == "cpu" else torch.device("npu:0")
    if args.device == "npu":
        import torch_npu  # noqa: F401
        print(f"  设备 npu:0  可用={torch.npu.is_available()} 卡数={torch.npu.device_count()}")
    O = torch.from_numpy(obs).to(DEV)
    A = torch.from_numpy(act).to(DEV)
    R = torch.from_numpy(rew).to(DEV)
    M = torch.from_numpy(mask).to(DEV)
    old_logp_all = old_logp_all.to(DEV) if False else None

    rng = np.random.default_rng(args.seed)
    n = len(t.seqs)

    # —— 一次性算好 old_logprob / 优势 / 回报，整轮复用 ——
    #
    # **必须在任何梯度步之前算完。** 之前这里是在每个 minibatch 里重算的，
    # 后果有两个，且都不报错：
    #   · ratio = exp(new-old) 恒等于 1 —— PPO 的裁剪永远不触发，KL 恒为 0，
    #     整个算法退化成普通策略梯度，"clip 比例"这个指标永远是 0
    #   · GAE 用的 value 随权重一起变 —— 价值目标每一步都在漂移，价值头
    #     变成追一个移动靶，损失不降反升（实测 0.235 -> 0.795）
    # 正确的做法与 PPO 论文一致：old 是**采样时**的策略，整轮更新期间冻结。
    print("\n  预计算 old_logprob / GAE")
    old_logp_all = torch.zeros(n, obs.shape[1], device=DEV)
    adv_all = torch.zeros(n, obs.shape[1], device=DEV)
    ret_all = torch.zeros(n, obs.shape[1], device=DEV)
    with torch.no_grad():
        for s in range(0, n, args.batch):
            idx = np.arange(s, min(s + args.batch, n))
            b = torch.from_numpy(idx)
            bo, blen = O[b], lengths[idx]
            logits, value = forward(net, bo, blen)
            logp = torch.log_softmax(logits, dim=-1)
            old_logp_all[b] = logp.gather(-1, A[b].unsqueeze(-1)).squeeze(-1)
            adv, ret = compute_gae(R[b].cpu().numpy(), value.cpu(),
                                   M[b].cpu().numpy(), args.gamma, args.lam)
            adv_all[b] = torch.from_numpy(adv).to(DEV)
            ret_all[b] = torch.from_numpy(ret).to(DEV)

        # 优势标准化：只统计有效步，否则 padding 的 0 会把均值方差带偏
        valid_all = M.sum()
        adv_mean = (adv_all * M).sum() / valid_all
        adv_std = (((adv_all - adv_mean) ** 2 * M).sum() / valid_all).sqrt() + 1e-8
        adv_all = (adv_all - adv_mean) / adv_std
    print(f"  优势 均值 {float(adv_mean):+.4f} 标准差 {float(adv_std):.4f}")

    print("\n  epoch |   策略损失    价值损失    熵      近似KL   clip比例")
    for ep in range(1, args.epochs + 1):
        order = rng.permutation(n)
        agg = np.zeros(5)
        nb = 0
        for s in range(0, n, args.batch):
            idx = order[s : s + args.batch]
            b = torch.from_numpy(idx)
            bo, ba, bm = O[b], A[b], M[b]
            blen = lengths[idx]
            old_logp = old_logp_all[b]
            adv_t = adv_all[b]
            ret_t = ret_all[b]
            valid = bm.sum()

            logits, value = forward(net, bo, blen)
            logp = torch.log_softmax(logits, dim=-1)
            new_logp = logp.gather(-1, ba.unsqueeze(-1)).squeeze(-1)

            ratio = torch.exp(new_logp - old_logp)
            unclipped = ratio * adv_t
            clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_t
            pi_loss = -(torch.min(unclipped, clipped) * bm).sum() / valid
            v_loss = (((value - ret_t) ** 2) * bm).sum() / valid
            ent = -(logp.exp() * logp * bm.unsqueeze(-1)).sum() / valid

            loss = pi_loss + args.vf_coef * v_loss - args.ent_coef * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                approx_kl = (((ratio - 1) - (new_logp - old_logp)) * bm).sum() / valid
                clip_frac = (((ratio - 1).abs() > args.clip).float() * bm).sum() / valid
            with torch.no_grad():
                agg += np.array([float(pi_loss), float(v_loss), float(ent),
                                 float(approx_kl), float(clip_frac)])
            nb += 1

        agg /= max(nb, 1)
        print(f"  {ep:5d} | {agg[0]:10.4f} {agg[1]:11.4f} {agg[2]:8.4f} "
              f"{agg[3]:9.4f} {agg[4]:9.3f}")

    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, out_path)
    print(f"\n检查点 -> {out_path.relative_to(ROOT)}")
    export_weights(net, WEIGHTS_HEADER)
    print("\n下一步：make selfplay-ppo（重建生成器）后即可用新策略采样。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

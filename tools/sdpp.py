#!/usr/bin/env python3
"""sdpp.py —— SDPP 轨迹格式的读写（**只此一份**）

生成端是 selfplay/selfplay_ppo.cpp，消费端是 tools/train_ppo.py 与
tests/test_traj_format.py。三处必须严格同一套布局，所以读写只写在这里，
不要在任何地方再实现一遍。

格式：
    magic "SDPP" 4B | version u32 | obs_dim u32 | act_dim u32 | games u32
    每局：for side in {R,B}:
        steps u32
        steps 条 { obs f32[obs_dim], action u8, reward f32 }

**红蓝各成一条序列**，因为双方的 hidden state 绝不能共享。每条序列都以真实
终局结束，所以没有 done 标志，bootstrap 值恒为 0。

**不存 logprob**：old_logprob 由 torch 在 PPO 更新前用自己的前向重算，
这样 C++ 手写 GRU 与 torch 之间的数值差异不会污染重要性比率。
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"SDPP"
VERSION = 1


@dataclass
class Sequence:
    """一局中某一方的完整决策序列。"""
    obs: np.ndarray      # (T, obs_dim) float32
    action: np.ndarray   # (T,) uint8
    reward: np.ndarray   # (T,) float32


@dataclass
class Trajectories:
    seqs: list[Sequence]
    obs_dim: int
    act_dim: int
    declared_games: int


def read(path: str | Path) -> Trajectories:
    raw = Path(path).read_bytes()
    if raw[:4] != MAGIC:
        raise ValueError(f"{path} 不是 SDPP 文件（magic={raw[:4]!r}）")
    version, obs_dim, act_dim, games = struct.unpack_from("<IIII", raw, 4)
    if version != VERSION:
        raise ValueError(f"不支持的版本 {version}（本工具只认 {VERSION}）")

    step_bytes = obs_dim * 4 + 1 + 4
    seqs: list[Sequence] = []
    off = 20
    for _ in range(games):
        for _side in range(2):  # 每局两条：红、蓝
            (n,) = struct.unpack_from("<I", raw, off)
            off += 4
            end = off + n * step_bytes
            if end > len(raw):
                raise ValueError("轨迹文件被截断")

            # 逐字段切片再 copy：obs 是定长浮点块，action 是 1 字节，
            # 中间夹着这两个字段，所以不能整块 view 成结构化 dtype 后直接读。
            buf = np.frombuffer(raw, dtype=np.uint8, count=n * step_bytes, offset=off)
            buf = buf.reshape(n, step_bytes)
            obs = buf[:, : obs_dim * 4].copy().view(np.float32).reshape(n, obs_dim)
            action = buf[:, obs_dim * 4].copy()
            reward = buf[:, obs_dim * 4 + 1 :].copy().view(np.float32).reshape(n)

            seqs.append(Sequence(obs=obs, action=action, reward=reward))
            off = end

    if off != len(raw):
        raise ValueError(f"文件尾部有 {len(raw) - off} 字节未消费——读写两侧的布局已经不一致了")

    return Trajectories(seqs=seqs, obs_dim=obs_dim, act_dim=act_dim,
                        declared_games=games)


def write(path: str | Path, trajs: Trajectories) -> None:
    """按同一套布局写回。用于 round-trip 测试。"""
    out = bytearray()
    out += MAGIC
    out += struct.pack("<IIII", VERSION, trajs.obs_dim, trajs.act_dim, trajs.declared_games)
    for s in trajs.seqs:
        n = len(s.obs)
        out += struct.pack("<I", n)
        buf = np.empty((n, trajs.obs_dim * 4 + 5), dtype=np.uint8)
        buf[:, : trajs.obs_dim * 4] = s.obs.astype(np.float32).view(np.uint8).reshape(
            n, trajs.obs_dim * 4)
        buf[:, trajs.obs_dim * 4] = s.action.astype(np.uint8)
        buf[:, trajs.obs_dim * 4 + 1 :] = (
            s.reward.astype(np.float32).view(np.uint8).reshape(n, 4))
        out += buf.tobytes()
    Path(path).write_bytes(bytes(out))

#!/usr/bin/env python3
"""pack.py —— 生成可直接上传到赛事平台的源码包

平台契约（server/upload.py）：
  · 压缩包 ≤20MB、≤64 文件、单文件 ≤8MB
  · 项目根须有 Makefile（否则只编译根层 .cpp，深层子目录不参与）
  · Makefile 可从环境变量拿 ENGINE_INCLUDE / ENGINE_LIB_DIR
  · .so 必须产出到项目根
  · 主源码优先取根层的 my_ai.cpp（决定 AI 列表里展示的源码）

所以这里把工程摆成：
  my_ai.cpp             ← agent/act.cpp 的副本
  Makefile              ← tools/pack_Makefile
  sim/ brain/           ← 原样拷贝（include 路径不变）
并打成 build/upload/my_ai_pack.zip。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "build" / "upload" / "pack"
ZIP_PATH = ROOT / "build" / "upload" / "my_ai_pack.zip"

# (源文件, 包内路径)
FILES: list[tuple[str, str]] = [
    ("agent/act.cpp", "my_ai.cpp"),
    ("sim/state.h", "sim/state.h"),
    ("sim/rules.h", "sim/rules.h"),
    ("sim/rules.cpp", "sim/rules.cpp"),
    ("sim/belief.h", "sim/belief.h"),
    ("sim/belief.cpp", "sim/belief.cpp"),
    ("obs/encode.h", "obs/encode.h"),
    ("obs/encode.cpp", "obs/encode.cpp"),
    ("brain/eval.h", "brain/eval.h"),
    ("brain/eval.cpp", "brain/eval.cpp"),
    ("brain/actions.h", "brain/actions.h"),
    ("brain/actions.cpp", "brain/actions.cpp"),
    ("brain/search.h", "brain/search.h"),
    ("brain/search.cpp", "brain/search.cpp"),
    ("brain/mcts.h", "brain/mcts.h"),
    ("brain/mcts.cpp", "brain/mcts.cpp"),
    ("brain/belief_state.h", "brain/belief_state.h"),
    ("brain/belief_state.cpp", "brain/belief_state.cpp"),
    ("brain/net.h", "brain/net.h"),
    ("brain/net.cpp", "brain/net.cpp"),
    # —— 阶段③ 的策略网络路径（ST_POLICY=1；默认不启用，但 act.cpp 引用了
    #    policy_forward 符号，清单不齐就是 dlopen 事故）——
    ("brain/policy_net.h", "brain/policy_net.h"),
    ("brain/policy_net.cpp", "brain/policy_net.cpp"),
    ("brain/value_net.h", "brain/value_net.h"),
    ("brain/value_net.cpp", "brain/value_net.cpp"),
    ("brain/opening_book.h", "brain/opening_book.h"),
    ("brain/opening_book.cpp", "brain/opening_book.cpp"),
    ("brain/ab_search.h", "brain/ab_search.h"),
    ("brain/ab_search.cpp", "brain/ab_search.cpp"),
    ("obs/encode_v3.h", "obs/encode_v3.h"),
    ("obs/encode_v3.cpp", "obs/encode_v3.cpp"),
    ("sim/view_mirror.h", "sim/view_mirror.h"),
    ("sim/view_mirror.cpp", "sim/view_mirror.cpp"),
    # 训练产出的权重。net_weights.h 缺了会回退手工评估（等于没带模型）；
    # policy_weights.h 支撑 ST_POLICY 路径。打包时缺失即报错。
    ("brain/net_weights.h", "brain/net_weights.h"),
    ("brain/policy_weights.h", "brain/policy_weights.h"),
    ("tools/pack_Makefile", "Makefile"),
]

LIMITS = {"total": 20 * 1024 * 1024, "files": 64, "single": 8 * 1024 * 1024}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="打包后在本机模拟平台的编译流程并跑一局验证")
    args = ap.parse_args()

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    for src, dst in FILES:
        s = ROOT / src
        if not s.exists():
            sys.exit(f"缺少源文件 {s}")
        d = STAGE / dst
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s, d)

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(STAGE))

    members = [p for p in STAGE.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in members)
    biggest = max(members, key=lambda p: p.stat().st_size)
    print(f"打包完成: {ZIP_PATH.relative_to(ROOT)}")
    print(f"  {len(members)} 个文件, 未压缩共 {total / 1024:.1f} KB, "
          f"最大单文件 {biggest.name} {biggest.stat().st_size / 1024:.1f} KB")

    for label, value, limit in (("总大小", total, LIMITS["total"]),
                                ("文件数", len(members), LIMITS["files"]),
                                ("最大单文件", biggest.stat().st_size, LIMITS["single"])):
        if value > limit:
            sys.exit(f"❌ {label} 超过平台限制: {value} > {limit}")
    if not (STAGE / "my_ai.cpp").exists():
        sys.exit("❌ 包根缺少 my_ai.cpp")
    if not (STAGE / "Makefile").exists():
        sys.exit("❌ 包根缺少 Makefile")
    print("  ✅ 满足平台限制（大小/文件数/主源码/Makefile）")

    if args.verify:
        return verify()
    print("\n提示：加 --verify 可在本机模拟平台编译流程并跑一局")
    return 0


def verify() -> int:
    """模拟平台：解压到干净目录 → make -C root（注入 ENGINE_*）→ 跑一局。"""
    print("\n=== 模拟平台编译 ===")
    work = ROOT / "build" / "upload" / "verify"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(work)

    import os
    env = os.environ.copy()
    env["ENGINE_INCLUDE"] = str((ROOT / "../sentry-duel/engine/include").resolve())
    env["ENGINE_LIB_DIR"] = str(ROOT / "build/engine")

    proc = subprocess.run(["make", "-C", str(work)], capture_output=True, text=True,
                          env=env, timeout=120)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        sys.exit("❌ make 失败")
    so = work / "my_ai.so"
    if not so.exists():
        sys.exit("❌ 未产出 my_ai.so")
    print(f"  ✅ make 成功，产出 my_ai.so ({so.stat().st_size / 1024:.1f} KB)")

    print("\n=== 用这个 .so 跑两局（红/蓝各一）===")
    runner = ROOT / "build" / "engine" / "runner"
    renv = os.environ.copy()
    renv["LD_LIBRARY_PATH"] = str(ROOT / "build" / "engine")
    for label, ai_is_red in (("执红", True), ("执蓝", False)):
        red, blue = ((so, ROOT / "build/opponents/det_ai_a.so") if ai_is_red
                     else (ROOT / "build/opponents/det_ai_a.so", so))
        p = subprocess.run([str(runner), "--red", str(red), "--blue", str(blue),
                            "--game-id", "pack", "--max-turns", "20"],
                           capture_output=True, text=True, env=renv)
        line = next((l for l in p.stdout.splitlines() if '"game_over"' in l), None)
        if line is None:
            sys.exit(f"❌ {label} 没有 game_over")
        print(f"  {label}: {line.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

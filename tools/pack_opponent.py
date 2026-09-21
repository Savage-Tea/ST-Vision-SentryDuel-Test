#!/usr/bin/env python3
"""pack_opponent.py —— 把引擎自带的单文件 AI 打成平台可上传包

用途：把 det_ai_a / det_ai_b / baseline_ai / hunter_ai 这类单文件 AI 丢上榜单。
它们强度已知、实现只有一百来行，可以当**标尺**：如果一段 115 行的反射程序在
平台上的得分率高于我们 580 行的搜索+评估 AI，那说明平台奖励的是别的东西。

平台契约与 tools/pack.py 相同，差别只有两点：
  · 只打一个源文件，不依赖 sim/ brain/ obs/；
  · 引擎头文件（sentry_duel.h / utils.h）由平台的 ENGINE_INCLUDE 提供，不打进包。

用法:
  python3 tools/pack_opponent.py ../sentry-duel/engine/tests/det_ai_a.cpp
  python3 tools/pack_opponent.py ../sentry-duel/ai/baseline_ai.cpp --out xxx.zip
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "build" / "upload"

MAKEFILE = """\
# 单文件对手 AI —— 由 tools/pack_opponent.py 生成
#
# 平台执行 `make -C <项目根>`，并注入 ENGINE_INCLUDE / ENGINE_LIB_DIR，
# 要求把 .so 产出到项目根。

CXX            ?= g++
ENGINE_INCLUDE ?= ../engine/include
ENGINE_LIB_DIR ?= ../engine/build

CXXFLAGS = -std=c++17 -O2 -fPIC -Wall -Wextra -I. -I$(ENGINE_INCLUDE)
# move/turn/fire/scan 由 runner 进程在 dlopen 时提供，故必须 lazy + 允许未定义符号
LDFLAGS  = -shared -Wl,-z,lazy -Wl,--allow-shlib-undefined \\
           -L$(ENGINE_LIB_DIR) -lsentry_duel_engine

SRC = my_ai.cpp

all: my_ai.so

my_ai.so: $(SRC)
\t$(CXX) $(CXXFLAGS) $(LDFLAGS) $(SRC) -o $@

clean:
\trm -f my_ai.so

.PHONY: all clean
"""

LIMITS = {"total": 20 * 1024 * 1024, "files": 64, "single": 8 * 1024 * 1024}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="单文件 AI 的 .cpp 路径")
    ap.add_argument("--out", default=None, help="输出 zip 名（默认 my_ai_pack_<源文件名>.zip）")
    ap.add_argument("--verify", action="store_true",
                    help="解压到干净目录模拟平台编译，再用产出的 .so 跑一局")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.is_absolute():
        src = (ROOT / src).resolve()
    if not src.is_file():
        sys.exit(f"找不到源文件: {src}")

    zip_path = UPLOAD_DIR / (args.out or f"my_ai_pack_{src.stem}.zip")
    if zip_path.exists():
        sys.exit(f"❌ {zip_path.relative_to(ROOT)} 已存在——打包不覆盖历史包，"
                 f"请用 --out 指定新名字")

    stage = UPLOAD_DIR / f"pack_{src.stem}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copyfile(src, stage / "my_ai.cpp")
    (stage / "Makefile").write_text(MAKEFILE)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(stage))

    members = [p for p in stage.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in members)
    print(f"打包完成: {zip_path.relative_to(ROOT)}")
    print(f"  源文件 {src.name} → my_ai.cpp ({src.stat().st_size} B)")
    print(f"  {len(members)} 个文件, 未压缩共 {total / 1024:.1f} KB")
    if len(members) > LIMITS["files"] or total > LIMITS["total"]:
        sys.exit("❌ 超过平台限制")

    return verify(zip_path, src.stem) if args.verify else 0


def verify(zip_path: Path, stem: str) -> int:
    """模拟平台：解压 → make -C root（注入 ENGINE_*）→ 用它跑一局。"""
    print("\n=== 模拟平台编译 ===")
    work = ROOT / "build" / "upload" / f"verify_{stem}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(work)

    import os
    env = os.environ.copy()
    env["ENGINE_INCLUDE"] = str((ROOT / "../sentry-duel/engine/include").resolve())
    env["ENGINE_LIB_DIR"] = str(ROOT / "build/engine")
    proc = subprocess.run(["make", "-C", str(work)], capture_output=True,
                          text=True, env=env, timeout=120)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        sys.exit("❌ make 失败")
    so = work / "my_ai.so"
    if not so.exists():
        sys.exit("❌ 未产出 my_ai.so")
    print(f"  ✅ make 成功，产出 my_ai.so ({so.stat().st_size / 1024:.1f} KB)")

    print("\n=== 用这个 .so 各跑一局（红/蓝）===")
    runner = ROOT / "build" / "engine" / "runner"
    renv = os.environ.copy()
    renv["LD_LIBRARY_PATH"] = str(ROOT / "build" / "engine")
    mine = ROOT / "build" / "my_ai.so"
    for label, is_red in (("执红", True), ("执蓝", False)):
        red, blue = (so, mine) if is_red else (mine, so)
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

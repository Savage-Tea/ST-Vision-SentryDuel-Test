#!/usr/bin/env python3
"""make_book_header.py —— 开局书二进制 → C++ 头文件

用法: python3 tools/make_book_header.py [--bin build/opening_book.bin]
输出: brain/opening_book_data.h (kBookData / kBookN / kBookTurns)
"""
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    bin_path = ROOT / "build" / "opening_book.bin"
    out_path = ROOT / "brain" / "opening_book_data.h"
    for a in ("--bin",):
        for i, arg in enumerate(sys.argv):
            if arg == a and i + 1 < len(sys.argv):
                bin_path = Path(sys.argv[i + 1])
    raw = bin_path.read_bytes()
    (book_turns,) = struct.unpack_from("<I", raw, 0)
    body = raw[4:]
    assert len(body) % 9 == 0, f"body {len(body)} 不是 9 的倍数"
    n = len(body) // 9

    lines = [
        "// 自动生成，请勿手工修改 —— 由 tools/make_book_header.py 导出",
        "#pragma once",
        "",
        f"static const int kBookTurns = {book_turns};",
        f"static const int kBookN = {n};",
        f"static const unsigned char kBookData[] = {{",
    ]
    for i in range(0, len(body), 16):
        chunk = body[i : i + 16]
        lines.append("    " + ",".join(str(b) for b in chunk) + ",")
    lines.append("};")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  开局书头文件 {out_path}  (turns={book_turns}, {n} 条)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

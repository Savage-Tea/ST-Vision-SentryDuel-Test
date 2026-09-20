#!/usr/bin/env bash
# tools/pool_eval.sh —— 候选 AI 对风格池的完整评测
#
# 用法: bash tools/pool_eval.sh <候选.so> [局数=200]
#
# 风格池：
#   baseline     官方守中狙路过（camper 型）
#   hunter       官方追击型
#   det_ai_a     确定性 SCAN→FIRE→占中心（camper+雷达 型）
#   det_ai_b     确定性测试型
#   sitter       纯占点流（我们写的最简策略）
#   pool/ 里的历史版本（我们自己之前的样子）
#
# 选型标准：最差象限（四配对中最低的胜率）最大化。
set -euo pipefail
cd "$(dirname "$0")/.."

CANDIDATE=${1:?用法: pool_eval.sh <候选.so> [局数]}
GAMES=${2:-200}

make opponents >/dev/null 2>&1
make opponents-det >/dev/null 2>&1

# 收集池成员
POOL_FILES=(
    build/opponents/baseline_ai.so
    build/opponents/hunter_ai.so
    build/opponents/det_ai_a.so
    build/opponents/det_ai_b.so
    build/sitter.so
)
POOL_NAMES=(baseline hunter det_a det_b sitter)

# 加 pool/ 目录里的历史版本
for f in pool/*.so; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .so)
    POOL_FILES+=("$f")
    POOL_NAMES+=("$name")
done

echo "候选: $CANDIDATE"
echo "池: ${POOL_NAMES[*]}"
echo "局数: 每配对 $GAMES"
echo

WORST=101
WORST_NAME=""
TOTAL=0
COUNT=0

for i in "${!POOL_FILES[@]}"; do
    opp="${POOL_FILES[$i]}"
    name="${POOL_NAMES[$i]}"
    [ -f "$opp" ] || { echo "  $name: 文件缺失，跳过"; continue; }

    result=$(python3 tools/league.py --a "$CANDIDATE" --b "$opp" \
        --games "$GAMES" --jobs 100 2>&1)
    rate=$(echo "$result" | grep "综合得分率" | grep -o '[0-9.]*' | head -1)
    detail=$(echo "$result" | grep "执红\|执蓝" | sed 's/^    //')

    echo "  vs $name: 综合 $rate  |  $detail"

    # 用最差颜色作为该配对的代表
    red_rate=$(echo "$result" | grep "执红" | grep -o '0\.[0-9]*' | head -1)
    blue_rate=$(echo "$result" | grep "执蓝" | grep -o '0\.[0-9]*' | head -1)
    for r in $red_rate $blue_rate; do
        r_float=$(python3 -c "print($r)" 2>/dev/null || echo "$r")
        if python3 -c "exit(0 if $r_float < $WORST else 1)" 2>/dev/null; then
            WORST=$r_float
            WORST_NAME="$name"
        fi
    done
    TOTAL=$(python3 -c "print($TOTAL + $rate)" 2>/dev/null || echo "$TOTAL")
    COUNT=$((COUNT + 1))
done

if [ "$COUNT" -gt 0 ]; then
    MEAN=$(python3 -c "print(f'{$TOTAL / $COUNT:.4f}')")
    echo
    echo "════════ 汇总 ════════"
    echo "  平均综合: $MEAN"
    echo "  最差颜色: $WORST ($WORST_NAME)"
fi

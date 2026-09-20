#!/usr/bin/env bash
# tools/weight_search.sh —— 权重坐标下降自动调优（风格池评测防过拟合）
#
# 每轮：固定其他权重，对单个权重扫描 3 个值，选最差象限最高的
# 一轮结束后自动进入下一个权重。所有评测使用同一个对手池（4 风格 + pool/ 历史）。
#
# 用法: bash tools/weight_search.sh [轮数=2]
set -euo pipefail
cd "$(dirname "$0")/.."

ROUNDS=${1:-2}
GAMES=${GAMES:-100}
RESULTS_FILE=build/weight_search_results.txt

# 对手池
OPPONENTS=(
    build/opponents/baseline_ai.so
    build/opponents/hunter_ai.so
    build/opponents/det_ai_a.so
    build/sitter.so
)

# 可调权重名（env 后缀）和当前值
declare -A W_NAMES=(
    [THREAT]=0.40
    [ZONE]=0.80
    [DIST]=0.40
)

# 扫描倍率：当前值 × {0.5, 1.0, 2.0}
MULTIPLIERS=(0.5 1.0 2.0)

# ── 工具函数 ──

# 评一个权重配置，输出 worst-quadrant（最低的颜色胜率）
eval_config() {
    local tag="$1"; shift
    local env_args=()
    for arg in "$@"; do env_args+=("ST_W_${arg}"); done

    local worst=1.01
    local worst_opp=""
    for opp_file in "${OPPONENTS[@]}"; do
        local opp_name
        opp_name=$(basename "$opp_file" .so | sed 's/_ai$//')
        local result
        result=$(env "${env_args[@]}" python3 tools/league.py \
            --a build/my_ai.so --b "$opp_file" --games "$GAMES" --jobs 100 2>&1)
        local rate
        rate=$(echo "$result" | grep "综合得分率" | grep -oE '0\.[0-9]+' | head -1)
        [ -z "$rate" ] && rate=0
        # 取两色最低
        local br br_f
        br=$(echo "$result" | grep "执红" | grep -oP '得分率 \K0\.[0-9]+' || echo 0)
        br_f=$(python3 -c "print($br)" 2>/dev/null || echo 0)
        local bb
        bb=$(echo "$result" | grep "执蓝" | grep -oP '得分率 \K0\.[0-9]+' || echo 0)
        local bb_f
        bb_f=$(python3 -c "print($bb)" 2>/dev/null || echo 0)
        local min_color
        min_f=$(python3 -c "print(min($br_f, $bb_f))" 2>/dev/null || echo 0)

        echo "  $tag vs $opp_name: $rate ($br/$bb)"

        if python3 -c "exit(0 if $min_f < $worst else 1)" 2>/dev/null; then
            worst=$min_f
            worst_opp=$opp_name
        fi
    done
    echo "$worst"  # 返回值 = worst quadrant
}

# ── 主循环 ──
echo "════════ 权重自动调优 ════════"
echo "轮数: $ROUNDS  局数/配对: $GAMES  对手池: ${#OPPONENTS[@]}"
echo

> "$RESULTS_FILE"

for round in $(seq 1 "$ROUNDS"); do
    echo "════════ 第 $round / $ROUNDS 轮 ════════"

    for wname in THREAT ZONE DIST; do
        current=${W_NAMES[$wname]}
        echo
        echo "--- 扫描 $wname (当前 $current) ---"

        local_best="-1"
        local_best_val="$current"

        for mult in "${MULTIPLIERS[@]}"; do
            val=$(python3 -c "print(round($current * $mult, 2))")
            # 跳过与当前相同的值（除非就是当前轮）
            worst=$(eval_config "$wname=$val" "${wname}=${val}")
            echo "  → $wname=$val: worst-quadrant $worst"

            if python3 -c "exit(0 if $worst > $local_best else 1)" 2>/dev/null; then
                local_best=$worst
                local_best_val=$val
            fi
        done

        # 更新默认值
        W_NAMES[$wname]=$local_best_val
        echo "$round $wname $local_best_val $local_best" >> "$RESULTS_FILE"
        echo "  ★ 选定 $wname=$local_best_val (worst $local_best)"
    done
done

echo
echo "════════ 最终结果 ════════"
for wname in THREAT ZONE DIST; do
    echo "  $wname = ${W_NAMES[$wname]}"
done
echo
echo "详细日志: $RESULTS_FILE"
echo "用 ST_W_$(printf '%s' THREAT)=... 等环境变量测试新默认值"

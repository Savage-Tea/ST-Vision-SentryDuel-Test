#!/usr/bin/env bash
# tools/iterate.sh —— 多轮自对弈策略迭代
#
# 每轮：用当前策略生成自对弈数据 → 训练 → 导出权重 → 重建 → 存档一个 .so
# 存档是为了能回答"新一轮是否真的强于旧一轮"——这是策略迭代唯一的验收标准，
# 不看训练损失。
#
# 【重要】网络只在 MCTS 根部当先验，所以评测必须带 ST_MCTS=1；
# 默认策略是 phase① 的穷举搜索，那条路径根本不读网络。
#
# 环境变量：
#   ROUNDS=3  GAMES=4000  THREADS=160  BUDGET_MS=20  EPOCHS=20
set -euo pipefail
cd "$(dirname "$0")/.."

ROUNDS=${ROUNDS:-3}
GAMES=${GAMES:-4000}
THREADS=${THREADS:-160}
BUDGET_MS=${BUDGET_MS:-20}
EPOCHS=${EPOCHS:-20}

mkdir -p build/rounds

for r in $(seq 1 "$ROUNDS"); do
    echo
    echo "################ 第 $r / $ROUNDS 轮 ################"
    GAMES=$GAMES THREADS=$THREADS BUDGET_MS=$BUDGET_MS EPOCHS=$EPOCHS \
        DATA=build/selfplay_out/round$r.bin bash tools/pipeline.sh

    cp build/my_ai.so "build/rounds/my_ai_r$r.so"
    cp brain/net_weights.h "build/rounds/net_weights_r$r.h"
    echo "存档 build/rounds/my_ai_r$r.so"
done

echo
echo "################ 回合间对比（带 ST_MCTS=1）################"
for r in $(seq 2 "$ROUNDS"); do
    prev=$((r - 1))
    echo "--- r$r vs r$prev ---"
    ST_MCTS=1 python3 tools/league.py --a "build/rounds/my_ai_r$r.so" \
        --b "build/rounds/my_ai_r$prev.so" --games 200 --jobs "$THREADS" \
        2>&1 | grep -E "胜 |综合得分率" || true
done

echo
echo "################ 对官方对手（带 ST_MCTS=1）################"
for opp in baseline hunter; do
    echo "--- r$ROUNDS vs $opp ---"
    ST_MCTS=1 python3 tools/league.py --a "build/rounds/my_ai_r$ROUNDS.so" \
        --b "build/opponents/${opp}_ai.so" --games 200 --jobs "$THREADS" \
        2>&1 | grep -E "胜 |综合得分率" || true
done

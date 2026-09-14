#!/usr/bin/env bash
# tools/pipeline.sh —— 自对弈策略迭代的一轮：生成 → 训练 → 导出 → 重建 → 校验
#
# 目标函数是**自对弈强度**，评测用对手池（历史版本 + baseline + hunter），
# 不是对某两个模型特攻。
#
# 环境变量（都有默认值）:
#   GAMES=2000     本轮自对弈局数
#   THREADS=nproc  并行线程数
#   BUDGET_MS=20   每步 MCTS 时间预算
#   EPOCHS=15      训练轮数
#   DATA=...       样本输出路径
set -euo pipefail
cd "$(dirname "$0")/.."

GAMES=${GAMES:-2000}
THREADS=${THREADS:-$(nproc)}
BUDGET_MS=${BUDGET_MS:-20}
EPOCHS=${EPOCHS:-15}
DATA=${DATA:-build/selfplay_out/round1.bin}

build() { # 构建并**真的检查产物**——之前 grep 加 || true 把编译失败吞掉了
    local log
    log=$(make ai selfplay 2>&1) || { echo "$log" | tail -20; echo "构建失败"; exit 1; }
    echo "$log" | grep -E "error|warning" || true
    [ -f build/my_ai.so ] && [ -f build/selfplay ] || { echo "构建未产出预期文件"; exit 1; }
}

echo "=== [1/5] 构建 ==="
build

echo
echo "=== [2/5] 自对弈生成: $GAMES 局 / $THREADS 线程 / 每步 ${BUDGET_MS}ms ==="
mkdir -p "$(dirname "$DATA")"
./build/selfplay --games "$GAMES" --threads "$THREADS" --budget-ms "$BUDGET_MS" \
                 --out "$DATA"

echo
echo "=== [3/5] 训练 $EPOCHS 轮 ==="
python3 tools/train_mlp.py --data "$DATA" --epochs "$EPOCHS" --out build/net.npz

echo
echo "=== [4/5] 用训练出的权重重建 ==="
build

echo
echo "=== [5/5] 校验网络已被加载 ==="
./build/selfplay --games 1 --threads 1 --budget-ms 5 --out /tmp/netcheck.bin \
    | grep -E "^网络:"

echo
echo "一轮完成。权重: brain/net_weights.h   样本: $DATA"
echo "下一步评测（用对手池，不是特攻）:"
echo "  python3 tools/league.py --a build/my_ai.so --b build/opponents/hunter_ai.so --games 400"

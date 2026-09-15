#!/usr/bin/env bash
# tools/iterate_ppo.sh —— 阶段③ 的联盟自对弈迭代
#
# 每轮：用当前策略生成轨迹 → PPO 更新 → 导出权重 → **重编** → 存档
#
# 【为什么每轮都要重编】
# 权重是编译进 selfplay_ppo 的（brain/policy_weights.h）。这是刻意的：
# 平台上传的是一份 .so，没有运行时的权重文件可以加载。所以"用新策略采样"
# 必然要重编一次，约 1 分钟，相对生成本身可以忽略。
#
# 【对手池】
# 双方都是当前策略。**不把 baseline/hunter 放进训练**——那是特攻，用户明确
# 否过。这两个模型只用于评测。
#
# 【为什么每轮换 seed】
# 这个游戏完全确定性，同一套动作必然重现同一局。采样虽然带温度，但每轮用
# 同一个 seed 会让各轮的数据高度重叠。按轮次偏移 seed 保证每轮见到新局面。
#
# 环境变量：ROUNDS=5 GAMES=4000 THREADS=160 EPOCHS=4 TEMPERATURE=1.0
set -euo pipefail
cd "$(dirname "$0")/.."

ROUNDS=${ROUNDS:-5}
GAMES=${GAMES:-4000}
THREADS=${THREADS:-$(nproc)}
EPOCHS=${EPOCHS:-4}
TEMPERATURE=${TEMPERATURE:-1.0}
# 设备与 minibatch。cluster34 有 V100，训练从分钟级降到秒级；
# batch 必须保持小值——实测 batch=2048 时每轮只切出 4 次参数更新，
# 近似 KL 只有 0.0009、clip 比例恒为 0，策略被冻住。
DEVICE=${DEVICE:-cpu}
BATCH=${BATCH:-128}

OUT=build/ppo
mkdir -p "$OUT"

# 从**全新随机权重**开始，而不是沿用冒烟留下的检查点——那样第一轮的
# 数据来源才可复现。
if [ ! -f "$OUT/net_r0.npz" ]; then
    echo "=== 初始化：全新随机策略 ==="
    python3 tools/policy_net.py --seed 0 --out brain/policy_weights.h
fi
make selfplay-ppo >/dev/null

for r in $(seq 1 "$ROUNDS"); do
    echo
    echo "################ 第 $r / $ROUNDS 轮 ################"

    echo "--- [1/4] 用当前策略生成 $GAMES 局轨迹 ---"
    # 对手池：50% 概率让蓝方用近 5 轮中的随机历史版本。
    # 打破"永远和当前自己打"的循环退化（实测 r4→r5 出现回退）。
    OPP_ARG=""
    if [ "$r" -gt 1 ] && [ $((RANDOM % 2)) -eq 0 ]; then
        lo=$((r - 5)); [ "$lo" -lt 1 ] && lo=1
        hi=$((r - 1))
        k=$((lo + RANDOM % (hi - lo + 1)))
        OPP_ARG="--opp-weights $OUT/policy_w_r$k.bin"
        echo "    对手: 第 $k 轮的历史版本（池）"
    else
        echo "    对手: 当前策略"
    fi
    ./build/selfplay_ppo --games "$GAMES" --threads "$THREADS" \
        --temperature "$TEMPERATURE" --seed "$((r * 1000))" \
        $OPP_ARG --out "$OUT/round$r.bin"

    echo "--- [2/4] PPO 更新 $EPOCHS 轮 ---"
    python3 tools/train_ppo.py --data "$OUT/round$r.bin" --epochs "$EPOCHS" \
        --batch "$BATCH" --device "$DEVICE" --threads 8 --out "$OUT/net_r$r.npz"

    echo "--- [3/4] 用新权重重编（下一轮采样要用它）---"
    make selfplay-ppo >/dev/null
    cp brain/policy_weights.h "$OUT/policy_weights_r$r.h"
    cp brain/policy_weights.bin "$OUT/policy_w_r$r.bin"

    echo "--- [4/4] 存档 ---"
    # 存样本量而不只是局数：有效步数才是训练的规模
    python3 - "$OUT/round$r.bin" <<'PY'
import sys
sys.path.insert(0, ".")
from tools import sdpp
t = sdpp.read(sys.argv[1])
steps = sum(len(s.obs) for s in t.seqs)
nz = sum(int((s.reward != 0).sum()) for s in t.seqs)
print(f"    序列 {len(t.seqs)}  步 {steps:,}  有奖励的步 {nz:,} ({nz*100.0/steps:.2f}%)")
PY
    echo "    存档 $OUT/policy_weights_r$r.h"
done

echo
echo "################ 完成 ################"
ls -la "$OUT"/policy_weights_r*.h
echo
echo "下一步（需要 M4 的 ST_POLICY=1 部署路径）：分色评测，与阶段① 同口径对比。"

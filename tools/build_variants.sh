#!/usr/bin/env bash
# tools/build_variants.sh —— 把若干"风格"烘进独立的 .so，放进风格池
#
# 为什么要独立 .so：引擎把红蓝两个 AI dlopen 在同一个进程里，ST_W_* 环境变量
# 是进程级共享的，没法只改对手一侧。所以做对手只能编译期定值。
#
# 为什么需要风格变体：目前的池子（baseline/hunter/det_ai_*）我们已经全部接近
# 100%，没有区分度，验收不了任何改动。同源但权重差异大的变体至少能测出
# "策略对风格扰动是否稳健"。
#
# 用法: bash tools/build_variants.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="agent/act.cpp brain/eval.cpp brain/actions.cpp brain/search.cpp \
     brain/mcts.cpp brain/belief_state.cpp brain/net.cpp \
     brain/policy_net.cpp brain/value_net.cpp brain/ab_search.cpp \
     obs/encode.cpp obs/encode_v3.cpp \
     sim/view_mirror.cpp sim/rules.cpp sim/belief.cpp"

mkdir -p pool
build() {  # build <名字> <额外的 -D 参数...>
    local name="$1"; shift
    echo "  构建 pool/${name}.so  $*"
    # shellcheck disable=SC2086
    if ! g++ -std=c++17 -O2 -fPIC -Wall -Wextra -I. \
             -I../sentry-duel/engine/include \
             "$@" $SRC -shared -Wl,-z,lazy -Wl,--allow-shlib-undefined \
             -o "pool/${name}.so" 2>"build/${name}.build.log"; then
        echo "    ❌ 编译失败，见 build/${name}.build.log" >&2
        tail -20 "build/${name}.build.log" >&2
        exit 1
    fi
    [ -s "pool/${name}.so" ] || { echo "    ❌ 未产出 .so" >&2; exit 1; }
}

echo "构建风格变体（同源、烘死不同权重，用于风格池）..."
# 只看权重这一个维度就够：它在这一版 AI 里是决策的全部（叶评估是线性的）
build var_turtle   -DSD_W_DANGER=16.0 -DSD_W_ZONE=0.4    # 极度保守，不进危险区
build var_rush     -DSD_W_DANGER=0.5  -DSD_W_ZONE=1.6    # 极度激进，抢区优先
build var_sniper   -DSD_W_DANGER=8.0  -DSD_W_THREAT=1.2 -DSD_W_READY=1.0  # 重视击杀
build var_camper   -DSD_W_ZONE=1.6    -DSD_W_DIST=0.1    # 死守得分区
build var_danger8  -DSD_W_DANGER=8.0                     # 只调死亡定价（之前的候选值）

# 当前部署版本也放进池子（作为"我们自己"这个风格）
cp -f build/my_ai.so pool/var_current.so
echo "  pool/var_current.so  ← build/my_ai.so 的副本"
ls -la pool/*.so | awk '{print "    ", $9, $5}'

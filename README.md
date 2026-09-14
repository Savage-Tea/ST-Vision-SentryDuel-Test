# ST-Vision-SentryDuel-Test

哨兵大战（7×7 回合制 1v1 AI 对战）的**自研 AI 工程**。

与赛事仓库 `sentry-duel` 的关系：**只读引用**。引擎源码与选手 ABI 头从
`../sentry-duel` 读取，所有构建产物落在本仓库的 `build/`。
`../sentry-duel` 是 fork（不可推送），本仓库永不写入它。

## 路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| ⓪ | `sim/` 前向模型 + 差分测试 | **规则核心已完成**（对局层待补） |
| ① | 简单搜索 + policy model | **可跑可上传，但未达入围赛口径** |
| ② | MCTS | 未开始 |
| ③ | RNN + PPO | 未开始 |

## 阶段① 现状（必读）

已实现：`sim/` 前向模型 + 本回合搜索（我方 0-3 行动穷举 → 对手 0-3 行动
minimax 回应 → 评估），执行层带时间预算、分叉重规划与兜底。

**实测（各 400 局，红蓝各半）：**

| 对手 | 胜 | 负 | 综合得分率 | 平均净胜分 |
|---|---|---|---|---|
| baseline | 210 | 190 | 0.525 | −4.38 |
| hunter | 188 | 212 | **0.470** | −3.32 |

入围赛要求对两者**胜场都严格大于对手**，所以**目前还不达标**。

### 调参已到平台期，瓶颈是结构

对 `w_danger / w_threat / w_dist` 做过网格扫描（`tools/sweep_pool.py`），
按"对池中最差对手的得分率"取最优，最好也只到 **0.487**。继续调不会有用。

过程中踩到并记录下来的两个坑：

1. **只对单一对手调参会严重过拟合。** 找到一个配置对 baseline 是 **400 胜 0 负
   (100%)**，对 hunter 却是 **17 胜 383 负 (4.3%)**。所以评测的目标函数必须是
   "对池中每个对手的最小得分率"，不能是单一对手的平均值。
2. **官方对手是随机的。** `baseline_ai.cpp:25` 与 `hunter_ai.cpp:25` 都用
   `steady_clock::now()` 给 xorshift 播种，每次运行都不同——所以胜率是采样
   估计，局数不够时不要相信两位小数。引擎与我们的 AI 本身都是确定性的
   （`tests/test_invariants.py` 有断言）。

### 下一刀应该切哪里

搜索的敌方位置是**单点信念**（最后已知位置），而 hunter 每 3 回合扫描一次、
从 3 格外开火。以下两件事按收益排序：

1. **敌方位置用可达集信念而非单点**（rl 管线的 `obs_builder.h` 有现成做法）。
   单点信念会让"危险"估计完全失真——对手从视野外接近时我们以为自己是安全的。
2. **给 SCAN 定价。** 现在 SCAN 靠 `agent/act.cpp` 里的一条显式规则触发，因为
   在单点信念下搜索**无法**给信息定价（模拟中扫描不改变敌方位置，于是它在评估
   函数眼里是纯亏）。信念建模之后这条规则才能被真正的信息价值取代。

阶段③ 的 RNN 能从历史维持信念，正是为了解决这一点 —— 但在那之前，
阶段①/② 用显式的可达集也能拿到大部分收益。

## 为什么需要 `sim/`

引擎暴露给选手的 `move/turn/fire/scan` **直接改写真实状态、没有撤销**，
因此任何搜索都无法通过调用它们来实现。`sim/` 是一个纯函数前向模型：
给定状态与行动返回新状态，可复制、可回溯。

它是三个阶段共用的支点：

- 阶段① 在本回合 3 个行动的序列空间里穷举
- 阶段② MCTS 的推演内核（同一个 `apply`）
- 阶段③ 训练环境建在 `sim` 上，比 dlopen + subprocess 快几个数量级，
  且训练与部署共用同一份观测构建代码，从构造上杜绝 train/serve 偏差

## 差分测试

`sim` 最大的风险是**与引擎静默不一致**——那会让搜索基于错误的世界模型
悄悄变蠢。所以 `sim` 的每一行都必须被引擎验证过。

`tests/difftest_rules.cpp` 把引擎的规则函数当作 **oracle**：同一个输入分别
喂给引擎与 `sim`，逐字段比对。这比"跑几局随机对局看看"强得多，因为随机
对局几乎撞不到火力通道遮挡、贴边、CD 非零时开火这类分支。

覆盖范围（8 种障碍布局 × 全位置 × 全朝向 × 全 CD 组合）：

| 项目 | 说明 |
|---|---|
| `apply_action` | move / turn（含非法字符、转向当前朝向）/ fire / scan |
| `can_see` | T 形视野 + 逐条视线遮挡 |
| `fire_hit` | 前向 3×3 + 同通道障碍遮挡；命中后的落点与双方状态 |
| `end_side_turn` | 占点计分（含双方同时占点） |
| `end_round` | 双方 CD 递减与回合推进 |
| 初始状态 | 出生点、朝向、障碍、得分区 |

```bash
make test          # 全量，约 790 万次比对
make test-quick    # 抽样，CI / 快速回归
```

**当前结果：7,893,731 次比对，0 处不一致。**

测试本身经过**变异测试**验证有效：往 `sim` 注入 8 个不同的已知 bug
（火力/雷达冷却、遮挡步数、视野偏移、命中后朝向、移动忽略障碍、漏计分、
击杀分值），全部被抓到。一个抓不到 bug 的测试等于没有测试。

### 尚未覆盖

差分测试目前只覆盖**规则核心**。以下属于对局层（引擎 `Match::do_action`
/ `run`），尚未比对：

- 每回合 3 次消耗行动的上限，以及失败不消耗额度
- 开局 / 复活后的免费转向（出生点窗口、离开即失效）
- 行动后的实时视野刷新与 SCAN 临时视野
- 传给选手的视图构造（蓝方 180° 镜像、情报遮蔽、敌方 CD 置 -1）
- 加时赛与胜负判定

这是下一步 `sim/` 的 match 层与 Part B 差分测试的工作。

## 构建与评测

```bash
make engine           # CMake out-of-source 构建引擎到 build/engine
make opponents        # 从仓库源码编译 baseline / hunter
make opponents-det    # 编译引擎自带的确定性测试 AI（不变量测试用）
make ai               # 构建我们的 build/my_ai.so
make all              # 以上

make test             # sim 与引擎规则核心的差分测试（约 790 万次比对）
make test-quick       # 同上，抽样
python3 tests/test_invariants.py   # 确定性与颜色对称回归测试

python3 tools/league.py --a build/my_ai.so --b build/opponents/hunter_ai.so --games 400
python3 tools/sweep_pool.py --games 150        # 对对手池扫描权重
python3 tools/pack.py --verify                 # 打包并模拟平台编译
```

依赖：`g++`（C++17）、`cmake`、`make`、`python3`（仅标准库）。

若引擎源码位置不同，覆盖 `SENTRY_DUEL_ROOT`：

```bash
make SENTRY_DUEL_ROOT=/path/to/sentry-duel all
```

### 权重调参

`Weights` 的编译期默认值写在 `brain/eval.h`。离线调参可用环境变量临时覆盖
（平台评测时环境里没有这些变量，线上行为完全由默认值决定）：

```bash
ST_W_DANGER=0.5 ST_W_THREAT=1.5 python3 tools/league.py ...
ST_BUDGET_MS=1000 ST_DEBUG=1 build/engine/runner ...   # 打印每步决策
```

## 目录

```
sim/         前向模型（纯函数，只依赖选手 ABI 头 sentry_duel.h）
brain/       决策器：eval.cpp 评估函数 + search.cpp 本回合搜索
agent/       act() 入口：状态组装 → 搜索 → 真机执行 → 分叉重规划 → 兜底
tests/       sim 差分测试 + AI 不变量回归测试
tools/       评测（league / sweep / sweep_pool）、打包（pack）
build/       全部产物（gitignore）
```

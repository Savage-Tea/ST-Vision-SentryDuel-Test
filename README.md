# ST-Vision-SentryDuel-Test

哨兵大战（7×7 回合制 1v1 AI 对战）的**自研 AI 工程**。

与赛事仓库 `sentry-duel` 的关系：**只读引用**。引擎源码与选手 ABI 头从
`../sentry-duel` 读取，所有构建产物落在本仓库的 `build/`。
`../sentry-duel` 是 fork（不可推送），本仓库永不写入它。

## 路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| ⓪ | `sim/` 前向模型 + 差分测试 | **规则核心已完成**（match 层待补） |
| ① | 简单搜索 + policy model | 未开始 |
| ② | MCTS | 未开始 |
| ③ | RNN + PPO | 未开始 |

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

## 构建

```bash
make engine      # CMake out-of-source 构建引擎到 build/engine
make opponents   # 从仓库源码编译 baseline / hunter 到 build/opponents
make all         # 以上两者
```

依赖：`g++`（C++17）、`cmake`、`make`。

若引擎源码位置不同，覆盖 `SENTRY_DUEL_ROOT`：

```bash
make SENTRY_DUEL_ROOT=/path/to/sentry-duel all
```

## 目录

```
sim/         前向模型（纯函数，只依赖选手 ABI 头 sentry_duel.h）
tests/       差分测试
brain/       决策器：search/ mcts/ nn/          （待建）
agent/       act() 入口：状态组装 → 决策 → 真机执行 → 分叉重规划 → 兜底
obs/         观测/特征构建（训练与部署共用）
tools/       评测与脚本
build/       全部产物（gitignore）
```

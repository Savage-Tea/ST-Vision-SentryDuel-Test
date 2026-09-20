# ST-Vision-SentryDuel-Test
#
# 引擎源码从 SENTRY_DUEL_ROOT 只读引用，所有产物落在本目录 build/。
# 绝不写入 ../sentry-duel（那是 fork，只读）。

SENTRY_DUEL_ROOT ?= ../sentry-duel

BUILD        := build
ENGINE_INC   := $(SENTRY_DUEL_ROOT)/engine/include
ENGINE_BUILD := $(BUILD)/engine

CXX      ?= g++
CXXFLAGS := -std=c++17 -O2 -fPIC -Wall -Wextra -MMD -MP -I. -I$(ENGINE_INC)

ENGINE_LINK := -L$(ENGINE_BUILD) -lsentry_duel_engine -Wl,-rpath,'$$ORIGIN/engine' -ldl

# 所有头文件都当作先决条件，**刻意不依赖 -MMD 的自动依赖**。
#
# 为什么不靠 -MMD：下面的规则是**一次 g++ 调用编译多个 .cpp**，而 GCC 在
# 这种模式下只会写出其中一个源文件的依赖（实测只剩最后一个），于是
# obs/brain/sim 下任何头文件的改动都不会触发重编。本项目全是 header+source
# 结构，这个洞意味着：改了观测布局或评估权重，make 会安静地拿着旧产物去跑
# 测试、甚至把过期的 .so 传上平台。实测确认过：touch obs/encode_v3.h 后
# make 不重编。
#
# 本项目编译只要几秒，宁可多编一次，也不要用旧二进制。
HDRS := $(shell find . -name '*.h' -not -path './build/*' 2>/dev/null)
# Makefile 自身也算先决条件：改了 AI_SRC 这类变量时必须重链，否则 make 认为
# "产物比源码新"直接跳过——实测踩过：给 AI_SRC 补了两个 .cpp，编译没发生，
# .so 里缺符号，直到 dlopen 才炸。
HDRS += Makefile

.PHONY: all engine opponents ai test test-quick clean selfplay-ppo test-policy test-view test-obs-v3

all: engine opponents ai

# —— 引擎（CMake out-of-source：源码在仓库，产物在我们这里）——
engine:
	cmake -S $(SENTRY_DUEL_ROOT)/engine -B $(ENGINE_BUILD) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(ENGINE_BUILD) --parallel

# —— 对手：从仓库源码编译到我们自己的 build/ ——
OPPONENTS := $(BUILD)/opponents/baseline_ai.so $(BUILD)/opponents/hunter_ai.so

opponents: $(OPPONENTS)

$(BUILD)/opponents/%.so: $(SENTRY_DUEL_ROOT)/ai/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -shared -Wl,-z,lazy -Wl,--allow-shlib-undefined $< -o $@

# 引擎自带的确定性测试 AI —— 不变量回归测试的对手（官方 baseline/hunter 是随机的，
# 断言没法严格）
DET_OPPONENTS := $(BUILD)/opponents/det_ai_a.so $(BUILD)/opponents/det_ai_b.so

opponents-det: $(DET_OPPONENTS)

$(BUILD)/opponents/det_ai_%.so: $(SENTRY_DUEL_ROOT)/engine/tests/det_ai_%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -shared -Wl,-z,lazy -Wl,--allow-shlib-undefined $< -o $@

# —— 我们的 AI（阶段①：搜索 + policy model）——
AI_SRC := agent/act.cpp brain/eval.cpp brain/actions.cpp brain/search.cpp \
          brain/mcts.cpp brain/belief_state.cpp brain/net.cpp \
          brain/policy_net.cpp brain/value_net.cpp brain/ab_search.cpp \
          obs/encode.cpp obs/encode_v3.cpp \
          sim/view_mirror.cpp sim/rules.cpp sim/belief.cpp

$(BUILD)/my_ai.so: $(AI_SRC) $(HDRS) | engine
	$(CXX) $(CXXFLAGS) -shared -Wl,-z,lazy -Wl,--allow-shlib-undefined \
	    $(AI_SRC) -o $@ $(ENGINE_LINK)

ai: $(BUILD)/my_ai.so

# —— 差分测试：sim 必须与引擎规则核心逐字段一致 ——
$(BUILD)/difftest_rules: tests/difftest_rules.cpp sim/rules.cpp $(HDRS) | engine
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@ $(ENGINE_LINK)

test: $(BUILD)/difftest_rules
	$(BUILD)/difftest_rules

test-quick: $(BUILD)/difftest_rules
	$(BUILD)/difftest_rules --quick

# —— 信念单元测试 ——
$(BUILD)/test_belief: tests/test_belief.cpp sim/belief.cpp sim/rules.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@

test-belief: $(BUILD)/test_belief
	$(BUILD)/test_belief

# —— AI 不变量回归（确定性 + 颜色对称）——
test-ai: $(BUILD)/my_ai.so opponents-det
	python3 tests/test_invariants.py

# —— 坐标框架等变性（自对弈训练的正确性前提）——
$(BUILD)/test_frame: tests/test_frame.cpp sim/rules.cpp sim/belief.cpp brain/actions.cpp brain/eval.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@

test-frame: $(BUILD)/test_frame
	$(BUILD)/test_frame

# —— 阶段③ 循环策略网络：GRU 前向的差分测试（C++ 侧）——
# 权重由 tools/difftest_policy.py 在编译**之前**导出，所以这里刻意**不**把
# brain/policy_weights.h 写成先决条件：那样 make 会认为产物已是最新而不重编，
# 拿到的就是上一轮导出的权重。脚本每次都会 rm 掉二进制再编。
$(BUILD)/difftest_policy: tests/difftest_policy.cpp brain/policy_net.cpp $(HDRS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@

# Python 驱动负责：导出权重 → 生成用例 → 编译 → 比对
test-policy:
	python3 tools/difftest_policy.py

# —— obs v3 编码：布局 + 180° 旋转等变 ——
$(BUILD)/test_obs_v3: tests/test_obs_v3.cpp obs/encode_v3.cpp sim/rules.cpp $(HDRS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@

test-obs-v3: $(BUILD)/test_obs_v3
	$(BUILD)/test_obs_v3

# —— 对局层差分：选手视图（Intel）必须与引擎一致 ——
# 这是自对弈训练的前置门槛：sim 里的 last_known_pos 是真实位置，
# 而引擎给选手的是记忆。弄错就是信息泄漏，而且不会报错。
$(BUILD)/difftest_view: tests/difftest_view.cpp sim/view_mirror.cpp sim/rules.cpp $(HDRS) | engine
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@ $(ENGINE_LINK)

test-view: $(BUILD)/difftest_view
	$(BUILD)/difftest_view

clean:
	rm -rf $(BUILD)

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)

# —— 自对弈样本生成器（纯 sim，不依赖引擎）——
SELFPLAY_SRC := selfplay/selfplay.cpp brain/eval.cpp brain/actions.cpp \
                brain/search.cpp brain/mcts.cpp brain/belief_state.cpp \
                obs/encode.cpp brain/net.cpp sim/rules.cpp sim/belief.cpp

$(BUILD)/selfplay: $(SELFPLAY_SRC) $(HDRS)
	$(CXX) $(CXXFLAGS) $(SELFPLAY_SRC) -o $@ -pthread

selfplay: $(BUILD)/selfplay

# —— 阶段③ 策略自对弈（PPO 轨迹生成）——
# 注意这里**不要** brain/mcts.cpp 与 brain/belief_state.cpp：
# 阶段③ 的观测不含手工信念，动作也由策略网络直出，不经过搜索。
# sim/belief.cpp 是被 brain/actions.cpp 的 collect_candidates 拖进来的（它收
# 一个 Belief* 参数），虽然阶段③ 的观测不含手工信念、驱动也不调那个函数。
# 这里要的是 apply_step —— 它已经把额度与免费转向语义复刻好了，重写一遍不划算。
SELFPLAY_PPO_SRC := selfplay/selfplay_ppo.cpp brain/policy_net.cpp \
                    brain/actions.cpp brain/eval.cpp \
                    obs/encode_v3.cpp sim/view_mirror.cpp \
                    sim/rules.cpp sim/belief.cpp

$(BUILD)/selfplay_ppo: $(SELFPLAY_PPO_SRC) $(HDRS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(SELFPLAY_PPO_SRC) -o $@ -pthread

selfplay-ppo: $(BUILD)/selfplay_ppo

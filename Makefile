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

.PHONY: all engine opponents ai test test-quick clean

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
          brain/mcts.cpp brain/belief_state.cpp brain/net.cpp sim/rules.cpp sim/belief.cpp

$(BUILD)/my_ai.so: $(AI_SRC) | engine
	$(CXX) $(CXXFLAGS) -shared -Wl,-z,lazy -Wl,--allow-shlib-undefined \
	    $(AI_SRC) -o $@ $(ENGINE_LINK)

ai: $(BUILD)/my_ai.so

# —— 差分测试：sim 必须与引擎规则核心逐字段一致 ——
$(BUILD)/difftest_rules: tests/difftest_rules.cpp sim/rules.cpp | engine
	$(CXX) $(CXXFLAGS) $^ -o $@ $(ENGINE_LINK)

test: $(BUILD)/difftest_rules
	$(BUILD)/difftest_rules

test-quick: $(BUILD)/difftest_rules
	$(BUILD)/difftest_rules --quick

# —— 信念单元测试 ——
$(BUILD)/test_belief: tests/test_belief.cpp sim/belief.cpp sim/rules.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

test-belief: $(BUILD)/test_belief
	$(BUILD)/test_belief

# —— AI 不变量回归（确定性 + 颜色对称）——
test-ai: $(BUILD)/my_ai.so opponents-det
	python3 tests/test_invariants.py

# —— 坐标框架等变性（自对弈训练的正确性前提）——
$(BUILD)/test_frame: tests/test_frame.cpp sim/rules.cpp sim/belief.cpp brain/actions.cpp brain/eval.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

test-frame: $(BUILD)/test_frame
	$(BUILD)/test_frame

clean:
	rm -rf $(BUILD)

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)

# —— 自对弈样本生成器（纯 sim，不依赖引擎）——
SELFPLAY_SRC := selfplay/selfplay.cpp brain/eval.cpp brain/actions.cpp \
                brain/search.cpp brain/mcts.cpp brain/belief_state.cpp \
                obs/encode.cpp brain/net.cpp sim/rules.cpp sim/belief.cpp

$(BUILD)/selfplay: $(SELFPLAY_SRC)
	$(CXX) $(CXXFLAGS) $(SELFPLAY_SRC) -o $@ -pthread

selfplay: $(BUILD)/selfplay

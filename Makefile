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

.PHONY: all engine opponents test test-quick clean

all: engine opponents

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

# —— 差分测试：sim 必须与引擎规则核心逐字段一致 ——
$(BUILD)/difftest_rules: tests/difftest_rules.cpp sim/rules.cpp | engine
	$(CXX) $(CXXFLAGS) $^ -o $@ $(ENGINE_LINK)

test: $(BUILD)/difftest_rules
	$(BUILD)/difftest_rules

test-quick: $(BUILD)/difftest_rules
	$(BUILD)/difftest_rules --quick

clean:
	rm -rf $(BUILD)

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)

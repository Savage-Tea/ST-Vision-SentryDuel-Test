// tests/difftest_policy.cpp —— GRU 前向的差分测试（C++ 侧）
//
// 手写 GRU 与 torch 的 GRU 必须给出同样的结果。差异不会让程序崩溃，
// 只会让"在 C++ 里跑的策略"和"在 torch 里训练的策略"悄悄变成两个不同的东西
// ——这正是本项目已经踩过两次的那类坑（turn 参数被镜像两次、训练/部署 obs 漂移）。
//
// 做法：Python 侧（tools/difftest_policy.py）用固定种子生成权重与随机输入序列，
// 用自己的前向算出参考结果；这里读同一份输入、跑 C++ 前向、写回结果，
// 再由 Python 逐元素比对。**权重每次都由 Python 重新导出**，所以不存在
// "头文件是旧的"这种情况。
//
// 用法：difftest_policy <cases.bin> <out.bin>

#include "brain/policy_net.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr char kCasesMagic[4] = {'S', 'D', 'P', 'T'};
constexpr char kOutMagic[4] = {'S', 'D', 'P', 'O'};
constexpr uint32_t kVersion = 1;

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "difftest_policy: %s\n", msg.c_str());
    std::exit(1);
}

template <typename T>
void read_exact(std::ifstream& f, T* dst, std::size_t count, const char* what) {
    f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(sizeof(T) * count));
    if (!f) die(std::string("读取失败: ") + what);
}

template <typename T>
void write_exact(std::ofstream& f, const T* src, std::size_t count) {
    f.write(reinterpret_cast<const char*>(src),
            static_cast<std::streamsize>(sizeof(T) * count));
    if (!f) die("写入失败");
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) die("用法: difftest_policy <cases.bin> <out.bin>");

    if (!brain::policy_available()) {
        die("没有编译进策略权重——先跑 tools/difftest_policy.py（它会导出并重编）");
    }

    std::ifstream in(argv[1], std::ios::binary);
    if (!in) die(std::string("打不开 ") + argv[1]);

    char magic[4];
    read_exact(in, magic, 4, "magic");
    if (std::memcmp(magic, kCasesMagic, 4) != 0) die("case 文件 magic 不对");
    uint32_t version = 0, seqs = 0, steps = 0;
    read_exact(in, &version, 1, "version");
    if (version != kVersion) die("case 文件版本不支持");
    read_exact(in, &seqs, 1, "seqs");
    read_exact(in, &steps, 1, "steps");

    std::ofstream out(argv[2], std::ios::binary);
    if (!out) die(std::string("写不了 ") + argv[2]);
    write_exact(out, kOutMagic, 4);
    write_exact(out, &kVersion, 1);

    std::vector<float> obs(brain::kPolicyObsDim);
    std::vector<float> hidden(brain::kPolicyHidden);
    std::vector<float> logits(brain::kPolicyActDim);
    float value = 0.0f;

    for (uint32_t s = 0; s < seqs; ++s) {
        read_exact(in, hidden.data(), brain::kPolicyHidden, "init hidden");
        for (uint32_t t = 0; t < steps; ++t) {
            read_exact(in, obs.data(), brain::kPolicyObsDim, "obs");
            if (!brain::policy_forward(obs.data(), hidden.data(), logits.data(), &value)) {
                die("policy_forward 返回 false");
            }
            // 逐步写出：logits、value、以及**更新后**的 hidden。
            // 比对每一步的 hidden 而不只是最后一步，是为了让"隐状态在第几步开始漂"
            // 这件事在失败时一眼可见。
            write_exact(out, logits.data(), brain::kPolicyActDim);
            write_exact(out, &value, 1);
            write_exact(out, hidden.data(), brain::kPolicyHidden);
        }
    }

    std::printf("difftest_policy: %u 条序列 × %u 步, 已写出 %s\n", seqs, steps, argv[2]);
    return 0;
}

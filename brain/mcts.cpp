#include "brain/mcts.h"

#include "brain/net.h"
#include "obs/encode.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <random>

namespace brain {
namespace {

// 结束本方行动阶段的合成动作。
// 没有它，搜索只能"用满 3 个行动"才换手，无法表达"已经站进得分区、
// 再动只会暴露自己"这类局面。
constexpr int kStop = -1;

char other_side(char s) { return s == 'R' ? 'B' : 'R'; }

struct Node {
    sim::State state;
    sim::Belief belief;      // 敌方（局部框架下的 'B'）可能位置
    char to_move = 'R';
    int used = 0;
    bool free_turn = false;
    bool terminal = false;
    bool root_act = false;   // 是否仍在我方这一个 act 之内（根部屏蔽用）
    int depth = 0;

    bool expanded = false;
    std::vector<Cand> cands;
    std::vector<int> child;
    std::vector<double> prior;
    std::vector<double> value_sum; // 从**本节点走子方**视角累计
    std::vector<int> visits;

    int total_visits = 0;
    double total_value = 0.0;
};

class Search {
public:
    Search(const Weights& w, const MctsConfig& cfg, const Deadline& dl)
        : w_(w), cfg_(cfg), dl_(dl), rng_(cfg.seed) {}

    MctsResult run(const TurnInput& in) {
        root_bans_ = &in.bans;
        root_enemy_visible_ = in.enemy_visible;

        const int root = alloc(in.state, in.belief, 'R', in.used_by_now,
                               in.free_turn_available, 0, true);
        expand(root);
        if (cfg_.add_root_noise) add_dirichlet_noise(root);
        if (nodes_[root].child.empty()) {
            // 一个可行动作都没有（理论上不会发生，兜底不崩）
            MctsResult r;
            r.nodes = static_cast<long long>(nodes_.size());
            return r;
        }

        while (!out_of_time(dl_) && !time_exhausted_) {
            if (static_cast<long long>(nodes_.size()) >= cfg_.max_nodes) break;
            simulate(root);
        }

        MctsResult result;
        result.nodes = static_cast<long long>(nodes_.size());
        result.time_exhausted = time_exhausted_;
        result.root_value = nodes_[root].total_visits > 0
                                ? nodes_[root].total_value / nodes_[root].total_visits
                                : 0.0;
        result.plan = pick_plan(root, result.pi);
        return result;
    }

private:
    int alloc(const sim::State& s, const sim::Belief& b, char to_move, int used,
              bool free_turn, int depth, bool root_act) {
        Node n;
        n.state = s;
        n.belief = b;
        n.to_move = to_move;
        n.used = used;
        n.free_turn = free_turn;
        n.terminal = is_terminal(s);
        n.depth = depth;
        n.root_act = root_act;
        nodes_.push_back(std::move(n));
        return static_cast<int>(nodes_.size()) - 1;
    }

    // 叶值：从该节点走子方视角评估。不做随机 rollout —— 本游戏完全确定性，
    // 随机走子到终局几乎没有信息量。
    double leaf_value(const Node& n) const {
        if (n.terminal) return terminal_value(n.state, n.to_move);
        return evaluate_for(n.state, w_, n.belief, n.to_move);
    }

    // 由父节点的动作生成子节点；不可行返回 -1
    int make_child(int parent, const Cand& c) {
        const Node& n = nodes_[parent];
        sim::State ns = n.state;
        char to_move = n.to_move;
        int used = n.used;
        bool free_turn = n.free_turn;

        bool our_hit = false;
        if (c.action != kStop) {
            bool hit = false;
            if (!apply_step(ns, to_move, c, used, free_turn, &hit)) return -1;
            our_hit = hit && to_move == 'R';
        }

        // 阶段切换：主动收手，或用满额度
        const bool ends_phase = (c.action == kStop) || used >= kMaxActionsPerTurn;
        if (ends_phase) {
            end_phase(ns, to_move);
            to_move = other_side(to_move);
            used = 0;
            free_turn = sim::is_at_spawn(ns, to_move);
        }

        // 信念只由我方（'R'）的视野证伪，与树里轮到谁无关
        sim::Belief nb = n.belief;
        if (our_hit) {
            // 命中 → 对手瞬移回出生点。瞬移不属于"3 格内的移动"，
            // 可达集扩张覆盖不到，必须直接把信念塌缩过去。
            nb.reset_to(sim::spawn_of('B', ns.size));
        } else {
            sim::prune_by_vision(nb, ns.sentry_for('R'), ns.obstacles);
            sim::prune_impossible(nb, ns.sentry_for('R').last_known_pos, ns.obstacles);
            if (nb.empty()) nb = n.belief;
        }

        const bool still_root_act = n.root_act && !ends_phase;
        return alloc(ns, nb, to_move, used, free_turn, n.depth + 1, still_root_act);
    }

    void expand(int idx) {
        Node& n = nodes_[idx];
        n.expanded = true;
        if (n.terminal) return;

        const Sentry& me = n.state.sentry_for(n.to_move);
        const Sentry& opp = n.state.sentry_for(other_side(n.to_move));
        bool can_see_enemy = sim::can_see(me, opp.last_known_pos, n.state.obstacles);
        // 根节点还要算上 SCAN 的临时视野（sim 不建模它，由 agent 传进来）
        if (idx == 0) can_see_enemy = can_see_enemy || root_enemy_visible_;

        // 屏蔽列表只对"我方这一个 act 内"的节点有意义
        const Bans* bans =
            (n.root_act && n.to_move == 'R' && root_bans_ != nullptr) ? root_bans_ : nullptr;

        collect_candidates(n.state, n.to_move, n.free_turn, can_see_enemy, bans,
                           &n.belief, n.cands);
        n.cands.push_back({kStop, 0});

        std::vector<double> child_value;
        child_value.reserve(n.cands.size());
        for (const Cand& c : n.cands) {
            const int ci = make_child(idx, c);
            n.child.push_back(ci);
            child_value.push_back(ci >= 0 ? evaluate_for(nodes_[ci].state, w_,
                                                         nodes_[ci].belief, n.to_move)
                                          : -kWinValue);
        }

        // —— 先验 ——
        // 根节点优先用训练出来的策略网络（若已导出权重）；否则退回一步估值的
        // softmax。只在根节点用网络，因为树里对手节点的观测需要"对手的信念"，
        // 而我们没有——那部分继续用评估函数（evaluate_for 对任意一方都有定义）。
        n.prior.assign(n.cands.size(), 0.0);
        bool have_net_prior = false;
        if (idx == 0 && net_available()) {
            float obsbuf[obs::kObsDim];
            obs::encode(n.state, n.belief, root_enemy_visible_, n.used, n.free_turn,
                        obsbuf);
            float net_prior[kActionDim] = {0};
            float net_value = 0.0f;
            if (net_prior_value(obsbuf, net_prior, &net_value)) {
                double s = 0.0;
                for (size_t i = 0; i < n.cands.size(); ++i) {
                    const int ai = cand_to_index(n.cands[i]);
                    n.prior[i] = (ai >= 0) ? net_prior[ai] : 0.0;
                    s += n.prior[i];
                }
                if (s > 1e-8) {
                    for (double& p : n.prior) p /= s;
                    have_net_prior = true;
                }
            }
        }

        if (!have_net_prior) {
            const double vmax = *std::max_element(child_value.begin(), child_value.end());
            double sum = 0.0;
            for (size_t i = 0; i < child_value.size(); ++i) {
                n.prior[i] = std::exp((child_value[i] - vmax) / cfg_.prior_temperature);
                sum += n.prior[i];
            }
            if (sum > 0.0) {
                for (double& p : n.prior) p /= sum;
            } else {
                const double u = 1.0 / static_cast<double>(n.prior.size());
                for (double& p : n.prior) p = u;
            }
        }

        n.value_sum.assign(n.cands.size(), 0.0);
        n.visits.assign(n.cands.size(), 0);
    }

    void add_dirichlet_noise(int idx) {
        Node& n = nodes_[idx];
        if (n.prior.empty()) return;
        std::gamma_distribution<double> gamma(cfg_.dirichlet_alpha, 1.0);
        std::vector<double> noise(n.prior.size());
        double sum = 0.0;
        for (double& g : noise) {
            g = gamma(rng_);
            sum += g;
        }
        if (sum <= 0.0) return;
        const double eps = cfg_.dirichlet_eps;
        for (size_t i = 0; i < n.prior.size(); ++i) {
            n.prior[i] = (1.0 - eps) * n.prior[i] + eps * (noise[i] / sum);
        }
    }

    int select(const Node& n) const {
        int best = 0;
        double best_score = -1e300;
        const double sqrt_total = std::sqrt(static_cast<double>(n.total_visits) + 1.0);
        for (size_t i = 0; i < n.child.size(); ++i) {
            const double q =
                n.visits[i] > 0 ? n.value_sum[i] / static_cast<double>(n.visits[i]) : 0.0;
            const double u = cfg_.c_puct * n.prior[i] * sqrt_total /
                             (1.0 + static_cast<double>(n.visits[i]));
            const double score = q + u;
            if (score > best_score) {
                best_score = score;
                best = static_cast<int>(i);
            }
        }
        return best;
    }

    double simulate(int idx) {
        Node& n = nodes_[idx];
        n.total_visits += 1;

        if (n.terminal) return terminal_value(n.state, n.to_move);
        if (!n.expanded) {
            expand(idx);
            return leaf_value(nodes_[idx]);
        }
        if (n.depth >= cfg_.max_depth || out_of_time(dl_)) {
            time_exhausted_ = time_exhausted_ || out_of_time(dl_);
            return leaf_value(n);
        }

        const int i = select(n);
        const int c = n.child[i];
        if (c < 0) return leaf_value(n); // 不该发生，兜底

        const double child_value = simulate(c);
        // 负极大值回传的前提是"父子节点轮流走子"。
        // 但本游戏一个行动阶段内最多 3 步，同一个阶段内父子节点的走子方**相同**，
        // 此时绝不能取负——取了就会让值在阶段内无意义地翻符号，
        // 结果根节点会认为"什么都不做"是最优（实测正是如此）。
        const double v = (nodes_[c].to_move == nodes_[idx].to_move) ? child_value
                                                                     : -child_value;
        Node& m = nodes_[idx];
        m.value_sum[i] += v;
        m.visits[i] += 1;
        m.total_value += v;
        return v;
    }

    Plan pick_plan(int root, float* pi_out) {
        const Node& n = nodes_[root];
        Plan plan;
        if (n.child.empty()) return plan;

        // 训练目标 π：根节点各子节点的访问次数，压到固定动作空间后归一化
        double total = 0.0;
        for (size_t i = 0; i < n.child.size(); ++i) {
            const int idx = cand_to_index(n.cands[i]);
            if (idx < 0 || idx >= kActionDim) continue;
            pi_out[idx] += static_cast<float>(n.visits[i]);
            total += static_cast<double>(n.visits[i]);
        }
        if (total > 0.0) {
            for (int i = 0; i < kActionDim; ++i) {
                pi_out[i] = static_cast<float>(pi_out[i] / total);
            }
        }

        int best = 0;
        if (cfg_.sample_temperature > 0.0) {
            // 自对弈：按访问次数^(1/T) 采样，制造多样性
            std::vector<double> w(n.child.size());
            double sum = 0.0;
            for (size_t i = 0; i < n.child.size(); ++i) {
                w[i] = std::pow(static_cast<double>(n.visits[i]) + 1.0,
                                1.0 / cfg_.sample_temperature);
                sum += w[i];
            }
            std::uniform_real_distribution<double> uni(0.0, sum);
            double x = uni(rng_);
            for (size_t i = 0; i < w.size(); ++i) {
                x -= w[i];
                if (x <= 0.0) {
                    best = static_cast<int>(i);
                    break;
                }
            }
        } else {
            for (size_t i = 0; i < n.child.size(); ++i) {
                if (n.visits[i] > n.visits[best]) best = static_cast<int>(i);
            }
        }

        // 只返回第一步：agent 每步重新规划（分叉重规划），
        // 计划执行到一半被现实推翻时会自动纠正
        plan.count = 1;
        plan.actions[0] = n.cands[best].action;
        plan.args[0] = n.cands[best].arg;
        plan.value = nodes_[root].total_visits > 0
                         ? nodes_[root].total_value / nodes_[root].total_visits
                         : 0.0;
        plan.valid = true;
        return plan;
    }

    const Weights& w_;
    const MctsConfig& cfg_;
    const Deadline& dl_;
    std::mt19937 rng_;
    // 用 deque 而不是 vector：expand/make_child 里会持有 Node&，
    // 而 push_back 若导致 vector 重分配，那些引用就全部失效（悬垂引用）。
    // deque 的 push_back 不会使既有元素的引用失效。
    std::deque<Node> nodes_;
    const Bans* root_bans_ = nullptr;
    bool root_enemy_visible_ = false;
    bool time_exhausted_ = false;
};

} // namespace

MctsResult mcts_search(const TurnInput& in, const Weights& w, const MctsConfig& cfg,
                       const Deadline& deadline) {
    Search search(w, cfg, deadline);
    return search.run(in);
}

} // namespace brain

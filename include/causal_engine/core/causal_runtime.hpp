#ifndef CAUSAL_ENGINE_CORE_CAUSAL_RUNTIME_HPP
#define CAUSAL_ENGINE_CORE_CAUSAL_RUNTIME_HPP

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <atomic>
#include <memory>
#include <mutex>
#include <cmath>
#include <algorithm>

namespace causal_engine {
namespace core {

// =========================================================================
// 1. Causal Signal & Event Lock-Free Ring Buffer
// =========================================================================
struct CausalSignal {
    uint64_t signal_id = 0;
    uint32_t source_node_id = 0;
    uint32_t target_node_id = 0;
    float magnitude = 1.0f;
    float radius = 10.0f;
    float position[3] = {0.0f, 0.0f, 0.0f};
    uint64_t timestamp_ns = 0;
};

template <size_t Capacity = 1024>
class LockFreeEventRingBuffer {
private:
    CausalSignal buffer_[Capacity];
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};

public:
    LockFreeEventRingBuffer() = default;

    bool enqueue(const CausalSignal& signal) {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) % Capacity;

        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; // Buffer full
        }

        buffer_[current_tail] = signal;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    bool dequeue(CausalSignal& signal) {
        size_t current_head = head_.load(std::memory_order_relaxed);
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false; // Buffer empty
        }

        signal = buffer_[current_head];
        head_.store((current_head + 1) % Capacity, std::memory_order_release);
        return true;
    }

    size_t size() const {
        size_t h = head_.load(std::memory_order_relaxed);
        size_t t = tail_.load(std::memory_order_relaxed);
        return (t >= h) ? (t - h) : (Capacity - h + t);
    }
};

// =========================================================================
// 2. Causal Graph Manager
// =========================================================================
struct CausalNode {
    uint32_t id = 0;
    float position[3] = {0.0f, 0.0f, 0.0f};
    float state_value = 0.0f;
    bool manifested = false;
    uint64_t last_evaluated_tick = 0;
    std::vector<uint32_t> outgoing_edges;
};

class CausalGraphManager {
private:
    std::unordered_map<uint32_t, CausalNode> nodes_;

public:
    void add_node(uint32_t id, float x, float y, float z) {
        CausalNode node;
        node.id = id;
        node.position[0] = x;
        node.position[1] = y;
        node.position[2] = z;
        nodes_[id] = node;
    }

    bool add_edge(uint32_t from, uint32_t to) {
        if (nodes_.find(from) == nodes_.end() || nodes_.find(to) == nodes_.end()) {
            return false;
        }
        if (has_path(to, from)) {
            // Adding this edge would cause a cycle
            return false;
        }
        nodes_[from].outgoing_edges.push_back(to);
        return true;
    }

    bool has_path(uint32_t start, uint32_t target) const {
        if (start == target) return true;
        std::unordered_set<uint32_t> visited;
        std::queue<uint32_t> q;
        q.push(start);
        visited.insert(start);

        while (!q.empty()) {
            uint32_t curr = q.front();
            q.pop();

            auto it = nodes_.find(curr);
            if (it == nodes_.end()) continue;

            for (uint32_t neighbor : it->second.outgoing_edges) {
                if (neighbor == target) return true;
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    q.push(neighbor);
                }
            }
        }
        return false;
    }

    CausalNode* get_node(uint32_t id) {
        auto it = nodes_.find(id);
        return (it != nodes_.end()) ? &it->second : nullptr;
    }

    const std::unordered_map<uint32_t, CausalNode>& get_all_nodes() const {
        return nodes_;
    }

    size_t node_count() const { return nodes_.size(); }
};

// =========================================================================
// 3. Lazy Evaluation Ledger
// =========================================================================
struct PendingEvaluation {
    uint32_t node_id;
    uint64_t target_tick;
    float accumulated_delta;
};

class LazyEvaluationLedger {
private:
    std::vector<PendingEvaluation> ledger_;

public:
    void record_deferred_computation(uint32_t node_id, uint64_t tick, float delta) {
        ledger_.push_back({node_id, tick, delta});
    }

    size_t pending_count() const { return ledger_.size(); }

    void resolve_pending(CausalGraphManager& graph, uint64_t current_tick, int lod_step = 1) {
        std::vector<PendingEvaluation> remaining;
        for (const auto& item : ledger_) {
            if (item.node_id % lod_step == 0) {
                auto* node = graph.get_node(item.node_id);
                if (node) {
                    node->state_value += item.accumulated_delta;
                    node->last_evaluated_tick = current_tick;
                    if (std::abs(node->state_value) > 0.5f) {
                        node->manifested = true;
                    }
                }
            } else {
                // Keep deferred if skipped by LOD step
                remaining.push_back(item);
            }
        }
        ledger_ = std::move(remaining);
    }
};

} // namespace core
} // namespace causal_engine

#endif // CAUSAL_ENGINE_CORE_CAUSAL_RUNTIME_HPP

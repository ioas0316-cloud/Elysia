#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cmath>

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#else
#include "elysia/cuda_host_stub.h"
#endif

struct FractalNode {
    int id;                          // 노드 고유 ID
    int parent_id{-1};               // 부모 노드 ID (-1 = 루트)
    std::vector<int> children_ids;   // 분열 생성된 자식 노드 ID 목록

    int scale_level{0};              // 스케일 차원 깊이
    float2 phase_state{make_float2(1.0f, 0.0f)};  // 현재 클리포드 바이브이그저 위상 (cos θ, sin θ)
    float coherence{1.0f};           // 위상 안정도
    bool is_fused{false};            // 융합 여부

    FractalNode(int node_id, int level, int parent = -1)
        : id(node_id), parent_id(parent), scale_level(level) {}
};

class FractalCausalTree {
private:
    std::unordered_map<int, std::shared_ptr<FractalNode>> node_table_;
    int root_id_{0};
    int next_node_id_{1};

public:
    FractalCausalTree() {
        auto root = std::make_shared<FractalNode>(0, 0, -1);
        node_table_[0] = root;
    }

    std::pair<int, int> register_mitosis(int parent_id, float2 child1_phase, float2 child2_phase) {
        auto parent = node_table_[parent_id];
        if (!parent) return {-1, -1};

        int child1_id = next_node_id_++;
        int child2_id = next_node_id_++;

        int next_level = parent->scale_level + 1;

        auto c1 = std::make_shared<FractalNode>(child1_id, next_level, parent_id);
        auto c2 = std::make_shared<FractalNode>(child2_id, next_level, parent_id);

        c1->phase_state = child1_phase;
        c2->phase_state = child2_phase;

        parent->children_ids.push_back(child1_id);
        parent->children_ids.push_back(child2_id);

        node_table_[child1_id] = c1;
        node_table_[child2_id] = c2;

        return {child1_id, child2_id};
    }

    void register_fusion(int child1_id, int child2_id, float2 fused_phase) {
        auto c1 = node_table_[child1_id];
        auto c2 = node_table_[child2_id];
        if (!c1 || !c2 || c1->parent_id != c2->parent_id) return;

        auto parent = node_table_[c1->parent_id];
        if (parent) {
            parent->phase_state = fused_phase;
            parent->is_fused = true;

            node_table_.erase(child1_id);
            node_table_.erase(child2_id);
            parent->children_ids.clear();
        }
    }

    [[nodiscard]] float calculate_systemic_fractal_coherence() const {
        float total_coherence = 0.0f;
        size_t evaluated_count = 0;

        for (const auto& [id, node] : node_table_) {
            if (node->parent_id != -1 && node_table_.count(node->parent_id)) {
                auto parent = node_table_.at(node->parent_id);
                float dot = node->phase_state.x * parent->phase_state.x +
                            node->phase_state.y * parent->phase_state.y;
                total_coherence += std::abs(dot);
                evaluated_count++;
            }
        }

        return evaluated_count > 0 ? (total_coherence / static_cast<float>(evaluated_count)) : 1.0f;
    }

    [[nodiscard]] const std::unordered_map<int, std::shared_ptr<FractalNode>>& get_node_table() const {
        return node_table_;
    }

    void print_tree_structure() const {
        std::cout << "\n===================================================\n";
        std::cout << " [FractalCausalTree] Observational Hierarchy Graph\n";
        std::cout << "===================================================\n";
        std::cout << " Total Active Tree Nodes: " << node_table_.size() << "\n";
        std::cout << " Systemic Fractal Coherence: " << calculate_systemic_fractal_coherence() << "\n";

        for (const auto& [id, node] : node_table_) {
            std::cout << " Node [" << node->id << "] | Level: " << node->scale_level
                      << " | Parent: " << node->parent_id
                      << " | Children Count: " << node->children_ids.size() << "\n";
        }
        std::cout << "===================================================\n\n";
    }
};

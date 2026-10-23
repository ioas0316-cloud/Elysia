#pragma once

#include <iostream>
#include <vector>
#include <algorithm>

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#else
#include "elysia/cuda_host_stub.h"
#endif

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
extern "C" __global__ void k_mitotic_tensor_branching(
    const float2* parent_rotors, const float* curvature_map,
    float2* child1_rotors, float2* child2_rotors,
    int* branch_mask, const float curvature_threshold, const int num_nodes);

extern "C" __global__ void k_parasympathetic_node_fusion(
    const float2* child1_rotors, const float2* child2_rotors,
    float2* fused_rotors, int* fusion_mask,
    const float coherence_threshold, const int num_split_pairs);
#else
inline void k_mitotic_tensor_branching_host(
    const float2* parent_rotors, const float* curvature_map,
    float2* child1_rotors, float2* child2_rotors,
    int* branch_mask, const float curvature_threshold, const int num_nodes) {
    for (int idx = 0; idx < num_nodes; ++idx) {
        float cur = curvature_map[idx];
        if (cur >= curvature_threshold) {
            float2 p = parent_rotors[idx];
            const float norm_factor = 0.70710678f;
            const float cos_p = 0.70710678f;
            const float sin_p = 0.70710678f;

            float2 c1, c2;
            c1.x = (p.x * cos_p - p.y * sin_p) * norm_factor;
            c1.y = (p.x * sin_p + p.y * cos_p) * norm_factor;

            c2.x = (p.x * cos_p + p.y * sin_p) * norm_factor;
            c2.y = (-p.x * sin_p + p.y * cos_p) * norm_factor;

            child1_rotors[idx] = c1;
            child2_rotors[idx] = c2;
            branch_mask[idx] = 1;
        } else {
            child1_rotors[idx] = parent_rotors[idx];
            child2_rotors[idx] = make_float2(0.0f, 0.0f);
            branch_mask[idx] = 0;
        }
    }
}

inline void k_parasympathetic_node_fusion_host(
    const float2* child1_rotors, const float2* child2_rotors,
    float2* fused_rotors, int* fusion_mask,
    const float coherence_threshold, const int num_split_pairs) {
    for (int idx = 0; idx < num_split_pairs; ++idx) {
        float2 c1 = child1_rotors[idx];
        float2 c2 = child2_rotors[idx];

        float mag1_sq = c1.x * c1.x + c1.y * c1.y;
        float mag2_sq = c2.x * c2.x + c2.y * c2.y;

        if (mag1_sq < 1e-7f || mag2_sq < 1e-7f) {
            fusion_mask[idx] = 0;
            continue;
        }

        float dot_product = c1.x * c2.x + c1.y * c2.y;
        float phase_coherence = dot_product / std::sqrt(mag1_sq * mag2_sq);

        if (phase_coherence >= coherence_threshold) {
            float sum_x = c1.x + c2.x;
            float sum_y = c1.y + c2.y;
            float sum_mag = std::sqrt(sum_x * sum_x + sum_y * sum_y);

            if (sum_mag > 1e-7f) {
                fused_rotors[idx] = make_float2(sum_x / sum_mag, sum_y / sum_mag);
                fusion_mask[idx] = 1;
            } else {
                fusion_mask[idx] = 0;
            }
        } else {
            fusion_mask[idx] = 0;
        }
    }
}
#endif

class MitoticCausalTensorSpace {
private:
    size_t current_capacity_{0};    // 현재 관측 노드 총량
    size_t active_nodes_{0};        // 활성 노드 수
    int    scale_level_{0};         // 관측 스케일 계층 수

    // Device Buffers
    float2* d_rotors_{nullptr};
    float*  d_curvature_map_{nullptr};
    int*    d_branch_mask_{nullptr};

public:
    MitoticCausalTensorSpace(size_t initial_nodes)
        : current_capacity_(initial_nodes), active_nodes_(initial_nodes) {

        cudaMalloc(reinterpret_cast<void**>(&d_rotors_), current_capacity_ * sizeof(float2));
        cudaMalloc(reinterpret_cast<void**>(&d_curvature_map_), current_capacity_ * sizeof(float));
        cudaMalloc(reinterpret_cast<void**>(&d_branch_mask_), current_capacity_ * sizeof(int));

        std::vector<float2> h_init(current_capacity_, make_float2(1.0f, 0.0f));
        cudaMemcpy(d_rotors_, h_init.data(), current_capacity_ * sizeof(float2), cudaMemcpyHostToDevice);
        cudaMemset(d_curvature_map_, 0, current_capacity_ * sizeof(float));
    }

    ~MitoticCausalTensorSpace() {
        if (d_rotors_) cudaFree(d_rotors_);
        if (d_curvature_map_) cudaFree(d_curvature_map_);
        if (d_branch_mask_) cudaFree(d_branch_mask_);
    }

    void execute_mitotic_division(float curvature_threshold, cudaStream_t stream) {
        float2* d_child1 = nullptr;
        float2* d_child2 = nullptr;

        cudaMalloc(reinterpret_cast<void**>(&d_child1), active_nodes_ * sizeof(float2));
        cudaMalloc(reinterpret_cast<void**>(&d_child2), active_nodes_ * sizeof(float2));

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
        int block_size = 256;
        int grid_size = static_cast<int>((active_nodes_ + block_size - 1) / block_size);

        k_mitotic_tensor_branching<<<grid_size, block_size, 0, stream>>>(
            d_rotors_, d_curvature_map_, d_child1, d_child2, d_branch_mask_,
            curvature_threshold, static_cast<int>(active_nodes_)
        );
#else
        k_mitotic_tensor_branching_host(
            d_rotors_, d_curvature_map_, d_child1, d_child2, d_branch_mask_,
            curvature_threshold, static_cast<int>(active_nodes_)
        );
#endif

        std::vector<int> h_mask(active_nodes_);
        cudaMemcpyAsync(h_mask.data(), d_branch_mask_, active_nodes_ * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        size_t split_count = 0;
        for (int m : h_mask) if (m == 1) split_count++;

        if (split_count > 0) {
            scale_level_++;
            size_t new_active_nodes = active_nodes_ + split_count;
            std::cout << "[Mitosis Event] Scale Level: " << scale_level_
                      << " | Bypassed Cells Split: " << split_count
                      << " | Total Observational Nodes: " << active_nodes_ << " -> " << new_active_nodes << "\n";

            float2* d_new_rotors = nullptr;
            cudaMalloc(reinterpret_cast<void**>(&d_new_rotors), new_active_nodes * sizeof(float2));

            cudaMemcpyAsync(d_new_rotors, d_child1, active_nodes_ * sizeof(float2), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(d_new_rotors + active_nodes_, d_child2, split_count * sizeof(float2), cudaMemcpyDeviceToDevice, stream);

            cudaFree(d_rotors_);
            d_rotors_ = d_new_rotors;
            active_nodes_ = new_active_nodes;

            cudaFree(d_curvature_map_);
            cudaMalloc(reinterpret_cast<void**>(&d_curvature_map_), active_nodes_ * sizeof(float));
            cudaMemsetAsync(d_curvature_map_, 0, active_nodes_ * sizeof(float), stream);
        }

        cudaFree(d_child1);
        cudaFree(d_child2);
    }

    void execute_parasympathetic_fusion(float coherence_threshold, cudaStream_t stream) {
        if (scale_level_ == 0 || active_nodes_ <= 2) return;

        size_t pair_count = active_nodes_ / 2;
        float2 *d_c1 = d_rotors_;
        float2 *d_c2 = d_rotors_ + pair_count;

        float2* d_fused = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&d_fused), pair_count * sizeof(float2));

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
        int block_size = 256;
        int grid_size = static_cast<int>((pair_count + block_size - 1) / block_size);

        k_parasympathetic_node_fusion<<<grid_size, block_size, 0, stream>>>(
            d_c1, d_c2, d_fused, d_branch_mask_, coherence_threshold, static_cast<int>(pair_count)
        );
#else
        k_parasympathetic_node_fusion_host(
            d_c1, d_c2, d_fused, d_branch_mask_, coherence_threshold, static_cast<int>(pair_count)
        );
#endif

        std::vector<int> h_mask(pair_count);
        cudaMemcpyAsync(h_mask.data(), d_branch_mask_, pair_count * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        size_t fused_count = 0;
        for (int m : h_mask) if (m == 1) fused_count++;

        if (fused_count > 0) {
            size_t new_active_nodes = active_nodes_ - fused_count;
            scale_level_--;

            std::cout << "[Parasympathetic Sleep] Reverse-Mitosis Fusion Executed!\n"
                      << " -> Fused Nodes: " << fused_count
                      << " | Active Observational Capacity: " << active_nodes_ << " -> " << new_active_nodes
                      << " (VRAM Reclaimed)\n";

            float2* d_compacted = nullptr;
            cudaMalloc(reinterpret_cast<void**>(&d_compacted), new_active_nodes * sizeof(float2));
            cudaMemcpyAsync(d_compacted, d_fused, new_active_nodes * sizeof(float2), cudaMemcpyDeviceToDevice, stream);

            cudaFree(d_rotors_);
            d_rotors_ = d_compacted;
            active_nodes_ = new_active_nodes;
        }

        cudaFree(d_fused);
    }

    void inject_mock_curvature(const std::vector<float>& h_curvatures) {
        size_t copy_size = std::min(h_curvatures.size(), active_nodes_);
        cudaMemcpy(d_curvature_map_, h_curvatures.data(), copy_size * sizeof(float), cudaMemcpyHostToDevice);
    }

    [[nodiscard]] size_t get_active_nodes() const { return active_nodes_; }
    [[nodiscard]] int get_scale_level() const { return scale_level_; }
    [[nodiscard]] float2* get_rotors_device_ptr() { return d_rotors_; }
    [[nodiscard]] float* get_curvature_device_ptr() { return d_curvature_map_; }
};

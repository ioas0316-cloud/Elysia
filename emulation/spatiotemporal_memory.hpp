#ifndef SPATIOTEMPORAL_MEMORY_HPP
#define SPATIOTEMPORAL_MEMORY_HPP

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <memory>
#include <iostream>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HOST_DEVICE __host__ __device__
#define HOST_FUNC __host__
#define DEVICE_FUNC __device__
#define GLOBAL_FUNC __global__
#else
#include "cuda_stub.hpp"
#define HOST_DEVICE
#define HOST_FUNC
#define DEVICE_FUNC
#define GLOBAL_FUNC
#endif

namespace elysia::emulation {

// ============================================================================
// 0. 기본 기하 구조체 (Geometric Primitive Types)
// ============================================================================

/**
 * @brief Cl(3,0) 기하 대수 기반 스피너 / 회전자 (Rotor)
 * R = a + b(e12) + c(e23) + d(e31)
 */
struct alignas(16) Rotor {
    float scalar;        ///< 스칼라 성분 (a)
    float bivector_xy;   ///< e12 이중벡터 성분
    float bivector_yz;   ///< e23 이중벡터 성분
    float bivector_zx;   ///< e31 이중벡터 성분

    HOST_DEVICE inline Rotor reverse() const {
        return { scalar, -bivector_xy, -bivector_yz, -bivector_zx };
    }

    HOST_DEVICE inline Rotor multiply(const Rotor& other) const {
        // Geometric product of two rotors in Cl(3,0)
        return {
            scalar * other.scalar - bivector_xy * other.bivector_xy - bivector_yz * other.bivector_yz - bivector_zx * other.bivector_zx,
            scalar * other.bivector_xy + bivector_xy * other.scalar - bivector_yz * other.bivector_zx + bivector_zx * other.bivector_yz,
            scalar * other.bivector_yz + bivector_yz * other.scalar - bivector_zx * other.bivector_xy + bivector_xy * other.bivector_zx,
            scalar * other.bivector_zx + bivector_zx * other.scalar - bivector_xy * other.bivector_yz + bivector_yz * other.bivector_xy
        };
    }
};

/**
 * @brief 국소 시공간 계량 텐서 g_mem (3x3 대칭 텐서)
 */
struct alignas(16) MetricTensor3x3 {
    float g[3][3];
};

/**
 * @brief 국소 차트(Chart) 식별 및 바운딩 림
 */
struct LocalChart {
    uint32_t chart_id;
    float center[3];
    float radius;
    bool is_active;
};

/**
 * @brief 유량 및 유속 모니터링 셀 구조체
 */
struct FluxCell {
    float density;       ///< 현재 정보/상태 밀도 (rho)
    float velocity[3];   ///< 유속 벡터 (v)
    float divergence;    ///< 리만 발산 및 병목 누적치 (P_bottleneck)
};

/**
 * @brief 옥트리(Octree) 구조를 지원하는 확장 국소 차트 노드
 */
struct ScaledLocalChartNode {
    LocalChart chart_data;
    uint32_t parent_id{0xFFFFFFFF};
    uint32_t children_ids[8]{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                             0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
    bool is_leaf{true};
    uint8_t depth{0};
};

// ============================================================================
// 1. StaticRotorUnit (SRU) - 정적 로터 보조 메모리 인터페이스
// ============================================================================

class StaticRotorUnit {
public:
    struct RotorTagEntry {
        Rotor delta_r;           ///< 보존된 위상 오프셋 (Delta Rotor)
        float bg_drift_accum;    ///< 백그라운드 위상 드리프트 축적량
        bool is_pinned;          ///< SRAM Pinning 여부
    };

    explicit StaticRotorUnit(size_t max_tag_capacity);
    ~StaticRotorUnit();

    Rotor pin_evicted_phase(uint64_t cache_line_id, const Rotor& r_curr, const Rotor& r_base);
    Rotor restore_active_phase(uint64_t cache_line_id, const Rotor& r_global_now);
    void update_background_drift(float delta_t, float omega_bg);
    size_t get_pinned_count() const;

private:
    size_t capacity_;
    std::unordered_map<uint64_t, RotorTagEntry> tag_array_;
};

// ============================================================================
// 2. MetricFieldEngine - 리만 계량 텐서 및 가소성 연산 엔진
// ============================================================================

class MetricFieldEngine {
public:
    MetricFieldEngine(size_t grid_dim_x, size_t grid_dim_y, size_t grid_dim_z);
    ~MetricFieldEngine();

    void launch_riemannian_gradient_kernel(
        const float* d_pos_x, float* d_grad_out, size_t num_particles, cudaStream_t stream = 0);

    void launch_plasticity_update_kernel(
        const float* d_pos_x, const float* d_velocity, const Rotor* d_torques,
        float alpha, float beta, size_t num_particles, cudaStream_t stream = 0);

    void launch_entropy_decay_kernel(
        float gamma, float diffusion_D, float delta_t, cudaStream_t stream = 0);

    void launch_compute_flux_divergence_kernel(
        const float* d_density, const float* d_velocity, float* d_bottleneck_out,
        float grid_spacing, cudaStream_t stream = 0);

    void launch_apply_bottleneck_metric_stress_kernel(
        const float* d_bottleneck_index, float eta_stress_coefficient, cudaStream_t stream = 0);

    MetricTensor3x3* get_device_metric_ptr() const { return d_g_mem_field_; }
    size_t get_total_cells() const { return total_cells_; }
    size_t get_dim_x() const { return dim_x_; }
    size_t get_dim_y() const { return dim_y_; }
    size_t get_dim_z() const { return dim_z_; }

    void copy_metric_to_host(MetricTensor3x3* host_buffer) const;

private:
    size_t dim_x_, dim_y_, dim_z_;
    size_t total_cells_;
    MetricTensor3x3* d_g_mem_field_;
};

// ============================================================================
// 3. AtlasManager - 시공간 아틀라스 및 국소 차트 파편화 / 동적 스케일링
// ============================================================================

class AtlasManager {
public:
    struct ChartScaleTrigger {
        uint32_t chart_id;
        float max_bottleneck_pressure;
        bool requires_subdivision;
        bool requires_merge;
    };

    explicit AtlasManager(size_t initial_chart_count);
    ~AtlasManager();

    std::vector<uint32_t> query_active_charts(const float pos[3]);
    Rotor compute_chart_transition_rotor(uint32_t src_chart_id, uint32_t dst_chart_id);
    void update_chart_active_states(const std::vector<uint32_t>& active_ids);
    const std::vector<LocalChart>& get_all_charts() const;

    // Flux scaling extension
    std::vector<ChartScaleTrigger> evaluate_chart_bottlenecks(
        const float* host_bottleneck_map, float split_threshold, float merge_threshold,
        size_t dim_x = 64, size_t dim_y = 64, size_t dim_z = 64);
    void subdivide_chart(uint32_t parent_chart_id);
    void merge_charts(uint32_t parent_chart_id);

private:
    size_t max_depth_{4};
    uint32_t next_chart_id_{0};
    std::vector<LocalChart> charts_;
    std::unordered_map<uint32_t, ScaledLocalChartNode> chart_tree_;

    uint32_t generate_next_chart_id();
    float compute_max_bottleneck_in_chart(
        const LocalChart& chart, const float* bottleneck_map,
        size_t dim_x, size_t dim_y, size_t dim_z);
};

// ============================================================================
// 4. VirtualTierPipeline - Cache-RAM-SSD 비동기 에뮬레이터
// ============================================================================

class VirtualTierPipeline {
public:
    struct LatencyStats {
        double cache_latency_ns{0.5};
        double ram_latency_ns{50.0};
        double ssd_latency_ns{10000.0};
        size_t cache_misses{0};
        size_t ram_accesses{0};
        size_t ssd_dma_transfers{0};
    };

    VirtualTierPipeline();
    ~VirtualTierPipeline();

    void trigger_dma_ssd_inscription(uint32_t chart_id, const MetricTensor3x3* metric_data);
    LatencyStats get_stats() const { return stats_; }
    void simulate_eviction_fetch_cycle(StaticRotorUnit& sru, uint64_t cache_line_id, const Rotor& r_curr, const Rotor& r_base);

private:
    LatencyStats stats_;
};

} // namespace elysia::emulation

#endif // SPATIOTEMPORAL_MEMORY_HPP

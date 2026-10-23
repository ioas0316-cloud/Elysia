#ifndef SPATIOTEMPORAL_PIPELINE_HPP
#define SPATIOTEMPORAL_PIPELINE_HPP

#include "spatiotemporal_memory.hpp"
#include <unordered_map>
#include <vector>
#include <memory>

namespace elysia::emulation {

/**
 * @brief 2단계 실시간 통합 파이프라인 (Spatiotemporal Pipeline Stage 2)
 *
 * CUDA 리만 발산 커널과 AtlasManager 옥트리 제어 로직을 통합하고,
 * D2H 비동기 동기화 및 동적 GPU 버퍼 재할당/재바인딩을 오케스트레이션합니다.
 */
class SpatiotemporalPipelineStage2 {
public:
    SpatiotemporalPipelineStage2(size_t dim_x, size_t dim_y, size_t dim_z, float grid_spacing);
    ~SpatiotemporalPipelineStage2();

    /**
     * @brief 차트 버퍼 재바인딩 정보 구조체
     */
    struct ChartGPUBuffer {
        uint32_t chart_id;
        size_t cell_offset;
        size_t cell_count;
        MetricTensor3x3* d_chart_metric_ptr{nullptr};
        float* d_chart_density_ptr{nullptr};
        float* d_chart_velocity_ptr{nullptr};
    };

    /**
     * @brief 2단계 통합 실행 프레임 step()
     *
     * [유량/발산 계산 -> 계량 팽창 -> 비동기 D2H -> 차트 스케일링 평가 -> GPU 버퍼 재구성/바인딩]
     */
    void step(
        MetricFieldEngine& metric_engine,
        const float* d_density,
        const float* d_velocity,
        AtlasManager& atlas_manager,
        float eta_stress,
        float split_thresh,
        float merge_thresh);

    // Getters for inspection and testing
    const float* get_host_bottleneck_map() const { return host_bottleneck_map_; }
    const float* get_device_bottleneck_out() const { return d_bottleneck_out_; }
    const std::unordered_map<uint32_t, ChartGPUBuffer>& get_gpu_buffer_registry() const {
        return gpu_buffer_registry_;
    }

private:
    size_t dim_x_;
    size_t dim_y_;
    size_t dim_z_;
    size_t total_cells_;
    float grid_spacing_;

    float* d_bottleneck_out_{nullptr};
    float* host_bottleneck_map_{nullptr};

    cudaStream_t compute_stream_{nullptr};
    cudaStream_t copy_stream_{nullptr};
    cudaEvent_t divergence_done_event_{nullptr};

    std::unordered_map<uint32_t, ChartGPUBuffer> gpu_buffer_registry_;

    void update_gpu_buffer_bindings(MetricFieldEngine& metric_engine, float* d_density_base, float* d_velocity_base, const AtlasManager& atlas_manager);
};

} // namespace elysia::emulation

#endif // SPATIOTEMPORAL_PIPELINE_HPP

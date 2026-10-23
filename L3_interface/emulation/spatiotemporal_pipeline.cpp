#include "spatiotemporal_pipeline.hpp"
#include <iostream>
#include <algorithm>

namespace elysia::emulation {

SpatiotemporalPipelineStage2::SpatiotemporalPipelineStage2(
    size_t dim_x, size_t dim_y, size_t dim_z, float grid_spacing)
    : dim_x_(dim_x), dim_y_(dim_y), dim_z_(dim_z),
      total_cells_(dim_x * dim_y * dim_z), grid_spacing_(grid_spacing)
{
    // 1. CUDA 스트림 및 이벤트 생성
    cudaStreamCreate(&compute_stream_);
    cudaStreamCreate(&copy_stream_);
    cudaEventCreate(&divergence_done_event_);

    // 2. Host (Pinned) & Device 메모리 할당
    size_t float_bytes = total_cells_ * sizeof(float);
    cudaMallocHost(reinterpret_cast<void**>(&host_bottleneck_map_), float_bytes);
    std::memset(host_bottleneck_map_, 0, float_bytes);

    cudaMalloc(reinterpret_cast<void**>(&d_bottleneck_out_), float_bytes);
    cudaMemcpy(d_bottleneck_out_, host_bottleneck_map_, float_bytes, cudaMemcpyHostToDevice);
}

SpatiotemporalPipelineStage2::~SpatiotemporalPipelineStage2() {
    if (host_bottleneck_map_) {
        cudaFreeHost(host_bottleneck_map_);
    }
    if (d_bottleneck_out_) {
        cudaFree(d_bottleneck_out_);
    }
    if (compute_stream_) {
        cudaStreamDestroy(compute_stream_);
    }
    if (copy_stream_) {
        cudaStreamDestroy(copy_stream_);
    }
    if (divergence_done_event_) {
        cudaEventDestroy(divergence_done_event_);
    }
}

void SpatiotemporalPipelineStage2::step(
    MetricFieldEngine& metric_engine,
    const float* d_density,
    const float* d_velocity,
    AtlasManager& atlas_manager,
    float eta_stress,
    float split_thresh,
    float merge_thresh)
{
    // Step A: CUDA 커널 구동 - 리만 발산 및 병목 누적 수치 계산
    metric_engine.launch_compute_flux_divergence_kernel(
        d_density, d_velocity, d_bottleneck_out_,
        grid_spacing_, compute_stream_);

    // Step B: 계량 텐서 동적 팽창 (물리적 저항 형성)
    metric_engine.launch_apply_bottleneck_metric_stress_kernel(
        d_bottleneck_out_, eta_stress, compute_stream_);

    // GPU 계산 완료 이벤트 기록
    cudaEventRecord(divergence_done_event_, compute_stream_);

    // Step C: 비동기 D2H 메모리 복사 (Pinned Memory 및 copy_stream 활용)
    cudaStreamWaitEvent(copy_stream_, divergence_done_event_, 0);
    cudaMemcpyAsync(
        host_bottleneck_map_, d_bottleneck_out_,
        total_cells_ * sizeof(float),
        cudaMemcpyDeviceToHost, copy_stream_);

    // 복사 완료 동기화
    cudaStreamSynchronize(copy_stream_);

    // Step D: CPU AtlasManager - 병목 수치 기반 차트 분할/병합 평가
    auto triggers = atlas_manager.evaluate_chart_bottlenecks(
        host_bottleneck_map_, split_thresh, merge_thresh, dim_x_, dim_y_, dim_z_);

    // Step E: 동적 옥트리 차트 구조 변경 적용
    for (const auto& trigger : triggers) {
        if (trigger.requires_subdivision) {
            atlas_manager.subdivide_chart(trigger.chart_id);
            std::cout << "[AtlasManager] Chart " << trigger.chart_id
                      << " Subdivided (Bottleneck pressure: " << trigger.max_bottleneck_pressure << ")\n";
        } else if (trigger.requires_merge) {
            atlas_manager.merge_charts(trigger.chart_id);
            std::cout << "[AtlasManager] Chart " << trigger.chart_id << " Merged Back.\n";
        }
    }

    // Step F: 차트 변동에 대응한 GPU 메모리 바인딩 레지스트리 업데이트
    update_gpu_buffer_bindings(
        metric_engine,
        const_cast<float*>(d_density),
        const_cast<float*>(d_velocity),
        atlas_manager);
}

void SpatiotemporalPipelineStage2::update_gpu_buffer_bindings(
    MetricFieldEngine& metric_engine,
    float* d_density_base,
    float* d_velocity_base,
    const AtlasManager& atlas_manager)
{
    gpu_buffer_registry_.clear();

    MetricTensor3x3* d_metric_base = metric_engine.get_device_metric_ptr();
    const auto& charts = atlas_manager.get_all_charts();

    for (const auto& chart : charts) {
        if (!chart.is_active) continue;

        int center_x = std::min(std::max(static_cast<int>(chart.center[0] * dim_x_), 0), static_cast<int>(dim_x_) - 1);
        int center_y = std::min(std::max(static_cast<int>(chart.center[1] * dim_y_), 0), static_cast<int>(dim_y_) - 1);
        int center_z = std::min(std::max(static_cast<int>(chart.center[2] * dim_z_), 0), static_cast<int>(dim_z_) - 1);

        size_t cell_offset = center_x + center_y * dim_x_ + center_z * dim_x_ * dim_y_;

        ChartGPUBuffer buf;
        buf.chart_id = chart.chart_id;
        buf.cell_offset = cell_offset;
        buf.cell_count = 1; // Primary reference cell index
        buf.d_chart_metric_ptr = d_metric_base + cell_offset;
        if (d_density_base) buf.d_chart_density_ptr = d_density_base + cell_offset;
        if (d_velocity_base) buf.d_chart_velocity_ptr = d_velocity_base + cell_offset * 3;

        gpu_buffer_registry_[chart.chart_id] = buf;
    }
}

} // namespace elysia::emulation

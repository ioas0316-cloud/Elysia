#include "emulation/spatiotemporal_pipeline.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "=== Testing SpatiotemporalPipelineStage2 ===" << std::endl;

    size_t dim_x = 8, dim_y = 8, dim_z = 8;
    size_t total_cells = dim_x * dim_y * dim_z;
    float grid_spacing = 0.1f;

    // 1. Initialize MetricFieldEngine, AtlasManager, Pipeline
    elysia::emulation::MetricFieldEngine metric_engine(dim_x, dim_y, dim_z);
    elysia::emulation::AtlasManager atlas_manager(8);
    elysia::emulation::SpatiotemporalPipelineStage2 pipeline(dim_x, dim_y, dim_z, grid_spacing);

    // 2. Setup mock density & velocity buffers
    std::vector<float> density(total_cells, 1.0f);
    std::vector<float> velocity(total_cells * 3, 0.0f);

    // Induce a positive flux divergence / bottleneck at cell (4, 4, 4)
    size_t center_idx = 4 + 4 * dim_x + 4 * dim_x * dim_y;
    // Set divergence flux: +x neighbor (5, 4, 4) has velocity +10.0, -x neighbor (3, 4, 4) has velocity -10.0
    size_t px_idx = 5 + 4 * dim_x + 4 * dim_x * dim_y;
    velocity[px_idx * 3 + 0] = 10.0f;
    size_t nx_idx = 3 + 4 * dim_x + 4 * dim_x * dim_y;
    velocity[nx_idx * 3 + 0] = -10.0f;

    float* d_density = nullptr;
    float* d_velocity = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_density), total_cells * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_velocity), total_cells * 3 * sizeof(float));

    cudaMemcpy(d_density, density.data(), total_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocity, velocity.data(), total_cells * 3 * sizeof(float), cudaMemcpyHostToDevice);

    float eta_stress = 0.1f;
    float split_thresh = 5.0f;
    float merge_thresh = 0.1f;

    std::cout << "[Step 1] Running pipeline.step() - expecting subdivision on high bottleneck..." << std::endl;
    pipeline.step(metric_engine, d_density, d_velocity, atlas_manager, eta_stress, split_thresh, merge_thresh);

    const float* host_map = pipeline.get_host_bottleneck_map();
    assert(host_map != nullptr);

    std::cout << "Max bottleneck measured in host map at center: " << host_map[center_idx] << std::endl;
    assert(host_map[center_idx] > split_thresh);

    const auto& buffers = pipeline.get_gpu_buffer_registry();
    std::cout << "Active GPU buffer bindings count: " << buffers.size() << std::endl;
    assert(!buffers.empty());

    // Verify metric stress expansion occurred on device metric field
    elysia::emulation::MetricTensor3x3 host_metric_sample;
    cudaMemcpy(&host_metric_sample, metric_engine.get_device_metric_ptr() + center_idx, sizeof(elysia::emulation::MetricTensor3x3), cudaMemcpyDeviceToHost);
    std::cout << "Center metric g[0][0] after stress expansion: " << host_metric_sample.g[0][0] << std::endl;
    assert(host_metric_sample.g[0][0] > 1.0f);

    // Cleanup device memory
    cudaFree(d_density);
    cudaFree(d_velocity);

    std::cout << "=== SpatiotemporalPipelineStage2 Test PASSED ===" << std::endl;
    return 0;
}

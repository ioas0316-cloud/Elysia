#include "spatiotemporal_memory.hpp"
#include <cmath>

namespace elysia::emulation {

// Helper: 3x3 determinant
static inline float compute_determinant_3x3(const MetricTensor3x3& m) {
    return m.g[0][0] * (m.g[1][1] * m.g[2][2] - m.g[1][2] * m.g[2][1])
         - m.g[0][1] * (m.g[1][0] * m.g[2][2] - m.g[1][2] * m.g[2][0])
         + m.g[0][2] * (m.g[1][0] * m.g[2][1] - m.g[1][1] * m.g[2][0]);
}

void MetricFieldEngine::launch_compute_flux_divergence_kernel(
    const float* d_density, const float* d_velocity, float* d_bottleneck_out,
    float grid_spacing, cudaStream_t stream) {
    (void)stream;

    auto fetch_weighted_flux = [&](int x, int y, int z) -> std::tuple<float, float, float> {
        x = std::min(std::max(x, 0), static_cast<int>(dim_x_) - 1);
        y = std::min(std::max(y, 0), static_cast<int>(dim_y_) - 1);
        z = std::min(std::max(z, 0), static_cast<int>(dim_z_) - 1);
        size_t idx = x + y * dim_x_ + z * dim_x_ * dim_y_;

        float det_g = fmaxf(compute_determinant_3x3(d_g_mem_field_[idx]), 1e-6f);
        float sqrt_det = std::sqrt(det_g);
        float rho = d_density[idx];

        return {
            sqrt_det * rho * d_velocity[idx * 3 + 0],
            sqrt_det * rho * d_velocity[idx * 3 + 1],
            sqrt_det * rho * d_velocity[idx * 3 + 2]
        };
    };

    for (size_t gz = 0; gz < dim_z_; ++gz) {
        for (size_t gy = 0; gy < dim_y_; ++gy) {
            for (size_t gx = 0; gx < dim_x_; ++gx) {
                size_t center_idx = gx + gy * dim_x_ + gz * dim_x_ * dim_y_;

                float center_det = fmaxf(compute_determinant_3x3(d_g_mem_field_[center_idx]), 1e-6f);
                float inv_sqrt_det_center = 1.0f / std::sqrt(center_det);

                auto [px_x, px_y, px_z] = fetch_weighted_flux(gx + 1, gy, gz);
                auto [nx_x, nx_y, nx_z] = fetch_weighted_flux(gx - 1, gy, gz);
                auto [py_x, py_y, py_z] = fetch_weighted_flux(gx, gy + 1, gz);
                auto [ny_x, ny_y, ny_z] = fetch_weighted_flux(gx, gy - 1, gz);
                auto [pz_x, pz_y, pz_z] = fetch_weighted_flux(gx, gy, gz + 1);
                auto [nz_x, nz_y, nz_z] = fetch_weighted_flux(gx, gy, gz - 1);

                float dJ_dx = (px_x - nx_x) / (2.0f * grid_spacing);
                float dJ_dy = (py_y - ny_y) / (2.0f * grid_spacing);
                float dJ_dz = (pz_z - nz_z) / (2.0f * grid_spacing);

                float riemannian_div = inv_sqrt_det_center * (dJ_dx + dJ_dy + dJ_dz);
                d_bottleneck_out[center_idx] = riemannian_div;
            }
        }
    }
}

void MetricFieldEngine::launch_apply_bottleneck_metric_stress_kernel(
    const float* d_bottleneck_index, float eta_stress_coefficient, cudaStream_t stream) {
    (void)stream;

    for (size_t idx = 0; idx < total_cells_; ++idx) {
        float p_stress = d_bottleneck_index[idx];
        if (p_stress > 0.0f) {
            float stress_delta = eta_stress_coefficient * p_stress;
            d_g_mem_field_[idx].g[0][0] += stress_delta;
            d_g_mem_field_[idx].g[1][1] += stress_delta;
            d_g_mem_field_[idx].g[2][2] += stress_delta;
        }
    }
}

} // namespace elysia::emulation

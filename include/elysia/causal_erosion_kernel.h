#ifndef ELYSIA_CAUSAL_EROSION_KERNEL_H
#define ELYSIA_CAUSAL_EROSION_KERNEL_H

#include <vector>
#include <cmath>
#include <algorithm>

namespace elysia {

struct Point3D {
    float x, y, z;
};

/**
 * Causal Erosion Kernel Solver
 * Provides Tiled Shared Memory Gaussian Causal Erosion calculation and O(1) Spatial Field Rasterization sampling.
 */
class CausalErosionKernelSolver {
private:
    int num_particles_;
    float sigma_;
    float base_k_;
    float erosion_rate_;

    std::vector<Point3D> particles_;
    std::vector<Point3D> well_centers_;
    std::vector<float> well_depths_;
    std::vector<Point3D> forces_;

    // O(1) Spatial Grid Field Parameters
    bool is_phase_transformed_ = false;
    int nc_threshold_ = 100;
    int grid_res_x_ = 32, grid_res_y_ = 32, grid_res_z_ = 32;
    std::vector<float> grid_v_;

public:
    CausalErosionKernelSolver(int num_particles, float sigma = 0.4f, float base_k = 0.2f, float erosion_rate = 0.15f)
        : num_particles_(num_particles), sigma_(sigma), base_k_(base_k), erosion_rate_(erosion_rate)
    {
        particles_.resize(num_particles_, {0.0f, 0.0f, 0.0f});
        forces_.resize(num_particles_, {0.0f, 0.0f, 0.0f});
        grid_v_.resize(grid_res_x_ * grid_res_y_ * grid_res_z_, 0.0f);
    }

    void ErodeAtTrajectory(float x, float y, float z, float depth_multiplier = 1.0f) {
        well_centers_.push_back({x, y, z});
        well_depths_.push_back(erosion_rate_ * depth_multiplier);

        if (!is_phase_transformed_ && static_cast<int>(well_centers_.size()) >= nc_threshold_) {
            RasterizeToGrid();
        }
    }

    void RasterizeToGrid() {
        is_phase_transformed_ = true;
        std::fill(grid_v_.begin(), grid_v_.end(), 0.0f);

        float inv_two_sigma_sq = 1.0f / (2.0f * sigma_ * sigma_);
        float min_b = -5.0f, max_b = 5.0f;
        float dx = (max_b - min_b) / grid_res_x_;
        float dy = (max_b - min_b) / grid_res_y_;
        float dz = (max_b - min_b) / grid_res_z_;

        for (size_t w = 0; w < well_centers_.size(); ++w) {
            float cx = well_centers_[w].x;
            float cy = well_centers_[w].y;
            float cz = well_centers_[w].z;
            float depth = well_depths_[w];

            for (int z = 0; z < grid_res_z_; ++z) {
                float wz = min_b + (z + 0.5f) * dz;
                for (int y = 0; y < grid_res_y_; ++y) {
                    float wy = min_b + (y + 0.5f) * dy;
                    for (int x = 0; x < grid_res_x_; ++x) {
                        float wx = min_b + (x + 0.5f) * dx;

                        float dist_sq = (wx - cx) * (wx - cx) + (wy - cy) * (wy - cy) + (wz - cz) * (wz - cz);
                        int idx = z * (grid_res_x_ * grid_res_y_) + y * grid_res_x_ + x;
                        grid_v_[idx] -= depth * std::exp(-dist_sq * inv_two_sigma_sq);
                    }
                }
            }
        }
    }

    void ComputeForces() {
        float inv_two_sigma_sq = 1.0f / (2.0f * sigma_ * sigma_);
        float inv_sigma_sq = 1.0f / (sigma_ * sigma_);

        for (int i = 0; i < num_particles_; ++i) {
            float px = particles_[i].x;
            float py = particles_[i].y;
            float pz = particles_[i].z;

            float fx = -base_k_ * px;
            float fy = -base_k_ * py;
            float fz = -base_k_ * pz;

            if (!is_phase_transformed_) {
                // Discrete O(N) well iteration
                for (size_t w = 0; w < well_centers_.size(); ++w) {
                    float cx = well_centers_[w].x;
                    float cy = well_centers_[w].y;
                    float cz = well_centers_[w].z;
                    float depth = well_depths_[w];

                    float diff_x = px - cx;
                    float diff_y = py - cy;
                    float diff_z = pz - cz;
                    float dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;

                    float well_v = depth * std::exp(-dist_sq * inv_two_sigma_sq);
                    float coeff = well_v * inv_sigma_sq;

                    fx += coeff * diff_x;
                    fy += coeff * diff_y;
                    fz += coeff * diff_z;
                }
            } else {
                // Continuous O(1) field central difference gradient
                float min_b = -5.0f, max_b = 5.0f;
                float step_x = (max_b - min_b) / grid_res_x_;
                float step_y = (max_b - min_b) / grid_res_y_;
                float step_z = (max_b - min_b) / grid_res_z_;

                int gx = std::clamp(static_cast<int>((px - min_b) / step_x), 1, grid_res_x_ - 2);
                int gy = std::clamp(static_cast<int>((py - min_b) / step_y), 1, grid_res_y_ - 2);
                int gz = std::clamp(static_cast<int>((pz - min_b) / step_z), 1, grid_res_z_ - 2);

                auto get_v = [&](int x, int y, int z) {
                    return grid_v_[z * (grid_res_x_ * grid_res_y_) + y * grid_res_x_ + x];
                };

                float dv_dx = (get_v(gx + 1, gy, gz) - get_v(gx - 1, gy, gz)) / (2.0f * step_x);
                float dv_dy = (get_v(gx, gy + 1, gz) - get_v(gx, gy - 1, gz)) / (2.0f * step_y);
                float dv_dz = (get_v(gx, gy, gz + 1) - get_v(gx, gy, gz - 1)) / (2.0f * step_z);

                fx -= dv_dx;
                fy -= dv_dy;
                fz -= dv_dz;
            }

            forces_[i] = {fx, fy, fz};
        }
    }

    void Step(float dt) {
        ComputeForces();
        for (int i = 0; i < num_particles_; ++i) {
            particles_[i].x += forces_[i].x * dt;
            particles_[i].y += forces_[i].y * dt;
            particles_[i].z += forces_[i].z * dt;
        }
    }

    const float* GetForceBufferPointer() const {
        return reinterpret_cast<const float*>(forces_.data());
    }

    const float* GetParticleBufferPointer() const {
        return reinterpret_cast<const float*>(particles_.data());
    }

    int GetWellCount() const { return static_cast<int>(well_centers_.size()); }
    bool IsPhaseTransformed() const { return is_phase_transformed_; }
};

} // namespace elysia

#endif // ELYSIA_CAUSAL_EROSION_KERNEL_H

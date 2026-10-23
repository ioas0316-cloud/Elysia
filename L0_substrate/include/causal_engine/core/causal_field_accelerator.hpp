#ifndef CAUSAL_ENGINE_CORE_CAUSAL_FIELD_ACCELERATOR_HPP
#define CAUSAL_ENGINE_CORE_CAUSAL_FIELD_ACCELERATOR_HPP

#include <vector>
#include <cmath>
#include <utility>
#include <cstddef>
#include <omp.h>

namespace causal_engine {

/**
 * @brief High-performance contiguous memory Causal Control Point in R^4.
 */
struct ControlPoint {
    double pos[4]{0.0, 0.0, 0.0, 0.0};
    double vel[4]{0.0, 0.0, 0.0, 0.0};
    double weight{1.0};
};

/**
 * @brief OpenMP/SIMD accelerated zero-branching field dynamics engine.
 */
class CausalFieldAccelerator {
public:
    void step_parallel(
        std::vector<ControlPoint>& points,
        const std::vector<std::pair<int, int>>& edges,
        const std::vector<double>& tensions,
        const double telos[4],
        double dt = 0.05,
        double damping = 0.85
    ) {
        int num_points = static_cast<int>(points.size());
        int num_edges = static_cast<int>(edges.size());

        if (num_points == 0) return;

        // Zero-branching force accumulation arrays in contiguous memory
        std::vector<double> forces_x(num_points, 0.0);
        std::vector<double> forces_y(num_points, 0.0);
        std::vector<double> forces_z(num_points, 0.0);
        std::vector<double> forces_w(num_points, 0.0);

        // 1. Internal control point topological tension forces (OpenMP SIMD parallelized)
        #pragma omp parallel for
        for (int e = 0; e < num_edges; ++e) {
            int src = edges[e].first;
            int tgt = edges[e].second;
            if (src < 0 || src >= num_points || tgt < 0 || tgt >= num_points) continue;

            double k = tensions[e];

            double dx = points[tgt].pos[0] - points[src].pos[0];
            double dy = points[tgt].pos[1] - points[src].pos[1];
            double dz = points[tgt].pos[2] - points[src].pos[2];
            double dw = points[tgt].pos[3] - points[src].pos[3];

            #pragma omp atomic
            forces_x[src] += k * dx;
            #pragma omp atomic
            forces_y[src] += k * dy;
            #pragma omp atomic
            forces_z[src] += k * dz;
            #pragma omp atomic
            forces_w[src] += k * dw;

            #pragma omp atomic
            forces_x[tgt] -= k * dx;
            #pragma omp atomic
            forces_y[tgt] -= k * dy;
            #pragma omp atomic
            forces_z[tgt] -= k * dz;
            #pragma omp atomic
            forces_w[tgt] -= k * dw;
        }

        // 2. Telos gravitational potential gradient (-∇E) and physical vector integration
        #pragma omp parallel for
        for (int i = 0; i < num_points; ++i) {
            double telos_fx = points[i].weight * (telos[0] - points[i].pos[0]);
            double telos_fy = points[i].weight * (telos[1] - points[i].pos[1]);
            double telos_fz = points[i].weight * (telos[2] - points[i].pos[2]);
            double telos_fw = points[i].weight * (telos[3] - points[i].pos[3]);

            forces_x[i] += telos_fx;
            forces_y[i] += telos_fy;
            forces_z[i] += telos_fz;
            forces_w[i] += telos_fw;

            // Velocity and position integration (Zero-branching vector integration)
            points[i].vel[0] = points[i].vel[0] * damping + forces_x[i] * dt;
            points[i].vel[1] = points[i].vel[1] * damping + forces_y[i] * dt;
            points[i].vel[2] = points[i].vel[2] * damping + forces_z[i] * dt;
            points[i].vel[3] = points[i].vel[3] * damping + forces_w[i] * dt;

            points[i].pos[0] += points[i].vel[0] * dt;
            points[i].pos[1] += points[i].vel[1] * dt;
            points[i].pos[2] += points[i].vel[2] * dt;
            points[i].pos[3] += points[i].vel[3] * dt;
        }
    }

    double compute_system_energy(
        const std::vector<ControlPoint>& points,
        const std::vector<std::pair<int, int>>& edges,
        const std::vector<double>& tensions,
        const double telos[4]
    ) const {
        int num_points = static_cast<int>(points.size());
        int num_edges = static_cast<int>(edges.size());
        double total_energy = 0.0;

        #pragma omp parallel for reduction(+:total_energy)
        for (int e = 0; e < num_edges; ++e) {
            int src = edges[e].first;
            int tgt = edges[e].second;
            if (src < 0 || src >= num_points || tgt < 0 || tgt >= num_points) continue;

            double k = tensions[e];
            double dx = points[tgt].pos[0] - points[src].pos[0];
            double dy = points[tgt].pos[1] - points[src].pos[1];
            double dz = points[tgt].pos[2] - points[src].pos[2];
            double dw = points[tgt].pos[3] - points[src].pos[3];

            double dist_sq = dx * dx + dy * dy + dz * dz + dw * dw;
            total_energy += 0.5 * k * dist_sq;
        }

        #pragma omp parallel for reduction(+:total_energy)
        for (int i = 0; i < num_points; ++i) {
            double dx = points[i].pos[0] - telos[0];
            double dy = points[i].pos[1] - telos[1];
            double dz = points[i].pos[2] - telos[2];
            double dw = points[i].pos[3] - telos[3];

            double dist_sq = dx * dx + dy * dy + dz * dz + dw * dw;
            total_energy += 0.5 * points[i].weight * dist_sq;
        }

        return total_energy;
    }
};

} // namespace causal_engine

#endif // CAUSAL_ENGINE_CORE_CAUSAL_FIELD_ACCELERATOR_HPP

#ifndef ELYSIA_RELAXATION_SOLVER_H
#define ELYSIA_RELAXATION_SOLVER_H

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <iostream>

namespace elysia {

struct Vector3 {
    float x, y, z;
};

/**
 * Elysia Relaxation Solver
 *
 * Provides zero-copy pointer access to node position arrays for real-time 3D engine integration (60fps/120fps).
 * Implements gradient flow dΨ/dt = -∇V(Ψ) relaxation, impact tension reception, and spontaneous bifurcation mode switching.
 */
class ElysiaRelaxationSolver {
private:
    int num_nodes_;
    std::vector<Vector3> body_nodes_;
    std::vector<Vector3> normal_attractor_;
    std::vector<Vector3> berserk_attractor_;

    std::string current_mode_;
    float accumulated_tension_;
    float bifurcation_threshold_;

public:
    ElysiaRelaxationSolver(int num_nodes = 12, const float* initial_pos_ptr = nullptr)
        : num_nodes_(num_nodes), accumulated_tension_(0.0f), bifurcation_threshold_(5.0f), current_mode_("normal")
    {
        body_nodes_.resize(num_nodes_);
        normal_attractor_.resize(num_nodes_);
        berserk_attractor_.resize(num_nodes_);

        for (int i = 0; i < num_nodes_; ++i) {
            if (initial_pos_ptr != nullptr) {
                body_nodes_[i] = { initial_pos_ptr[i * 3 + 0], initial_pos_ptr[i * 3 + 1], initial_pos_ptr[i * 3 + 2] };
            } else {
                body_nodes_[i] = { static_cast<float>(i % 3 - 1), static_cast<float>(i % 4 - 1.5f), static_cast<float>(i % 2) };
            }

            normal_attractor_[i] = { static_cast<float>(i % 3 - 1), static_cast<float>(i % 4 - 1.5f), static_cast<float>(i % 2) };
            berserk_attractor_[i] = { static_cast<float>(i % 3 - 1) * 2.5f, static_cast<float>(i % 4 - 1.5f) * 2.5f + 1.0f, static_cast<float>(i % 2) * 2.5f };
        }
    }

    void ApplyImpact(int hit_node_index, float fx, float fy, float fz) {
        if (hit_node_index < 0 || hit_node_index >= num_nodes_) return;

        body_nodes_[hit_node_index].x += fx;
        body_nodes_[hit_node_index].y += fy;
        body_nodes_[hit_node_index].z += fz;

        float force_magnitude = std::sqrt(fx * fx + fy * fy + fz * fz);
        accumulated_tension_ += force_magnitude;
    }

    void Step(float dt, float relaxation_rate = 0.4f) {
        // 1. Check bifurcation threshold for mode shift
        if (accumulated_tension_ > bifurcation_threshold_ && current_mode_ == "normal") {
            current_mode_ = "berserk";
            std::cout << "[ElysiaRelaxationSolver] Attractor Field Shifted -> BERSERK MODE\n";
        }

        // 2. Target attractor basin
        const auto& active_attractor = (current_mode_ == "normal") ? normal_attractor_ : berserk_attractor_;

        // 3. Gradient flow dBody/dt = -∇V(Ψ)
        for (int i = 0; i < num_nodes_; ++i) {
            float dx = active_attractor[i].x - body_nodes_[i].x;
            float dy = active_attractor[i].y - body_nodes_[i].y;
            float dz = active_attractor[i].z - body_nodes_[i].z;

            body_nodes_[i].x += relaxation_rate * dx * dt;
            body_nodes_[i].y += relaxation_rate * dy * dt;
            body_nodes_[i].z += relaxation_rate * dz * dt;
        }

        // 4. Tension relaxation
        accumulated_tension_ *= 0.90f;
    }

    const float* GetNodeDataPointer() const {
        return reinterpret_cast<const float*>(body_nodes_.data());
    }

    int GetNumNodes() const { return num_nodes_; }
    float GetAccumulatedTension() const { return accumulated_tension_; }
    std::string GetCurrentMode() const { return current_mode_; }
};

} // namespace elysia

#endif // ELYSIA_RELAXATION_SOLVER_H

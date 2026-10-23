#ifndef CAUSAL_ENGINE_CORE_TOPOLOGICAL_FIELD_2D_HPP
#define CAUSAL_ENGINE_CORE_TOPOLOGICAL_FIELD_2D_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

namespace causal_engine {

/**
 * @brief Topological Field 2D (2D Multi-dimensional Phase Substrate)
 *
 * Supports phase gradient flows, local vorticity (vortex detection as conceptual boundary),
 * coherence gates, and multi-directional zero-scattering energy transport.
 */
struct TopologicalField2D {
    size_t width{0};
    size_t height{0};

    // 1. 2D State Arrays (Flattened for Cache Alignment)
    std::vector<float> phase;          // Local Phase phi(x, y)
    std::vector<float> amplitude;      // Local Energy / Amplitude A(x, y)

    // 2. Spatial Phase Gradient Field (2D Flow Vectors)
    std::vector<float> grad_x;         // d_phi / dx
    std::vector<float> grad_y;         // d_phi / dy

    // 3. Coherence & Topological Barrier Field
    std::vector<float> coherence_gate; // Local Superconducting Gate (0.0 ~ 1.0)
    std::vector<float> vorticity;      // Local Phase Winding / Vortex Density (Demarcation Barrier)

    TopologicalField2D() = default;

    TopologicalField2D(size_t w, size_t h)
        : width(w),
          height(h),
          phase(w * h, 0.0f),
          amplitude(w * h, 0.0f),
          grad_x(w * h, 0.0f),
          grad_y(w * h, 0.0f),
          coherence_gate(w * h, 0.0f),
          vorticity(w * h, 0.0f) {}

    void resize(size_t w, size_t h) {
        width = w;
        height = h;
        size_t total = w * h;
        phase.assign(total, 0.0f);
        amplitude.assign(total, 0.0f);
        grad_x.assign(total, 0.0f);
        grad_y.assign(total, 0.0f);
        coherence_gate.assign(total, 0.0f);
        vorticity.assign(total, 0.0f);
    }
};

/**
 * @brief Step Multi-dimensional Topological Field Dynamics
 */
inline void step_multidim_topological_transport(
    TopologicalField2D& field,
    float phase_lock_thresh = 0.1f,
    float dt = 0.1f
) {
    const size_t w = field.width;
    const size_t h = field.height;

    if (w < 3 || h < 3) return;

    // 1. Spatial Phase Gradient Calculation
    #pragma omp parallel for collapse(2)
    for (size_t y = 1; y < h - 1; ++y) {
        for (size_t x = 1; x < w - 1; ++x) {
            size_t idx = y * w + x;

            float dphi_dx = (field.phase[idx + 1] - field.phase[idx - 1]) * 0.5f;
            float dphi_dy = (field.phase[idx + w] - field.phase[idx - w]) * 0.5f;

            field.grad_x[idx] = dphi_dx;
            field.grad_y[idx] = dphi_dy;
        }
    }

    // 2. Vorticity (Curl) and Coherence Gate Calculation (After implicit barrier)
    #pragma omp parallel for collapse(2)
    for (size_t y = 1; y < h - 1; ++y) {
        for (size_t x = 1; x < w - 1; ++x) {
            size_t idx = y * w + x;

            float curl = (field.grad_y[idx + 1] - field.grad_y[idx - 1]) -
                         (field.grad_x[idx + w] - field.grad_x[idx - w]);
            field.vorticity[idx] = std::abs(curl);

            float dphi_dx = field.grad_x[idx];
            float dphi_dy = field.grad_y[idx];
            float grad_mag = std::sqrt(dphi_dx * dphi_dx + dphi_dy * dphi_dy);

            if (grad_mag < phase_lock_thresh) {
                field.coherence_gate[idx] = 1.0f; // Friction-free 2D transport channel
            } else {
                field.coherence_gate[idx] *= 0.85f; // Dissipative barrier formation
            }
        }
    }

    // 3. Isotropic & Directional Wave Energy Transport
    std::vector<float> next_amplitude = field.amplitude;

    #pragma omp parallel for collapse(2)
    for (size_t y = 1; y < h - 1; ++y) {
        for (size_t x = 1; x < w - 1; ++x) {
            size_t idx = y * w + x;

            float active_coherence = field.coherence_gate[idx];
            float flow = field.amplitude[idx] * 0.25f * active_coherence;

            float dphi_dx = field.grad_x[idx];
            float dphi_dy = field.grad_y[idx];

            float flow_x = flow * std::max(0.0f, 1.0f - std::abs(dphi_dx));
            float flow_y = flow * std::max(0.0f, 1.0f - std::abs(dphi_dy));

            #pragma omp atomic
            next_amplitude[idx + 1] += flow_x;

            #pragma omp atomic
            next_amplitude[idx + w] += flow_y;

            #pragma omp atomic
            next_amplitude[idx] -= (flow_x + flow_y);
        }
    }

    // Numerical Clamping and Update for all elements
    #pragma omp parallel for
    for (size_t i = 0; i < w * h; ++i) {
        if (std::isnan(next_amplitude[i])) {
            next_amplitude[i] = 0.0f;
        } else {
            next_amplitude[i] = std::min(1e5f, std::max(0.0f, next_amplitude[i]));
        }
    }

    field.amplitude = std::move(next_amplitude);
}

} // namespace causal_engine

#endif // CAUSAL_ENGINE_CORE_TOPOLOGICAL_FIELD_2D_HPP

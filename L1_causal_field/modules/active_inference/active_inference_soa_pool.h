#ifndef ACTIVE_INFERENCE_SOA_POOL_H
#define ACTIVE_INFERENCE_SOA_POOL_H

#include "active_inference_config.h"
#include <vector>
#include <cmath>
#include <random>

namespace active_inference {

// Structure of Arrays (SoA) Pool for high-performance multi-agent Continuous Active Inference.
// Flat contiguous std::vector memory buffers ensure SIMD and cache locality friendly iteration.
class ActiveInferenceSoAPool {
public:
    size_t count = 0;

    // 1D Active Inference agent state arrays
    std::vector<float> x;       // True physical state
    std::vector<float> a;       // Action / force command
    std::vector<float> mu;      // Agent internal belief
    std::vector<float> mu_d;    // Target / desired state
    std::vector<float> e_y;     // Sensory prediction error
    std::vector<float> e_p;     // Prior / target prediction error

    // 2nd-Order Active Inference agent state arrays (generalized coordinates)
    std::vector<float> v;       // True physical velocity
    std::vector<float> mu0;     // Position belief
    std::vector<float> mu1;     // Velocity belief
    std::vector<float> mu_d0;   // Target position
    std::vector<float> mu_d1;   // Target velocity
    std::vector<float> e_y0;    // Position sensory prediction error
    std::vector<float> e_y1;    // Velocity sensory prediction error
    std::vector<float> e_p0;    // Position target error
    std::vector<float> e_p1;    // Velocity target error

    ActiveInferenceSoAPool() = default;

    explicit ActiveInferenceSoAPool(size_t num_agents) {
        resize(num_agents);
    }

    void resize(size_t num_agents) {
        count = num_agents;
        x.resize(num_agents, 0.0f);
        a.resize(num_agents, 0.0f);
        mu.resize(num_agents, 0.0f);
        mu_d.resize(num_agents, 5.0f);
        e_y.resize(num_agents, 0.0f);
        e_p.resize(num_agents, 0.0f);

        v.resize(num_agents, 0.0f);
        mu0.resize(num_agents, 0.0f);
        mu1.resize(num_agents, 0.0f);
        mu_d0.resize(num_agents, 10.0f);
        mu_d1.resize(num_agents, 0.0f);
        e_y0.resize(num_agents, 0.0f);
        e_y1.resize(num_agents, 0.0f);
        e_p0.resize(num_agents, 0.0f);
        e_p1.resize(num_agents, 0.0f);
    }

    void reset_agent_1d(size_t index, float init_x = 0.0f, float init_mu = 0.0f, float target_mu_d = 5.0f) {
        if (index >= count) return;
        x[index] = init_x;
        a[index] = 0.0f;
        mu[index] = init_mu;
        mu_d[index] = target_mu_d;
        e_y[index] = 0.0f;
        e_p[index] = 0.0f;
    }

    void reset_agent_2d(size_t index, float init_x = 0.0f, float init_v = 0.0f, float target_pos = 10.0f, float target_vel = 0.0f) {
        if (index >= count) return;
        x[index] = init_x;
        v[index] = init_v;
        a[index] = 0.0f;
        mu0[index] = init_x;
        mu1[index] = init_v;
        mu_d0[index] = target_pos;
        mu_d1[index] = target_vel;
        e_y0[index] = 0.0f;
        e_y1[index] = 0.0f;
        e_p0[index] = 0.0f;
        e_p1[index] = 0.0f;
    }
};

class ActiveInferenceEngine {
private:
    ActiveInferenceConfig config_;
    unsigned int base_seed_{42};

public:
    ActiveInferenceEngine() = default;
    explicit ActiveInferenceEngine(const ActiveInferenceConfig& config) : config_(config) {}

    const ActiveInferenceConfig& get_config() const { return config_; }
    void set_config(const ActiveInferenceConfig& config) { config_ = config; }
    void set_seed(unsigned int seed) { base_seed_ = seed; }

    // Performs 1 Step Euler Integration across all 1D Agents in the SoA Pool
    void step_1d(ActiveInferenceSoAPool& pool, float noise_std = 0.01f) {
        #pragma omp parallel if(pool.count > 1000)
        {
            std::mt19937 thread_rng(base_seed_ + static_cast<unsigned int>(pool.count));
            std::normal_distribution<float> dist(0.0f, noise_std);

            #pragma omp for
            for (size_t i = 0; i < pool.count; ++i) {
                float noise = (noise_std > 0.0f) ? dist(thread_rng) : 0.0f;
                float y = pool.x[i] + noise;

                // (1) Prediction Errors
                pool.e_y[i] = config_.pi_y * (y - pool.mu[i]);
                pool.e_p[i] = config_.pi_p * (pool.mu_d[i] - pool.mu[i]);

                // (2) Perception Euler Update
                float d_mu = pool.e_y[i] + pool.e_p[i];
                d_mu = config_.clamp_deriv(d_mu);
                pool.mu[i] = config_.clamp_state(pool.mu[i] + config_.dt * config_.lr_mu * d_mu);

                // (3) Action Euler Update (Reflex Arc)
                float d_a = config_.pi_y * (pool.mu[i] - y);
                d_a = config_.clamp_deriv(d_a);
                pool.a[i] = config_.clamp_state(pool.a[i] + config_.dt * config_.lr_a * d_a);

                // (4) Physical Environment Update (dx/dt = -alpha * x + a)
                float dx = -config_.alpha * pool.x[i] + pool.a[i];
                dx = config_.clamp_deriv(dx);
                pool.x[i] = config_.clamp_state(pool.x[i] + config_.dt * dx);
            }
        }
    }

    // Performs 1 Step Euler Integration across all 2nd-order (2D) Agents in the SoA Pool
    void step_2d(ActiveInferenceSoAPool& pool, float noise_std = 0.02f) {
        #pragma omp parallel if(pool.count > 1000)
        {
            std::mt19937 thread_rng(base_seed_ + static_cast<unsigned int>(pool.count));
            std::normal_distribution<float> dist(0.0f, noise_std);

            #pragma omp for
            for (size_t i = 0; i < pool.count; ++i) {
                float n0 = (noise_std > 0.0f) ? dist(thread_rng) : 0.0f;
                float n1 = (noise_std > 0.0f) ? dist(thread_rng) : 0.0f;

                float y0 = pool.x[i] + n0;
                float y1 = pool.v[i] + n1;

                // (1) Prediction Errors
                pool.e_y0[i] = config_.pi_y0 * (y0 - pool.mu0[i]);
                pool.e_y1[i] = config_.pi_y1 * (y1 - pool.mu1[i]);

                pool.e_p0[i] = config_.pi_p0 * (pool.mu_d0[i] - pool.mu0[i]);
                pool.e_p1[i] = config_.pi_p1 * (pool.mu_d1[i] - pool.mu1[i]);

                // (2) Perception Update with Kinematic Coupling (d_mu0 = mu1 + ...)
                float d_mu0 = pool.mu1[i] + config_.lr_mu0 * (pool.e_y0[i] + pool.e_p0[i]);
                float d_mu1 = config_.lr_mu1 * (pool.e_y1[i] + pool.e_p1[i]);

                d_mu0 = config_.clamp_deriv(d_mu0);
                d_mu1 = config_.clamp_deriv(d_mu1);

                pool.mu0[i] = config_.clamp_state(pool.mu0[i] + config_.dt * d_mu0);
                pool.mu1[i] = config_.clamp_state(pool.mu1[i] + config_.dt * d_mu1);

                // (3) Action Update driven by velocity and position prediction errors
                float d_a = config_.lr_a_2d * (config_.pi_y1 * (pool.mu1[i] - y1) + config_.pi_y0 * (pool.mu0[i] - y0));
                d_a = config_.clamp_deriv(d_a);
                pool.a[i] = config_.clamp_state(pool.a[i] + config_.dt * d_a);

                // (4) 2nd Order Physical System Update
                float dv = (pool.a[i] - config_.gamma * pool.v[i]) / config_.mass;
                float dx = pool.v[i];

                dv = config_.clamp_deriv(dv);
                dx = config_.clamp_deriv(dx);

                pool.v[i] = config_.clamp_state(pool.v[i] + config_.dt * dv);
                pool.x[i] = config_.clamp_state(pool.x[i] + config_.dt * dx);
            }
        }
    }
};

} // namespace active_inference

#endif // ACTIVE_INFERENCE_SOA_POOL_H

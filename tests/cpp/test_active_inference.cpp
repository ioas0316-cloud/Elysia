#include "../../modules/active_inference/active_inference_config.h"
#include "../../modules/active_inference/active_inference_soa_pool.h"
#include <iostream>
#include <cassert>
#include <cmath>

void test_1d_convergence() {
    std::cout << "[Test] Running 1D Active Inference Convergence Test..." << std::endl;
    active_inference::ActiveInferenceConfig config;
    config.dt = 0.005f;
    config.pi_y = 10.0f;
    config.pi_p = 2.0f;
    config.lr_mu = 2.0f;
    config.lr_a = 5.0f;
    config.alpha = 1.0f;

    active_inference::ActiveInferenceEngine engine(config);
    engine.set_seed(123);

    active_inference::ActiveInferenceSoAPool pool(1);
    pool.reset_agent_1d(0, 0.0f, 0.0f, 5.0f);

    int steps = static_cast<int>(10.0f / config.dt); // 10 seconds
    for (int t = 0; t < steps; ++t) {
        engine.step_1d(pool, 0.001f); // low noise for test determinism
    }

    std::cout << "  Final True State x[0]: " << pool.x[0] << std::endl;
    std::cout << "  Final Belief mu[0]: " << pool.mu[0] << std::endl;
    std::cout << "  Final Action a[0]: " << pool.a[0] << std::endl;

    // Check convergence to target 5.0 within tolerance
    assert(std::abs(pool.x[0] - 5.0f) < 0.2f && "1D State x failed to converge near target 5.0");
    assert(std::abs(pool.mu[0] - 5.0f) < 0.2f && "1D Belief mu failed to converge near target 5.0");
    std::cout << "[Test 1D] PASS!" << std::endl;
}

void test_2d_convergence() {
    std::cout << "[Test] Running 2nd-Order Active Inference Convergence Test..." << std::endl;
    active_inference::ActiveInferenceConfig config;
    config.dt = 0.005f;
    config.mass = 1.0f;
    config.gamma = 2.0f;
    config.pi_y0 = 10.0f;
    config.pi_y1 = 5.0f;
    config.pi_p0 = 3.0f;
    config.pi_p1 = 2.0f;
    config.lr_mu0 = 2.0f;
    config.lr_mu1 = 2.0f;
    config.lr_a_2d = 10.0f;

    active_inference::ActiveInferenceEngine engine(config);
    engine.set_seed(456);

    active_inference::ActiveInferenceSoAPool pool(1);
    pool.reset_agent_2d(0, 0.0f, 0.0f, 10.0f, 0.0f); // Target pos = 10.0, target vel = 0.0

    int steps = static_cast<int>(10.0f / config.dt); // 10 seconds
    for (int t = 0; t < steps; ++t) {
        engine.step_2d(pool, 0.001f);
    }

    std::cout << "  Final True Position x[0]: " << pool.x[0] << std::endl;
    std::cout << "  Final True Velocity v[0]: " << pool.v[0] << std::endl;
    std::cout << "  Final Position Belief mu0[0]: " << pool.mu0[0] << std::endl;
    std::cout << "  Final Velocity Belief mu1[0]: " << pool.mu1[0] << std::endl;

    // Check convergence: target pos ~ 10.0, velocity ~ 0.0
    assert(std::abs(pool.x[0] - 10.0f) < 0.5f && "2D Position x failed to converge near target 10.0");
    assert(std::abs(pool.v[0] - 0.0f) < 0.2f && "2D Velocity v failed to settle near target 0.0");
    std::cout << "[Test 2D] PASS!" << std::endl;
}

void test_clamping() {
    std::cout << "[Test] Running Clamping Boundary Test..." << std::endl;
    active_inference::ActiveInferenceConfig config;
    config.dt = 0.1f;
    config.pi_y = 1000.0f; // Extreme precision to force huge derivatives
    config.lr_a = 500.0f;
    config.enable_clamping = true;
    config.min_state = -10.0f;
    config.max_state = 10.0f;
    config.min_deriv = -5.0f;
    config.max_deriv = 5.0f;

    active_inference::ActiveInferenceEngine engine(config);
    active_inference::ActiveInferenceSoAPool pool(1);
    pool.x[0] = 100.0f; // Far away state

    engine.step_1d(pool, 0.0f);

    assert(pool.a[0] <= config.max_state && pool.a[0] >= config.min_state && "Action exceeded state clamp bounds");
    assert(pool.mu[0] <= config.max_state && pool.mu[0] >= config.min_state && "Belief exceeded state clamp bounds");
    std::cout << "[Test Clamping] PASS!" << std::endl;
}

void test_soa_multi_agent() {
    std::cout << "[Test] Running SoA Multi-Agent Scale Test (1000 agents)..." << std::endl;
    active_inference::ActiveInferenceConfig config;
    active_inference::ActiveInferenceEngine engine(config);

    size_t N = 1000;
    active_inference::ActiveInferenceSoAPool pool(N);

    for (size_t i = 0; i < N; ++i) {
        pool.reset_agent_1d(i, 0.0f, 0.0f, 5.0f);
    }

    for (int t = 0; t < 100; ++t) {
        engine.step_1d(pool, 0.01f);
    }

    assert(pool.x.size() == N);
    assert(pool.a.size() == N);
    assert(pool.mu.size() == N);
    std::cout << "[Test SoA Multi-Agent] PASS!" << std::endl;
}

int main() {
    try {
        test_1d_convergence();
        test_2d_convergence();
        test_clamping();
        test_soa_multi_agent();
        std::cout << "\n==========================================" << std::endl;
        std::cout << " ALL ACTIVE INFERENCE C++ TESTS PASSED! " << std::endl;
        std::cout << "==========================================" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

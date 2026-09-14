#include <iostream>
#include <cassert>
#include <memory>
#include "causal_engine/core/meta_causal_map.hpp"

using namespace causal_engine;

void test_mechanism_relaxation() {
    std::cout << "\n=== Test 1: Mechanism Structural Relaxation ===\n";
    auto bound_mech = std::make_shared<DifferentialBoundMechanism>("bound_1", 2.0f);
    bound_mech->state[0] = 10.0f;
    bound_mech->state[1] = 0.0f;

    float initial_err = bound_mech->evaluate_residual();
    std::cout << "Initial bound residual: " << initial_err << " (state: " << bound_mech->state[0] << ", " << bound_mech->state[1] << ")\n";
    assert(initial_err > 0.0f);

    for (int i = 0; i < 50; ++i) {
        bound_mech->project_structural_relaxation(1.0f);
        bound_mech->apply_deltas();
    }

    float final_err = bound_mech->evaluate_residual();
    float diff = std::abs(bound_mech->state[0] - bound_mech->state[1]);
    std::cout << "Final bound residual: " << final_err << " (diff: " << diff << ", target: 2.0)\n";
    assert(final_err < 1e-3f);
    assert(diff <= 2.01f);
    std::cout << "  -> Mechanism relaxation test passed.\n";
}

void test_meta_causal_ecosystem() {
    std::cout << "\n=== Test 2: Meta-Causal Ecosystem Convergence & Introspection ===\n";
    MetaCausalEngine engine;

    auto bound = std::make_shared<DifferentialBoundMechanism>("mech_bound", 1.5f);
    bound->state[0] = 8.0f;
    bound->state[1] = 2.0f;

    auto harmonic = std::make_shared<HarmonicConservationMechanism>("mech_harmonic", 10.0f);
    harmonic->state[0] = 6.0f;
    harmonic->state[1] = 5.0f;
    harmonic->state[2] = 4.0f;

    engine.add_mechanism(bound);
    engine.add_mechanism(harmonic);

    // Add meta-causal coupling: bound residual shifts harmonic target parameter
    engine.add_binding("mech_bound", "mech_harmonic", 0.2f);

    float total_initial = engine.compute_total_residual();
    std::cout << "Initial Total Ecosystem Residual: " << total_initial << "\n";
    assert(total_initial > 0.0f);

    int steps = engine.step_convergence(100, 1e-3f, 0.5f);
    float total_final = engine.compute_total_residual();
    std::cout << "Converged in " << steps << " iterations. Final Total Residual: " << total_final << "\n";
    assert(total_final < total_initial);

    auto intro = engine.introspect_causal_contributions();
    std::cout << "Causal Contribution Introspection:\n";
    for (const auto& kv : intro) {
        std::cout << "  - Mechanism [" << kv.first << "]: " << (kv.second * 100.0f) << "%\n";
    }

    std::cout << "  -> Meta-Causal ecosystem convergence test passed.\n";
}

int main() {
    std::cout << "Running Meta-Causal Map C++ Tests...\n";
    test_mechanism_relaxation();
    test_meta_causal_ecosystem();
    std::cout << "\nAll Meta-Causal C++ Tests Passed Successfully!\n";
    return 0;
}

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include "../../modules/causal_topology/preisach_tensor_field.h"

void TestHysteresisSwitchingAndRemanence() {
    std::cout << "[Test 1] Hysteresis Switching & Remanence (S_r) Freezing... ";

    size_t num_nodes = 16;
    size_t steps = 8;
    PreisachTensorFieldSoA field(num_nodes, steps);

    // Initial state: u = 0 -> All hysterons initialized OFF by default in bitmask
    std::fill(field.input_signals.begin(), field.input_signals.end(), 0.0f);
    UpdatePreisachTensorField(field);
    float sr_initial = field.remanence_states[0];

    // Excite: u = +1.0 (switch ON all hysterons with alpha <= 1.0)
    std::fill(field.input_signals.begin(), field.input_signals.end(), 1.0f);
    UpdatePreisachTensorField(field);
    float sr_excited = field.remanence_states[0];
    assert(sr_excited > sr_initial && "Remanence should increase when excited!");

    // Return to u = 0 -> S_r must NOT reset to sr_initial; it remains frozen positive (Remanence S_r > 0)
    std::fill(field.input_signals.begin(), field.input_signals.end(), 0.0f);
    UpdatePreisachTensorField(field);
    float sr_zero_after_excitation = field.remanence_states[0];

    assert(sr_zero_after_excitation > sr_initial && "Remanence state S_r must freeze past input history at u = 0!");
    assert(sr_zero_after_excitation > 0.0f && "Remanence state S_r must remain positive at zero input!");

    // Negative excitation: u = -1.0 (switch OFF hysterons with beta >= -1.0)
    std::fill(field.input_signals.begin(), field.input_signals.end(), -1.0f);
    UpdatePreisachTensorField(field);
    float sr_negative = field.remanence_states[0];
    assert(sr_negative < sr_zero_after_excitation && "Remanence should drop under negative input!");

    std::cout << "PASSED! (Initial S_r: " << sr_initial << " -> Excited S_r: " << sr_excited
              << " -> Frozen Remanence S_r at u=0: " << sr_zero_after_excitation
              << " -> Negative S_r: " << sr_negative << ")\n";
}

void TestDynamicDensityTuning() {
    std::cout << "[Test 2] Dynamic Density Tuning (Invariance Condensation vs Tension Unlocking)... ";

    size_t num_nodes = 8;
    size_t steps = 8;
    PreisachTensorFieldSoA field(num_nodes, steps);

    std::vector<float> do_interventions(num_nodes, 0.5f);
    float target_alpha = 0.5f;
    float target_beta  = -0.5f;

    // Find hysteron closest to (0.5, -0.5)
    size_t target_h = 0;
    float min_dist = 1e9f;
    for (size_t h = 0; h < field.num_hysterons; ++h) {
        float da = field.alpha_grid[h] - target_alpha;
        float db = field.beta_grid[h] - target_beta;
        float d = da * da + db * db;
        if (d < min_dist) {
            min_dist = d;
            target_h = h;
        }
    }

    float initial_mu = field.density_weights[target_h];

    // 1. Invariance Scenario: y_obs == y_pred -> Tension = 0, High Invariance -> Condensation (+)
    std::vector<float> observed_outputs(num_nodes, 1.0f);
    std::vector<float> predicted_outputs(num_nodes, 1.0f);

    UpdateCausalTensionAndDensity(field, do_interventions, observed_outputs, predicted_outputs, 0.1f);
    float mu_after_condensation = field.density_weights[target_h];

    assert(mu_after_condensation > initial_mu && "High invariance must induce density condensation (+)");

    // 2. High Tension Scenario: y_obs != y_pred -> Tension high -> Unlocking (-)
    std::vector<float> conflicting_outputs(num_nodes, 5.0f); // High error
    UpdateCausalTensionAndDensity(field, do_interventions, conflicting_outputs, predicted_outputs, 0.5f);
    float mu_after_unlocking = field.density_weights[target_h];

    assert(mu_after_unlocking < mu_after_condensation && "High tension must induce density unlocking (-)");

    std::cout << "PASSED! (Initial mu: " << initial_mu << " -> Condensed mu: " << mu_after_condensation
              << " -> Unlocked mu: " << mu_after_unlocking << ")\n";
}

void TestAttractorExtractionAndBacktracing() {
    std::cout << "[Test 3] Attractor Extraction & Backward Path Tracing... ";

    size_t num_nodes = 16;
    size_t steps = 8;
    PreisachTensorFieldSoA field(num_nodes, steps);

    // Boost density for certain hysterons to create attractor peaks
    if (field.num_hysterons > 4) {
        field.density_weights[0] = 0.8f;
        field.density_weights[2] = 0.9f;
        field.density_weights[4] = 0.85f;
    }

    std::fill(field.input_signals.begin(), field.input_signals.end(), 0.5f);
    UpdatePreisachTensorField(field);

    AttractorExtractionLayer extractor;
    std::vector<MacroSymbolNode> nodes;
    std::vector<CausalEdge> edges;
    extractor.ExtractCausalGraph(field, nodes, edges, 0.7f, 2.0f);

    assert(!nodes.empty() && "Attractor extraction should identify macro symbol nodes!");

    CausalBacktracer backtracer;
    uint32_t start_id = 0;
    uint32_t goal_id = static_cast<uint32_t>(nodes.size() - 1);

    std::vector<uint32_t> path = backtracer.TraceMinimalImpedancePath(goal_id, start_id, nodes, edges);
    assert(!path.empty() && "Backtracer should produce valid trajectory!");
    assert(path.front() == start_id && path.back() == goal_id && "Path must connect start to goal!");

    std::cout << "PASSED! (Extracted " << nodes.size() << " nodes, " << edges.size()
              << " edges. Path length: " << path.size() << ")\n";
}

void TestBiDirectionalClosedLoop() {
    std::cout << "[Test 4] Bi-directional Closed-Loop Feedback Control... ";

    size_t num_nodes = 16;
    size_t steps = 8;
    PreisachTensorFieldSoA field(num_nodes, steps);

    std::fill(field.input_signals.begin(), field.input_signals.end(), 0.8f);
    UpdatePreisachTensorField(field);

    AttractorExtractionLayer extractor;
    std::vector<MacroSymbolNode> nodes;
    std::vector<CausalEdge> edges;
    extractor.ExtractCausalGraph(field, nodes, edges, 0.4f, 2.0f);

    assert(!nodes.empty() && "Nodes required for closed loop test");

    std::vector<uint32_t> trajectory = {0, static_cast<uint32_t>(nodes.size() - 1)};

    // Disrupt field to trigger tension
    std::fill(field.input_signals.begin(), field.input_signals.end(), -0.9f);
    UpdatePreisachTensorField(field);

    ClosedLoopCausalEngine loop_engine;
    bool adapted = loop_engine.ExecuteAndAdaptTrajectory(trajectory, nodes, field, 0.1f, 0.02f);

    assert(adapted && "Closed-loop engine must adapt upon unexpected tension!");

    std::cout << "PASSED!\n";
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "   Running Preisach Tensor Field C++ Test Suite\n";
    std::cout << "========================================================\n";

    TestHysteresisSwitchingAndRemanence();
    TestDynamicDensityTuning();
    TestAttractorExtractionAndBacktracing();
    TestBiDirectionalClosedLoop();

    std::cout << "========================================================\n";
    std::cout << "   ALL PREISACH C++ TESTS PASSED SUCCESSFULLY!\n";
    std::cout << "========================================================\n";
    return 0;
}

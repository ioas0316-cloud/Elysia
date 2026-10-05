#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

#include "AutonomicStateController.hpp"
#include "VramAdaptiveController.hpp"
#include "CausalPhaseTensor.hpp"
#include "ElysiaAutonomicCausalPipeline.hpp"
#include "CausalGraphExtractor.hpp"
#include "MitoticCausalTensorSpace.hpp"
#include "FractalCausalTree.hpp"
#include "CausalScaleVisualizer.hpp"
#include "MetacognitiveFeedbackPipeline.hpp"
#include "LockFreeCausalMemoryPool.cuh"
#include "CausalDataNode.hpp"
#include "CausalMeaningEvaluator.hpp"
#include "AutonomicFrameSelector.hpp"

void test_autonomic_state_controller() {
    std::cout << "[TEST] AutonomicStateController...\n";
    AutonomicStateController controller;

    assert(controller.get_mode() == AutonomicMode::Parasympathetic);
    assert(!controller.is_parasympathetic() == false);

    EngineMetrics high_error{0.80f, 0.50f, 0.20f};
    controller.update(high_error);
    assert(controller.get_mode() == AutonomicMode::Sympathetic);
    assert(controller.get_ach_level() == 1.0f);

    EngineMetrics low_error{0.05f, 0.90f, 0.10f};
    controller.update(low_error);
    assert(controller.get_mode() == AutonomicMode::Parasympathetic);
    assert(controller.get_ach_level() == 0.02f);

    std::cout << "  -> PASSED\n";
}

void test_vram_adaptive_controller() {
    std::cout << "[TEST] VramAdaptiveController...\n";
    VramAdaptiveController vram_ctrl;

    vram_ctrl.set_simulated_free_ratio(0.60f); // Safe
    float t_gentle = vram_ctrl.get_dynamic_threshold();
    assert(std::abs(t_gentle - 0.05f) < 1e-4f);

    vram_ctrl.set_simulated_free_ratio(0.05f); // Critical
    float t_aggressive = vram_ctrl.get_dynamic_threshold();
    assert(std::abs(t_aggressive - 0.45f) < 1e-4f);

    std::cout << "  -> PASSED\n";
}

void test_causal_phase_tensor_and_singularities() {
    std::cout << "[TEST] CausalPhaseTensor and Singularities...\n";
    size_t N = 16;
    CausalPhaseTensor tensor(N);

    assert(tensor.get_current_space() == PhaseSpace::Real_1D);

    tensor.evaluate_phase_transition(0.85f, 0.10f);
    assert(tensor.get_current_space() == PhaseSpace::Complex_2D);

    tensor.evaluate_phase_transition(0.85f, 0.90f);
    assert(tensor.get_current_space() == PhaseSpace::Clifford_3D);

    tensor.inject_singularity_for_test(3);
    tensor.resolve_singularities(nullptr);

    auto nodes = CausalGraphExtractor::extract_topological_nodes(
        tensor.get_rotors_device_ptr(),
        tensor.get_curvature_device_ptr(),
        N,
        tensor.get_current_space()
    );

    assert(nodes.size() == 1);
    assert(nodes[0].index == 3);
    assert(nodes[0].curvature == 1.0f);

    std::cout << "  -> PASSED\n";
}

void test_mitosis_and_fractal_tree() {
    std::cout << "[TEST] Mitotic Tensor Space & Fractal Tree...\n";
    MitoticCausalTensorSpace tensor_space(4);
    assert(tensor_space.get_active_nodes() == 4);

    std::vector<float> curvatures = {0.1f, 0.95f, 0.2f, 0.88f};
    tensor_space.inject_mock_curvature(curvatures);
    tensor_space.execute_mitotic_division(0.80f, nullptr);

    assert(tensor_space.get_active_nodes() == 6);
    assert(tensor_space.get_scale_level() == 1);

    FractalCausalTree tree;
    float2 c1 = make_float2(0.7071f, 0.7071f);
    float2 c2 = make_float2(0.7071f, -0.7071f);
    auto [id1, id2] = tree.register_mitosis(0, c1, c2);
    assert(id1 != -1 && id2 != -1);
    assert(tree.get_node_table().size() == 3);

    MetacognitiveFeedbackPipeline feedback;
    AutonomicStateController auto_ctrl;
    feedback.process_bottom_up_feedback(tree, auto_ctrl);

    std::string json_str = CausalScaleVisualizer::export_to_json_graph(tree, PhaseSpace::Complex_2D);
    assert(!json_str.empty());

    std::cout << "  -> PASSED\n";
}

void test_meaning_and_frame_selection() {
    std::cout << "[TEST] Causal Data Node & Meaning Evaluator...\n";
    CausalDataNode data_node(1004, 1.0f, "Singularity_Event");
    AutonomicStateController auto_ctrl;
    AutonomicFrameSelector selector(0.80f);

    auto reports = CausalMeaningEvaluator::evaluate_data_across_frames(data_node, 0.0f, true);
    assert(reports.size() == 3);
    assert(reports[0].is_valid_reading == false);
    assert(reports[2].is_valid_reading == true);

    selector.auto_select_optimal_frame(data_node, reports, auto_ctrl);
    assert(data_node.get_frame().phase_space_type == 2);

    std::cout << "  -> PASSED\n";
}

int main() {
    std::cout << "===================================================\n";
    std::cout << " RUNNING ELYSIA AUTONOMIC CAUSAL PIPELINE C++ TESTS\n";
    std::cout << "===================================================\n\n";

    test_autonomic_state_controller();
    test_vram_adaptive_controller();
    test_causal_phase_tensor_and_singularities();
    test_mitosis_and_fractal_tree();
    test_meaning_and_frame_selection();

    std::cout << "\nALL C++ INTEGRATION TESTS PASSED SUCCESSFULLY!\n";
    return 0;
}

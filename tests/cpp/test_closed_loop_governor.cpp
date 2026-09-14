#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <thread>

#include "causal_engine/feedback/closed_loop.hpp"
#include "causal_engine/core/causal_runtime.hpp"
#include "causal_engine/hal/memory_pool.hpp"
#include "causal_engine/compute/gpu_simulator.hpp"

using namespace causal_engine;

void test_protocol_boundary_and_fracture_diagnostics() {
    std::cout << "\n=== Running Protocol Boundary & Structural Fracture Diagnostics Test ===\n";

    core::ProtocolBoundaryLedger boundary_ledger;

    // Register Protocol A (CPU Event Ingestion) & Protocol B (GPU Compute Buffer)
    core::ProtocolBoundary proto_a;
    proto_a.protocol_id = "PROTO_CPU_EVENT_INGEST";
    proto_a.min_state_bound = 0.0f;
    proto_a.max_state_bound = 50.0f;
    proto_a.max_latency_ms = 16.67f;

    core::ProtocolBoundary proto_b;
    proto_b.protocol_id = "PROTO_GPU_COLLAPSE_BUFFER";
    proto_b.min_state_bound = 0.0f;
    proto_b.max_state_bound = 100.0f;
    proto_b.max_latency_ms = 10.0f;

    boundary_ledger.register_protocol_boundary(proto_a);
    boundary_ledger.register_protocol_boundary(proto_b);

    // Test Case 1: Normal aligned boundaries
    auto diag_normal = boundary_ledger.diagnose_boundary_collision(
        "PROTO_CPU_EVENT_INGEST", 25.0f,
        "PROTO_GPU_COLLAPSE_BUFFER", 25.0f,
        12.0f);
    assert(!diag_normal.fracture_detected);
    std::cout << "  - Normal Alignment: " << diag_normal.cause_description << "\n";

    // Test Case 2: Value domain boundary overflow
    auto diag_overflow = boundary_ledger.diagnose_boundary_collision(
        "PROTO_CPU_EVENT_INGEST", 75.0f, // > max_state_bound 50.0
        "PROTO_GPU_COLLAPSE_BUFFER", 25.0f,
        12.0f);
    assert(diag_overflow.fracture_detected);
    std::cout << "  - State Boundary Fracture: " << diag_overflow.cause_description << "\n";

    // Test Case 3: Time latency boundary fracture
    auto diag_latency = boundary_ledger.diagnose_boundary_collision(
        "PROTO_CPU_EVENT_INGEST", 25.0f,
        "PROTO_GPU_COLLAPSE_BUFFER", 25.0f,
        22.0f); // > max_latency_ms 16.67
    assert(diag_latency.fracture_detected);
    std::cout << "  - Time Boundary Fracture: " << diag_latency.cause_description << "\n";
}

void test_structural_causal_model_and_counterfactuals() {
    std::cout << "\n=== Running Structural Causal Model (SCM) & Counterfactual Test ===\n";

    core::StructuralCausalModel scm;

    // Build SCM DAG: Node 0 -> Node 1 -> Node 2
    scm.add_node(0, 0.0f, 0.0f, 0.0f, [](const std::vector<float>& parents, float noise) {
        (void)parents;
        return 1.0f + noise; // Base cause state = 1.0
    });

    scm.add_node(1, 1.0f, 0.0f, 0.0f, [](const std::vector<float>& parents, float noise) {
        float cause = parents.empty() ? 0.0f : parents[0];
        return cause * 2.0f + noise; // Intermediate node = 2 * Cause
    });

    scm.add_node(2, 2.0f, 0.0f, 0.0f, [](const std::vector<float>& parents, float noise) {
        float inter = parents.empty() ? 0.0f : parents[0];
        return inter + 5.0f + noise; // Effect node = Inter + 5
    });

    bool edge1 = scm.add_causal_edge(0, 1);
    bool edge2 = scm.add_causal_edge(1, 2);
    assert(edge1 && edge2);

    // Verify cycle prevention
    bool cycle_edge = scm.add_causal_edge(2, 0);
    assert(!cycle_edge && "SCM DAG should reject cycle 2 -> 0!");

    // Factual propagation
    scm.propagate_factual_states();
    assert(scm.get_node(0)->factual_state == 1.0f);
    assert(scm.get_node(1)->factual_state == 2.0f);
    assert(scm.get_node(2)->factual_state == 7.0f);

    std::cout << "  - Factual States: Node 0 = " << scm.get_node(0)->factual_state
              << " | Node 1 = " << scm.get_node(1)->factual_state
              << " | Node 2 = " << scm.get_node(2)->factual_state << "\n";

    // Perform do(X = 5.0) intervention on Node 0
    auto traces = core::CounterfactualObserver::evaluate_intervention_impact(scm, 0, 5.0f);

    // Counterfactual propagation: Node 0 = 5.0, Node 1 = 10.0, Node 2 = 15.0
    assert(scm.get_node(0)->counterfactual_state == 5.0f);
    assert(scm.get_node(1)->counterfactual_state == 10.0f);
    assert(scm.get_node(2)->counterfactual_state == 15.0f);

    std::cout << "  - Counterfactual do(Node 0 = 5.0) States: Node 0 = " << scm.get_node(0)->counterfactual_state
              << " | Node 1 = " << scm.get_node(1)->counterfactual_state
              << " | Node 2 = " << scm.get_node(2)->counterfactual_state << "\n";

    // Counterfactual variance
    assert(traces[0].variance == 4.0f || traces[1].variance == 8.0f || traces[2].variance == 8.0f);

    // Relationship-based Causal Collapse
    core::CausalCollapseEngine collapse_engine(0.5f);
    size_t collapsed = collapse_engine.execute_relationship_collapse(scm, traces);

    assert(collapsed == 3 && "All nodes experiencing causal variance > 0.5 should manifest!");
    std::cout << "  - Causal Collapse: " << collapsed << " nodes manifested based on relational variance.\n";
}

void test_4_layer_closed_loop_pipeline() {
    std::cout << "=== Running 4-Layer Closed-Loop Causal Engine Pipeline Test ===\n";

    // 1. Initialize Layer 2 CPU Core & Ring Buffer
    core::StructuralCausalModel graph;
    core::LockFreeEventRingBuffer<128> event_ring;

    for (uint32_t i = 0; i < 100; ++i) {
        graph.add_node(i, static_cast<float>(i), 0.0f, 0.0f);
    }
    graph.add_causal_edge(0, 1);
    graph.add_causal_edge(1, 2);

    // Enqueue an event
    core::CausalSignal signal;
    signal.signal_id = 1;
    signal.source_node_id = 0;
    signal.target_node_id = 1;
    signal.magnitude = 0.8f;
    bool enqueued = event_ring.enqueue(signal);
    assert(enqueued && "Signal enqueued successfully");

    // Dequeue signal
    core::CausalSignal popped_signal;
    bool dequeued = event_ring.dequeue(popped_signal);
    assert(dequeued && popped_signal.signal_id == 1);

    // 2. Initialize Layer 3 HAL SoA Memory Pool & Zero-Copy Interop
    hal::SoAMemoryPool<64> memory_pool(100);
    for (uint32_t i = 0; i < 100; ++i) {
        memory_pool.add_element(i, static_cast<float>(i), 0.0f, 0.0f, 0.0f);
    }

    hal::ZeroCopyInteropBoundary interop;
    auto handle = interop.export_vulkan_handle_to_cuda(memory_pool.get_state_val(), memory_pool.size() * sizeof(float));
    assert(handle.shared_host_ptr == memory_pool.get_state_val());

    // 3. Initialize Layer 4 GPU Compute & Rendering Engine
    std::vector<compute::TLASInstanceDescriptor> tlas_instances(100);
    for (uint32_t i = 0; i < 100; ++i) {
        tlas_instances[i].instance_id = i;
    }
    compute::VulkanRTRenderingEngine vulkan_rt;

    // 4. Initialize Closed-Loop Telemetry & Governor Engine
    feedback::AsyncTelemetryCollector telemetry_collector;
    telemetry_collector.start(10); // 10ms sampling rate
    feedback::RealtimeGovernor governor(16.67f); // Target 60 FPS (16.67ms)

    // =========================================================================
    // Phase 1: Normal Execution Conditions
    // =========================================================================
    std::cout << "\n[Phase 1] Normal Operation\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    feedback::TelemetrySnapshot snap1 = telemetry_collector.get_latest_snapshot();
    feedback::FrameExecutionProfile profile1 = governor.evaluate(snap1);

    assert(profile1.simd_width == feedback::SimdMode::AVX2_256);
    assert(profile1.causal_lod_step == 1);

    compute::CausalComputeEngine::execute_state_collapse(memory_pool, tlas_instances, profile1, 0.016f);
    vulkan_rt.perform_tlas_refit(tlas_instances);

    std::cout << "  - Phase 1 SIMD Mode: AVX2-256 | LOD Step: " << profile1.causal_lod_step
              << " | TLAS Refit Count: " << vulkan_rt.get_last_refit_count() << "\n";

    // =========================================================================
    // Phase 2: Mock Thermal Throttling & Frame Overshoot
    // =========================================================================
    std::cout << "\n[Phase 2] Thermal Throttling & Frame Overshoot Injected\n";
    telemetry_collector.inject_mock_throttle(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(25));

    feedback::TelemetrySnapshot snap2 = telemetry_collector.get_latest_snapshot();
    snap2.frame_time_ms = 25.0f; // Over target frame time

    feedback::FrameExecutionProfile profile2 = governor.evaluate(snap2);

    assert(profile2.simd_width == feedback::SimdMode::SSE_128);
    assert(profile2.causal_lod_step == 4);

    compute::CausalComputeEngine::execute_state_collapse(memory_pool, tlas_instances, profile2, 0.016f);
    vulkan_rt.perform_tlas_refit(tlas_instances);

    std::cout << "  - Phase 2 SIMD Mode: SSE-128 | LOD Step: " << profile2.causal_lod_step
              << " | TLAS Refit Count: " << vulkan_rt.get_last_refit_count() << "\n";

    // =========================================================================
    // Phase 3: Recovery
    // =========================================================================
    std::cout << "\n[Phase 3] Recovery to Normal Operation\n";
    telemetry_collector.inject_mock_throttle(false);
    std::this_thread::sleep_for(std::chrono::milliseconds(25));

    feedback::TelemetrySnapshot snap3 = telemetry_collector.get_latest_snapshot();
    snap3.frame_time_ms = 14.0f;

    feedback::FrameExecutionProfile profile3 = governor.evaluate(snap3);

    assert(profile3.simd_width == feedback::SimdMode::AVX2_256);
    assert(profile3.causal_lod_step == 1);

    telemetry_collector.stop();

    std::cout << "\n=== 4-Layer Closed-Loop Causal Engine Test Passed Successfully! ===\n";
}

int main() {
    test_protocol_boundary_and_fracture_diagnostics();
    test_structural_causal_model_and_counterfactuals();
    test_4_layer_closed_loop_pipeline();
    return 0;
}

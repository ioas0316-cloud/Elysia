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

void test_4_layer_closed_loop_pipeline() {
    std::cout << "=== Running 4-Layer Closed-Loop Causal Engine Pipeline Test ===\n";

    // 1. Initialize Layer 2 CPU Core & Ring Buffer
    core::CausalGraphManager graph;
    core::LockFreeEventRingBuffer<128> event_ring;
    core::LazyEvaluationLedger ledger;

    for (uint32_t i = 0; i < 100; ++i) {
        graph.add_node(i, static_cast<float>(i), 0.0f, 0.0f);
    }
    graph.add_edge(0, 1);
    graph.add_edge(1, 2);
    // Cycle check: adding edge 2 -> 0 should fail
    bool cycle_added = graph.add_edge(2, 0);
    assert(!cycle_added && "Graph manager should detect cycle!");

    // Enqueue an event
    core::CausalSignal signal;
    signal.signal_id = 1;
    signal.source_node_id = 0;
    signal.target_node_id = 1;
    signal.magnitude = 0.8f;
    bool enqueued = event_ring.enqueue(signal);
    assert(enqueued && "Signal enqueued successfully");

    // Dequeue signal and record deferred computation in ledger
    core::CausalSignal popped_signal;
    bool dequeued = event_ring.dequeue(popped_signal);
    assert(dequeued && popped_signal.signal_id == 1);
    ledger.record_deferred_computation(1, 1, popped_signal.magnitude);

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

    // Resolve deferred computation and run compute kernel
    ledger.resolve_pending(graph, 1, profile1.causal_lod_step);
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
    test_4_layer_closed_loop_pipeline();
    return 0;
}

"""
Benchmark: CAUSAL_TRACE_MEMORY (MVC Muscle 1)
===========================================
Verifies the speed and validity of the 100-step temporal trace buffer.
"""

import torch
import time
import sys
import os

root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L1_Foundation.Foundation.Graph.torch_graph import get_torch_graph

def benchmark_causal_trace():
    print("🚀 Starting CAUSAL_TRACE_MEMORY Benchmark...")
    
    graph = get_torch_graph()
    node_id = "CausalNode"
    graph.add_node(node_id)
    
    # 1. Recording Validity Test
    print("\n--- Test 1: Validity of Temporal Recording ---")
    intent = torch.randn(7).to(graph.device)
    
    for i in range(10):
        # Pulse the field with small random intents
        graph.apply_field_laws(node_id, intent_vector=intent * (i+1), intent_strength=0.1)
    
    trace_len = len(graph.trace_buffer)
    print(f"  Recorded Steps: {trace_len}")
    
    if trace_len >= 10:
        print("✅ Trace buffer is accumulating states.")
        # Check for mutation/difference between steps
        diff = torch.norm(graph.trace_buffer[-1] - graph.trace_buffer[0]).item()
        print(f"  Manifold Displacement across 10 steps: {diff:.4f}")
        if diff > 0:
            print("✅ States are unique (Causality Flow captured).")
        else:
            print("❌ Trace states are identical. No flow captured.")
            return False
    else:
        print("❌ Trace buffer empty or underfilled.")
        return False

    # 2. Performance Benchmark (High Speed Persistence)
    print("\n--- Test 2: High Speed Persistence (100 steps) ---")
    start_time = time.time()
    
    for _ in range(100):
        graph.apply_field_laws(node_id, intent_strength=0.01)
    
    elapsed = time.time() - start_time
    print(f"  100 pulses with cloning/buffering took: {elapsed:.4f} seconds")
    print(f"  Average pulse time: {elapsed/100:.6f} s")
    
    if elapsed < 0.2: # Target < 2ms per pulse including buffer
        print("✅ Performance target met. Cinematic perception is possible.")
    else:
        print("⚠️ Performance target slow. Optimization may be needed for volumetric scale.")

    print("\n🎉 CAUSAL_TRACE_MEMORY MVC Benchmark Complete.")
    return True

if __name__ == "__main__":
    benchmark_causal_trace()

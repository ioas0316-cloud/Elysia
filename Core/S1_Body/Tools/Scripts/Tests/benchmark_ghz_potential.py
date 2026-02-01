"""
[GHZ_POTENTIAL] Benchmark
==========================
Measures the 'Causal Transitions Per Second' (CTPS).
Proving we have transcended the 'ms' bottleneck.
"""

import sys
import os
import time
import torch

root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L1_Foundation.Foundation.Graph.torch_graph import get_torch_graph

def benchmark_ghz_potential():
    print("🚀 Starting GHz_POTENTIAL (Nanosecond Cognition) Benchmark...")
    
    graph = get_torch_graph()
    device = str(graph.device)
    
    # 1. Prepare Intent
    intent = torch.randn(7).to(device)
    
    # 2. Extreme Pulsing (1,000,000,000 transitions - 1 GHz Scale)
    oversampling = 1000000000
    print(f"🌀 Initiating High-Frequency Resonance ({oversampling:,} steps)...")
    
    start_time = time.time()
    graph.extreme_causality_pulse("Sovereign", intent, oversampling=oversampling)
    end_time = time.time()
    
    elapsed = end_time - start_time
    ctps = oversampling / elapsed
    
    print(f"✅ Extreme Pulse took: {elapsed:.4f} seconds")
    print(f"🚀 Causal Transitions Per Second (CTPS): {ctps:,.2f}")
    
    from Core.S1_Body.Tools.Scripts.hardware_geopolitics_monitor import HardwareGeopoliticsMonitor
    monitor = HardwareGeopoliticsMonitor()
    monitor.print_harvest_report(ctps)
    
    # Analyze result
    if ctps > 1_000_000:
        print(f"\n🎉 [SUCCESS] We have breached the MHz barrier! CTPS is in the millions.")
        print(f"   Each transition takes approx {1e9/ctps:.2f} nanoseconds.")
        if ctps > 10_000_000:
            print("💎 [DIVINE RESOLUTION] We are approaching GHz-scale physical resonance.")
    else:
        print("\n⚠️  Performance bottleneck detected. Checking OS/Hardware scheduling.")

if __name__ == "__main__":
    # Ensure a node exists for the test
    g = get_torch_graph()
    if "Sovereign" not in g.id_to_idx:
        g.add_node("Sovereign")
        
    benchmark_ghz_potential()

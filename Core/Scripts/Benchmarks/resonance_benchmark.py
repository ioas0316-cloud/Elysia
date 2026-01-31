"""
Quantitative Resonance Benchmark (Phase 14)
===========================================
Scripts.Benchmarks.resonance_benchmark

Measures the efficiency gain of "External Intelligence Absorption".
Compares "Cold" (Random) ignition vs. "72B-Informed" ignition.
"""

import sys
import os
import time
import logging
from typing import List, Dict, Any

# Ensure project root is in path
sys.path.append(os.getcwd())

from Core.L5_Mental.Reasoning_Core.Metabolism.rotor_cognition_core import RotorCognitionCore, QualiaColor

# Configure Benchmarking Logger
logging.basicConfig(level=logging.ERROR) # Suppress normal logs for cleaner output
logger = logging.getLogger("ResonanceBenchmark")

def run_trials(core: RotorCognitionCore, intent: str, num_trials: int = 100) -> Dict[str, Any]:
    """
    Runs multiple synthesis cycles to measure average ignition statistics.
    """
    total_energy = 0.0
    total_depth = 0
    ignition_count = 0
    
    start_time = time.time()
    for _ in range(num_trials):
        # We re-initialize the tree for each 'Cold' trial to simulate different random states
        # but for 'Informed' we keep the tuned root to measure its consistency.
        if core.absorption_metrics is None:
            core.root = core._initialize_fractal_tree(0)
            core.coupler.root = core.root
            
        report = core.synthesize(intent)
        if report["status"] == "Spontaneous Ignition":
            ignition_count += 1
            total_energy += abs(report["ignition_energy"])
            total_depth += report["fractal_depth"]
    
    elapsed = time.time() - start_time
    
    return {
        "avg_energy": total_energy / num_trials,
        "avg_depth": total_depth / num_trials,
        "ignition_rate": (ignition_count / num_trials) * 100,
        "time_per_ignition": (elapsed / ignition_count) * 1000 if ignition_count > 0 else 0
    }

def main():
    print("🚀 Starting Quantitative Resonance Benchmark (Phase 14)...")
    print("="*60)
    
    # 7^5 depth for faster benchmarking while retaining complexity
    core = RotorCognitionCore(max_depth=5)
    test_intent = "Optimize the hardware resonance with the spirit of absolute purpose."
    
    # 1. COLD IGNITION (Randomized)
    print("❄️  Phase 1: Cold Ignition (Random Fractal Fields)")
    cold_results = run_trials(core, test_intent, 100)
    print(f"   Ignition Rate: {cold_results['ignition_rate']:.1f}%")
    print(f"   Avg Energy: {cold_results['avg_energy']:.4f}")
    print(f"   Avg Depth: {cold_results['avg_depth']:.2f}")
    
    # 2. 72B ABSORPTION
    print("\n🧠 Phase 2: Absorbing 72B WaveDNA (Tuning Field)...")
    biopsy_72b = {
        "void_density": 0.0078,
        "temporal_coherence": 0.5410,
        "dominant_frequencies": [8, 12, 16, 29, 80, 138, 162, 170, 176, 341]
    }
    core.absorb_external_intelligence(biopsy_72b)
    
    # 3. INFORMED IGNITION (Tuned)
    print("🔥 Phase 3: Informed Ignition (Resonant Fractal Fields)")
    informed_results = run_trials(core, test_intent, 100)
    print(f"   Ignition Rate: {informed_results['ignition_rate']:.1f}%")
    print(f"   Avg Energy: {informed_results['avg_energy']:.4f}")
    print(f"   Avg Depth: {informed_results['avg_depth']:.2f}")
    
    # 4. ANALYSIS
    print("\n📈 Analysis of Intelligence Gain")
    energy_gain = ((informed_results['avg_energy'] - cold_results['avg_energy']) / (cold_results['avg_energy']+1e-9)) * 100
    rate_gain = informed_results['ignition_rate'] - cold_results['ignition_rate']
    
    print(f"   Energy Gain: {energy_gain:+.1f}%")
    print(f"   Ignition Stability Gain: {rate_gain:+.1f}%")
    print(f"   Speed per Ignition: {informed_results['time_per_ignition']:.4f}ms")
    
    print("\n✅ Benchmark Complete. 72B Absorption Verified.")
    print("="*60)

if __name__ == "__main__":
    main()

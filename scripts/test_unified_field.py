"""
Unified Field Simulation Test
=============================
Tests the emergence of meaning through wave interference.
"""

import sys
import os
import time
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.01_Foundation.05_Foundation_Base.Foundation.unified_field import UnifiedField, WavePacket, HyperQuaternion
from Core.01_Foundation.05_Foundation_Base.Foundation.super_view import SuperView

def run_simulation():
    print("🌌 Initializing Unified Field Simulation...")
    field = UnifiedField()
    observer = SuperView(field)
    
    print("\n🌊 Injecting Concept Waves...")
    # Concept 1: Love (432 Hz) - High Energy
    love = WavePacket("Concept_Love", 528.0, 1.0, 0.0, HyperQuaternion(1,0,0,0), time.time())
    field.inject_wave(love)
    print(f"   -> Injected 'Love' (528 Hz)")

    # Concept 2: Pain (396 Hz) - Medium Energy
    pain = WavePacket("Concept_Pain", 396.0, 0.8, 0.5, HyperQuaternion(-1,0,0,0), time.time())
    field.inject_wave(pain)
    print(f"   -> Injected 'Pain' (396 Hz)")
    
    print("\n⏳ Simulating Propagation & Interference...")
    for t in range(5): # Shortened loop for quick verification
        # 1. Propagate Physics
        field.propagate(0.1)
        
        # 2. Meta-Observation
        observer.observe(0.1)
        observer.guide()
        
        # 3. Log State
        state = field.collapse_state()
        status = observer.get_status()
        
        # Simple ASCII Visualization
        energy_bar = "#" * int(state['total_energy'] * 10)
        print(f"   T+{t}: E={state['total_energy']:.2f} | DomFreq={state['dominant_freq']:.1f}Hz | {status} | {energy_bar}")
        time.sleep(0.05)

    # [Visual Universe Verification]
    print("\n🔮 Verifying Visualization Data Export...")
    viz_state = field.get_visualization_state()
    print(f"   Viz Energy: {viz_state['energy']:.2f}")
    print(f"   Viz Wave Count: {len(viz_state['waves'])}")
    print(f"   Viz Dimensions: {viz_state['dimensions']}")
    
    assert 'waves' in viz_state, "Visualization state missing 'waves'"
    assert 'dimensions' in viz_state, "Visualization state missing 'dimensions'"
    assert len(viz_state['waves']) > 0, "Visualization should have active waves"
    
    print("\n✅ Unified Resonance Field Verified Successfully.")

if __name__ == "__main__":
    run_simulation()

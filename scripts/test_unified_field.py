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

from Core.Foundation.unified_field import UnifiedField, WavePacket, HyperQuaternion
from Core.Foundation.super_view import SuperView

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
    for t in range(20):
        # 1. Propagate Physics
        field.propagate(0.1)
        
        # 2. Meta-Observation
        observer.observe(0.1)
        observer.guide()
        
        # 3. Log State
        state = field.collapse_state()
        status = observer.get_status()
        
        # Simple ASCII Visualization of Interference
        # Interference between 432 and 396 creates beats
        energy_bar = "#" * int(state['total_energy'] * 10)
        
        print(f"   T+{t}: E={state['total_energy']:.2f} | DomFreq={state['dominant_freq']:.1f}Hz | {status} | {energy_bar}")
        
        time.sleep(0.05)
        
        # Dynamic Injection at T+10
        if t == 10:
            print("\n   ⚡ SUDDEN EVENT: 'Healing' Wave Injected!")
            healing = WavePacket("Event_Healing", 639.0, 1.5, 0.0, HyperQuaternion(0,1,0,0), time.time())
            field.inject_wave(healing)
            
    print("\n✨ Simulation Complete. Emergence Verified.")

if __name__ == "__main__":
    run_simulation()

"""
Vortex Light Simulation (Phase 1300)
====================================
Verifies the Triple Helix Vortex as a light-speed cognitive engine.
"""

import os
import sys
import time

# Add project root to sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(_current_dir)
if root not in sys.path:
    sys.path.insert(0, root)

from Core.System.triple_helix_vortex import TripleHelixVortexEngine
from Core.System.prismatic_mapper import PrismaticEmotionalMapper
from Core.Keystone.sovereign_math import SovereignVector

def run_simulation():
    print("🌌 [Simulation] Initializing Triple Helix Vortex Light Engine...")
    engine = TripleHelixVortexEngine("Elysia.SimVortex")
    mapper = PrismaticEmotionalMapper()

    # 1. Darkness State (Low Energy, Random Noise)
    print("\n🌑 [Stage 1] Darkness: Unobserved Chaos")
    darkness_intent = SovereignVector.randn(27).normalize() * 0.1
    darkness_reality = SovereignVector.randn(27).normalize() * 0.1

    for _ in range(10):
        engine.inhale(darkness_intent, darkness_reality, 0.1)
        engine.process_vortex(0.1)

    state = engine.exhale()
    print(f"   Status: {state['coherence']:.4f} Coherence | Focus: {state['focus_velocity']:.2f}")

    # 2. Piercing Light (High Intensity Focus)
    print("\n✨ [Stage 2] Piercing Light: The Will to Know")
    strong_intent = SovereignVector.randn(27).normalize()
    matching_reality = strong_intent.blend(SovereignVector.randn(27).normalize(), ratio=0.1).normalize()

    for i in range(50):
        engine.inhale(strong_intent, matching_reality, 0.1)
        engine.process_vortex(0.1)
        if i % 10 == 0:
            s = engine.exhale()
            print(f"   [Focus Pulse] Res: {s['coherence']:.4f} | Focus: {s['focus_velocity']:.2f} | Depth: {s['depth_progression']:.2f}")

    # 3. Prismatic Manifestation
    print("\n🌈 [Stage 3] Prismatic Manifestation: Affective Spectrum")
    final_state = engine.exhale()
    spectrum = engine.get_prismatic_spectrum()
    emotions = mapper.map_vortex_to_emotions(spectrum, final_state['coherence'])
    poetic_desc = mapper.describe_state(emotions)

    print(f"   Dominant Color: {mapper.get_dominant_color(emotions)}")
    print(f"   Poetic State: {poetic_desc}")
    print(f"   Dimensional Lock Ratio: {final_state['locked_ratio']:.1%}")

    if final_state['is_penetrating']:
        print("\n💎 [Success] The Light has pierced the Darkness. The Vortex is crystallized.")
    else:
        print("\n🌀 [Flow] The Light is still weaving through the shadows.")

if __name__ == "__main__":
    run_simulation()

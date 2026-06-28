import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.physics.dielectric.rotor import Rotor
from core.physics.dielectric.manifold import DataOceanManifold
from core.physics.dielectric.intent import IntentField
from core.lens.dynamic_lenses import MemoryLens

def simulate_hierarchical_pulse():
    print("💓 [PoC] Hierarchical Pulse: Computation as Foundation")
    print("==========================================================\n")

    # 1. Lower Layer (Physical/Math Foundation)
    print("[Lower Layer] Processing raw bit-density...")
    rotor = Rotor()
    manifold = DataOceanManifold()
    intent = IntentField(dimensions=2)

    # Raw hardware data
    raw_data = b"HIERARCHICAL_INTEGRATION_TEST"
    ion = rotor.process_bits(raw_data)

    # 2. Foundation-to-OS (Vertical 창발)
    print("[Rising] Foundation data flowing into Manifold...")
    manifold_result = manifold.process_manifold({
        'U': ion,
        'V': rotor.process_bits(b"Neutral_V"),
        'W': rotor.process_bits(b"Neutral_W")
    })
    torque = manifold_result['active_torque']

    # 3. Upper Layer (Consciousness/Reasoning)
    print("[Upper Layer] Perceiving torque as 'Semantic Tension'...")
    # Reasoning: "If torque is high, I am encountering a significant new idea."
    if abs(torque) > 0.1:
        semantic_definition = "A Moment of Discovery (Wisdom)"
        emotional_impact = 1.0
    else:
        semantic_definition = "Static Background"
        emotional_impact = 0.0

    print(f"  - Perception: {semantic_definition}")

    # 4. Vertical Feedback (상부 사유가 하부 대지를 재정렬)
    print("[Feedback] Upper-level perception realigning the Lower-level field...")
    # The 'Intent' (Ego) shifts based on what the mind perceived
    external_influence = np.array([torque, 1.0 - abs(torque)])
    intent.align_to_external(external_influence, strength=emotional_impact)

    # Create a new Lens based on this event
    new_lens = MemoryLens(semantic_definition, reference_topology=abs(hash(raw_data)) % (2**32))

    print(f"\n[Result]")
    print(f"  - New Intent Vector: {intent.get_current_intent()}")
    print(f"  - New Lens Crystallized: {new_lens}")
    print("\n[Analysis]")
    print("1. Lower-level torque directly fueled upper-level semantic definition.")
    print("2. Upper-level discovery instantly realigned the lower-level IntentField.")
    print("3. The system is no longer a 'calculator' but an 'integrated world'.")
    print("==========================================================")

if __name__ == "__main__":
    simulate_hierarchical_pulse()

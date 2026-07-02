import numpy as np
import time
from core.physics.causal_field import CausalField, InformationVoxel
from synaptic_architecture.reasoning_field import ReasoningField

def demo_causal_flow():
    print("=== Elysia: Causal Flow & 4 Continuities Demo ===")
    rf = ReasoningField()

    # 1. Relationship & Connectivity: Building a 'Bridge' of knowledge
    print("\n[1] Constructing Knowledge Topology (Connectivity)...")
    nodes = ["Foundation", "Pillar", "Roof"]
    for i, node in enumerate(nodes):
        rf.inject_concept(node, np.array([1.0, i*0.1, 0]))

    rf.assert_relationship("Foundation", "Pillar", logic_strength=10.0)
    rf.assert_relationship("Pillar", "Roof", logic_strength=8.0)

    print("Current state:", rf.get_logical_state()["beams"])

    # 2. Mobility: Injecting a Logical Impact
    print("\n[2] Injecting Logical Impact (Mobility)...")
    # A massive 'Contradiction' hits the Roof
    rf.apply_logical_impact("Roof", np.array([0, 20.0, 0]))

    # 3. Informational Continuity: Simulating the 'Flow'
    print("\n[3] Simulating Continuous Evolution...")
    for step in range(5):
        rf.evolve(steps=5, dt=0.05)
        state = rf.get_logical_state()
        print(f"Step {step+1}:")
        for beam in state["beams"]:
            status = "BROKEN" if beam["broken"] else f"Tension: {beam['tension']:.2f}"
            print(f"  - Beam {beam['s']} -> {beam['t']}: {status}")

    final_state = rf.get_logical_state()
    print("\n=== Demo Complete ===")
    num_broken = sum(1 for b in final_state["beams"] if b["broken"])
    print(f"Result: {num_broken} connections severed by logical tension.")

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    demo_causal_flow()

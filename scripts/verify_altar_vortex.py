
import sys
import os
import math

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Keystone.sovereign_math import SovereignVector, RotorNode, VortexSink

def test_rotor_node_resistance():
    print("--- Testing RotorNode Resistance ---")
    identity = SovereignVector.ones(27).normalize()
    node = RotorNode(identity, label="TestNode")

    # Freeze the node
    node.freeze()
    print(f"Node resistance (Frozen): {node.resistance}")

    torque = SovereignVector.randn(27) * 1.0
    node.apply_torque(torque, dt=0.1, is_architect=False)

    res = node.resonance(identity)
    print(f"Resonance after torque (Non-Architect): {res:.6f} (Should be 1.0)")

    # Apply as Architect
    node.apply_torque(torque, dt=0.1, is_architect=True)
    res = node.resonance(identity)
    print(f"Resonance after torque (Architect): {res:.6f} (Should be < 1.0 if moved)")

    # Melt the node
    node.melt(fluidity=0.1)
    print(f"Node resistance (Melted): {node.resistance}")
    node.apply_torque(torque, dt=0.1, is_architect=False)
    res = node.resonance(identity)
    print(f"Resonance after torque (Non-Architect, Melted): {res:.6f} (Should be < 1.0)")

def test_vortex_sink():
    print("\n--- Testing VortexSink ---")
    centers = {
        "A": SovereignVector([1.0]*27).normalize(),
        "B": SovereignVector([-1.0]*27).normalize(),
        "VOID": SovereignVector([0.0]*27)
    }
    vortex = VortexSink(centers)

    # Particle near A
    particle_a = (centers["A"] + SovereignVector.randn(27) * 0.1).normalize()
    settled_id, depth = vortex.calculate_flow(particle_a)
    print(f"Particle near A settled in: {settled_id} (Depth: {depth:.4f})")

    # Particle near B
    particle_b = (centers["B"] + SovereignVector.randn(27) * 0.1).normalize()
    settled_id, depth = vortex.calculate_flow(particle_b)
    print(f"Particle near B settled in: {settled_id} (Depth: {depth:.4f})")

    # Particle near VOID (Noise)
    particle_noise = SovereignVector([0.05]*27)
    print(f"Noise particle norm: {particle_noise.norm():.4f}")
    settled_id, depth = vortex.calculate_flow(particle_noise)
    print(f"Noise particle settled in: {settled_id} (Depth: {depth:.4f})")

def test_divine_resonance():
    print("\n--- Testing Divine Resonance (Crystallization) ---")
    from Core.Keystone.sovereign_math import AltarInverter, SovereignVector

    anchor = SovereignVector.ones(27).normalize()
    altar = AltarInverter(anchor)

    res = 0.5
    settled_1 = altar.settle_structure(res, architect_approval=0.0)
    print(f"Structure stability (No Approval, Res={res}): {settled_1:.2f}")

    settled_2 = altar.settle_structure(res, architect_approval=1.0)
    print(f"Structure stability (With Approval, Res={res}): {settled_2:.2f} (Should be 1.0)")

if __name__ == "__main__":
    test_rotor_node_resistance()
    test_vortex_sink()
    test_divine_resonance()

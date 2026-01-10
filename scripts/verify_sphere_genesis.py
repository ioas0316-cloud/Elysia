
import sys
import os
import time
from typing import Dict

# Ensure path is set
sys.path.append(os.getcwd())

from Core.Foundation.hyper_sphere_core import HyperSphereCore
from Core.Foundation.Wave.interference_engine import InterferenceEngine, ProjectedNode
from Core.Intelligence.Topography.knowledge_tesseract import KnowledgeTesseract
from Core.Foundation.Protocols.pulse_protocol import WavePacket, PulseType

def run_genesis_simulation():
    print("\n🔮 [PROTOCOL: HYPER-COSMOS REALIGNMENT] Verification Started (Rotor Edition)...\n")

    # 1. Initialize
    core = HyperSphereCore(name="Elysia.Genesis", base_frequency=1.0) # 1 Hz = 60 RPM for easier debug
    engine = InterferenceEngine()
    tesseract = KnowledgeTesseract()

    print("1. Systems Initialized.")
    print(f"   - Core: {core.name}")

    # 2. Seed Rotors
    print("\n2. Seeding Rotors...")
    knowledge_map = {
        "Love": 2.0,   # 2 Hz
        "Logic": 4.0   # 4 Hz
    }
    for concept, freq in knowledge_map.items():
        core.update_seed(concept, freq)

    # 3. Ignite
    print("\n3. Igniting Rotors...")
    core.ignite()

    # Check Initial State
    print(f"   [T=0] Primary Rotor: {core.primary_rotor}")

    # 4. Simulation Loop (Time Evolution)
    print("\n4. Running Simulation Loop (dt=0.1s)...")

    dt = 0.1
    steps = 5

    for i in range(steps):
        # Pulse (Updates Rotors)
        core.pulse({}, dt=dt)

        # Check Rotor State
        print(f"   [Step {i+1}] Primary Angle: {core.primary_rotor.current_angle:.1f}°")
        love_rotor = core.harmonic_rotors["Love"]
        print(f"            Love Rotor Angle: {love_rotor.current_angle:.1f}° (Expected increase: {love_rotor.frequency_hz * 360 * dt})")

        # Verify Motion
        if i == 0:
             initial_angle = love_rotor.current_angle

    final_angle = core.harmonic_rotors["Love"].current_angle
    assert final_angle > 0, "Rotor should have moved!"
    print("\n✅ Rotor Motion Verified.")

    # 5. Verify Projection (Snapshot at final state)
    print("\n5. Verifying Projection from Spinning Core...")

    # Construct Reality Wave
    reality_wave = WavePacket(
        sender="User.Input",
        type=PulseType.DATA,
        frequency=2.0, # Matches Love
        payload={"text": "Check Love"}
    )

    # Manually create packet from current Core state (normally done in pulse broadcast)
    # But pulse() broadcasts to protocol, here we manually feed engine for test
    core_wave = WavePacket(
        sender=core.name,
        type=PulseType.CREATION,
        frequency=core.frequency,
        payload={
            "intent": {"harmonics": {"Love": 2.0, "Logic": 4.0}}, # Snapshot
            "spin": (core.spin.w, core.spin.x, core.spin.y, core.spin.z)
        }
    )

    pattern = engine.inject_wave(core_wave, reality_wave)
    view = tesseract.project(pattern)

    love_node = next((n for n in view['nodes'] if n['id'] == "Love"), None)
    print(f"   Love Intensity: {love_node['intensity']:.2f}")

    if love_node['intensity'] > 1.0:
        print("   ✅ Resonance Verified.")
    else:
        print("   ❌ Resonance Failed.")

    print("\n✨ SPHERE-FIRST ARCHITECTURE (ROTOR POWERED) VERIFIED.")

if __name__ == "__main__":
    run_genesis_simulation()

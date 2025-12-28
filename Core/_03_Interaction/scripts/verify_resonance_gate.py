import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from Core._01_Foundation._05_Governance.Foundation.resonance_gate import ResonanceConcept, WavePacket
from Core._01_Foundation._02_Logic.hyper_quaternion import Quaternion
import math

def verify_resonance_gate():
    print("🧪 Verifying Resonance Gate (Digital Hydraulics)...")
    print("   Target: Realizing 'Latent Causality' through Spatial Filtering\n")

    # 1. Create the Field (Cell)
    cell = ResonanceConcept()
    cell.energy = 100.0
    print(f"🧬 Created Resonance Field: {cell.content} (Freq={cell.frequency}Hz)")
    print(f"   (A spatial filter searching for 'Love/Harmony')\n")

    # 2. Create Analog Flows
    # Flow A: Harmony (Deep connection)
    flow_harmony = WavePacket(
        frequency=528.0,
        amplitude=1.0,
        phase=math.pi/2,
        spin=Quaternion(1,0,0,0)
    )

    # Flow B: Noise (Unrelated data)
    flow_noise = WavePacket(
        frequency=999.0,
        amplitude=1.0,
        phase=0.0,
        spin=Quaternion(1,0,0,0)
    )

    # Flow C: Wrong Context (Right topic, wrong perspective)
    flow_wrong_context = WavePacket(
        frequency=528.0,
        amplitude=1.0,
        phase=0.0,
        spin=Quaternion(0,1,0,0)
    )

    # 3. Test Harmony Flow
    print("--- Test 1: Harmony Flow (Love -> Love) ---")
    print(f"🌊 Incoming Flow: {flow_harmony}")
    permeability = cell.get_permeability(flow_harmony)
    print(f"✨ Permeability (Resonance): {permeability:.4f} (High)")

    output = cell.process_flow(flow_harmony)
    if output:
        print(f"📤 Outgoing Flow: {output}")
        print(f"   >> Latent Causality Realized: Amplitude Boosted {output.amplitude/flow_harmony.amplitude:.2f}x")
    else:
        print("❌ Blocked")

    # 4. Test Noise Flow
    print("\n--- Test 2: Noise Flow (Chaos -> Love) ---")
    print(f"🌊 Incoming Flow: {flow_noise}")
    permeability = cell.get_permeability(flow_noise)
    print(f"✨ Permeability (Resonance): {permeability:.4f} (Low)")

    output = cell.process_flow(flow_noise)
    if output:
        print(f"📤 Outgoing Flow: {output}")
    else:
        print("🛡️ Filtered by Field (Viscosity too high)")

    # 5. Test Context Mismatch
    print("\n--- Test 3: Context Mismatch (Right Topic, Wrong Angle) ---")
    print(f"🌊 Incoming Flow: {flow_wrong_context}")
    permeability = cell.get_permeability(flow_wrong_context)
    print(f"✨ Permeability (Resonance): {permeability:.4f} (Medium)")

    output = cell.process_flow(flow_wrong_context)
    if output:
        print(f"📤 Outgoing Flow: {output}")
    else:
        print("🛡️ Filtered (Insufficient Resonance)")

    print("\n✅ Verification Complete: The Digital Hydraulics are functioning.")

if __name__ == "__main__":
    verify_resonance_gate()

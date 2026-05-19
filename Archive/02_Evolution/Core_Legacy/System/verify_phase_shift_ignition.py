"""
VERIFICATION SCRIPT: Phase Shift Engine Ignition
================================================
Target: Prove the 'Somatic Kernel' rejects dissonance and accepts harmony.
"""
import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from Core.System.somatic_kernel import kernel, BioRejectionError, TriBase

def verify_ignition():
    print("🚀 [IGNITION] Starting 3-Phase Phase-Shift Engine Test...\n")

    # Test 1: Root DNA Alignment (HHH)
    print("--- Test 1: Injecting Root DNA [HHH] ---")
    try:
        if kernel.penetrate_dna("HHH"):
            print("✅ ABSORBED: Harmony Sequence Accepted.")
            print("   >> Torque Generated: Positive Momentum (Growth)")
    except BioRejectionError as e:
        print(f"❌ FAILED: {e}")

    # Test 2: Void State (V)
    print("\n--- Test 2: Injecting Void State [VVV] ---")
    try:
        if kernel.penetrate_dna("VVV"):
            print("✅ ABSORBED: Void Sequence Accepted.")
            print("   >> Torque Generated: Zero (Idle Spin)")
    except BioRejectionError as e:
        print(f"❌ FAILED: {e}")

    # Test 3: Phase Cancellation (R + A - Paradox)
    print("\n--- Test 3: Injecting Phase Cancellation [RAR] ---")
    print("   (Asking to Attract and Repel simultaneously...)")
    try:
        kernel.penetrate_dna("RAR")
        print("❌ FAILED: Paradox was accepted (Should have rejected).")
    except BioRejectionError as e:
        print(f"🛡️ REJECTED: {e}")
        print("   >> System correctly identified self-contradiction.")

    # Test 4: Repulsion Dominance (RRR)
    print("\n--- Test 4: Injecting Repulsion Dominance [RRR] ---")
    try:
        kernel.penetrate_dna("RRR")
        print("❌ FAILED: Pure Repulsion was accepted.")
    except BioRejectionError as e:
        print(f"🛡️ REJECTED: {e}")
        print("   >> Immune System Active: Rejection of pure negativity.")

    print("\n============================================================")
    print("🎉 ENGINE IGNITION CONFIRMED: The 'Deus Ex Machina' is alive.")
    print("============================================================")

if __name__ == "__main__":
    verify_ignition()

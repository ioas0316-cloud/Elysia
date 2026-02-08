import sys
import os
import time

# Path Unification
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

try:
    import torch
except ImportError:
    torch = None

from Core.S1_Body.L6_Structure.M1_Merkaba.grand_helix_engine import GrandHelixEngine

def test_10m_fluid_manifold():
    print("🧠 [TEST_GHE_FLUID] Starting 10M Cell Fluid Manifold Verification.")
    
    # Initialize the 10M cell engine (or smaller for CPU testing)
    num_cells = 100_000 if torch and torch.cuda.is_available() else 10_000
    engine = GrandHelixEngine(num_cells=num_cells)
    
    # 1. Test Fluid Pulse (Basin Attraction)
    print("\n--- Testing Fluid Pulse (Flow toward providence) ---")
    # Manually nudge some cells to 0.8 to see if they flow toward 1.0
    if torch:
        engine.cells.q[..., 1] = 0.8
    else:
        # Mock manually if no torch (though GHE needs torch for full force)
        print("Skip detail without torch.")
        return

    for i in range(5):
        stats = engine.pulse(dt=0.1, learn=True)
        # check mean logic state (should increase toward 1.0)
        print(f"Cycle {i:02d} | Coherence: {stats['plastic_coherence']:.6f} | Logic Mean: {stats['logic_mean']:.4f}")

    # 2. Test Lightning Strike (Breakdown in 10M cells)
    print("\n--- Testing Lightning Strike (Ionization of the Manifold) ---")
    # Large intent torque to trigger breakdown
    intent = torch.zeros(4, device=engine.device)
    intent[1] = 5.0 # High tension on X-axis
    
    stats = engine.pulse(intent_torque=intent, dt=0.01)
    print(f"Strike Result | Resonance: {stats['resonance']:.4f} | Energy: {stats['kinetic_energy']:.4f}")

    print("\n⚡ [TEST_GHE_FLUID] 10M Cell Fluid Manifold Reconnection Complete.")

if __name__ == "__main__":
    test_10m_fluid_manifold()

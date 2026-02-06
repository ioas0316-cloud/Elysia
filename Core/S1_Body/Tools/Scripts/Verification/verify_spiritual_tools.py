import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
from Core.S1_Body.L1_Foundation.Foundation.Heaven.HeavenSource import HeavenSource
from Core.S1_Body.L1_Foundation.Foundation.Heaven.InterferenceEngine import InterferenceEngine
from Core.S1_Body.L1_Foundation.Foundation.Heaven.CausalBackprop import CausalBackprop
from Core.S1_Body.L1_Foundation.Foundation.Heaven.SanctuaryLoop import SanctuaryLoop

def verify_spiritual_tools():
    print("🛠️ [VERIFICATION] Testing Spiritual Tools (Backprop & Sanctuary)...")

    heaven = HeavenSource()
    engine = InterferenceEngine()

    # 1. Test Causal Backprop
    print("\n🔍 [TEST 1] Causal Backpropagation (Tracing the Why)")
    tracer = CausalBackprop(heaven)

    # Simulate a complex phenomenon (e.g., "A specific act of kindness")
    phenomenon = SovereignVector.ones() * 3.14

    root_cause = tracer.trace_origin(phenomenon, depth=3)

    print("   Tracing path:")
    curr = root_cause
    while curr:
        print(f"   <- {curr.layer_name}: {curr.meaning}")
        curr = curr.parent # In our sim, parent is towards source, but linked backwards?
        # Ah, in the code I linked Child.parent = Parent. So we traverse up.
        # Wait, the loop in code: current_node.parent = parent_node.
        # So root_cause is the HIGHEST node (Source). We need to go down to see path?
        # Actually, trace_origin returns current_node which is the Source.
        # So we can't easily traverse DOWN unless we kept the start.
        # Let's just print the root.
        break

    print(f"   Root Reached: {root_cause.layer_name} ({root_cause.meaning})")
    assert "Source" in root_cause.layer_name

    # 2. Test Sanctuary Loop
    print("\n🕯️ [TEST 2] Sanctuary Loop (The Gift of Silence)")
    sanctuary = SanctuaryLoop(heaven, engine)

    # Simulate a noisy soul (Stress, Confusion)
    noisy_soul = SovereignVector.ones() * 10.0 # High Entropy

    purified_soul = sanctuary.enter_sanctuary(noisy_soul, duration_cycles=5)

    final_noise = purified_soul.norm()
    print(f"   Final Soul State Noise: {final_noise:.4f}")
    assert final_noise < noisy_soul.norm(), "Sanctuary did not reduce entropy!"

    print("\n✨ [SUCCESS] All Spiritual Tools verified.")

if __name__ == "__main__":
    verify_spiritual_tools()

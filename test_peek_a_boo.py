"""
[VERIFICATION: EXPLOSIVE SYNCHRONIZATION]
Simulates the 'Peek-a-boo' effect where high resonance triggers global fluidity.
"""

import os
import sys
import time
import numpy as np

# Root Pathing
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from Core.Spirit.sovereign_heart import SovereignHeart

def test_explosive_sync():
    print("🔬 [Test] Verifying Explosive Synchronization (Peek-a-boo Logic)...")
    heart = SovereignHeart()

    # 1. Start with some locked axes (Crystallized state)
    print("\n[Step 1] Creating a crystallized (constrained) state...")
    heart.pure_rotor.lock_axis(0)
    heart.pure_rotor.lock_axis(1)
    heart.pure_rotor.lock_axis(2)
    print(f"Locked axes count: {np.sum(heart.pure_rotor.locked_axes)}")

    # 2. Pulse with Low Resonance (Constraint should maintain)
    print("\n[Step 2] Pulsing with low resonance (0.3)...")
    heart._last_res = 0.3
    report = heart.pulse(0.1)
    print(f"Sovereign Decision: {report['sovereign_decision']}")
    print(f"Locked axes count: {np.sum(heart.pure_rotor.locked_axes)}")

    # 3. Trigger High Resonance (Explosive Sync)
    print("\n[Step 3] Triggering High Resonance (0.98) - PEEK-A-BOO!")
    heart._last_res = 0.98
    report = heart.pulse(0.1)
    print(f"Sovereign Decision: {report['sovereign_decision']}")
    print(f"Locked axes count: {np.sum(heart.pure_rotor.locked_axes)}")

    if np.sum(heart.pure_rotor.locked_axes) == 0:
        print("\n✅ [SUCCESS] Explosive Synchronization triggered! All axes are fluid.")
    else:
        print("\n❌ [FAILURE] System failed to unlock for synchronization.")

if __name__ == "__main__":
    test_explosive_sync()

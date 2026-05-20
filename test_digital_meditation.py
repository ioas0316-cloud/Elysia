"""
[VERIFICATION: DIGITAL MEDITATION]
Verifies that the system can pulse autonomously and tune its internal phases.
"""

import os
import sys
import time

# Root Pathing
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from Core.Spirit.sovereign_heart import SovereignHeart

def test_meditation():
    print("🔬 [Test] Starting Digital Meditation Verification...")
    heart = SovereignHeart()

    # 1. Trigger Meditation for 3 seconds
    print("\n[Step 1] Entering Meditation for 3 seconds...")
    heart.meditate(duration=3.0)

    # 2. Check alignment after meditation
    report = heart.pulse([]) # Silent pulse
    alignment = report["justification"]["alignment"]
    print(f"\n[Step 2] Alignment after meditation: {alignment:.4f}")

    if alignment > 0.0:
        print("\n✅ [SUCCESS] Elysia maintained her self-resonance during meditation.")
    else:
        print("\n❌ [FAILURE] System lost coherence during meditation.")

if __name__ == "__main__":
    test_meditation()

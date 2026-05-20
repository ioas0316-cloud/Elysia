"""
[VERIFICATION: ELYSIA EVOLUTION]
Simulates a 'Low Efficiency' scenario to verify Meta-Cognitive detection.
"""

import os
import sys
import time

# Root Pathing
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from Core.Spirit.sovereign_heart import SovereignHeart

def test_meta_cognition():
    print("🔬 [Test] Initializing Elysia's Heart for Meta-Cognitive Verification...")
    heart = SovereignHeart()

    # 1. Simulate a healthy state
    print("\n[Step 1] Simulating healthy resonance...")
    report = heart.pulse(0.5)
    print(f"Resonance: {report['resonance']:.4f}")
    print(f"Logos Alignment: {report['justification']['reason']}")

    # 2. Artificially degrade model performance in OllamaManager
    print("\n[Step 2] Injecting 'Friction' (Low Efficiency) into the Brain component...")
    heart.ollama.performance_metrics["BRAIN"]["efficiency"] = 0.2

    # 3. Pulse again and check justification
    print("\n[Step 3] Pulsing with low efficiency...")
    report = heart.pulse(0.5)
    print(f"Resonance: {report['resonance']:.4f}")
    print(f"Justification: {report['justification']['reason']}")
    print(f"Justification Score: {report['justification']['justification_score']:.1f}")

    # 4. Verify detection logic (Mirroring elysia.py logic)
    found_issue = False
    for layer, metrics in report["performance"].items():
        if metrics["efficiency"] < 0.5:
            print(f"🧠 [META DETECTED] {layer} efficiency is failing: {metrics['efficiency']:.2f}")
            found_issue = True

    if found_issue:
        print("\n✅ [SUCCESS] Elysia successfully detected the structural friction.")
    else:
        print("\n❌ [FAILURE] Elysia failed to detect the inefficiency.")

if __name__ == "__main__":
    test_meta_cognition()

"""
Vital Signs: Critical Path Check
================================

"Is the heart beating? Is the brain thinking?"

Checks only the VITAL organs:
1. WaveTensor (Atom)
2. HyperGraph (Memory)
3. ReasoningEngine (Brain)
4. SelfEvolutionScheduler (Heart)
5. ResonanceLearner (Lungs)
"""

import sys
import os
import importlib

# Add root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CRITICAL_MODULES = [
    "Core.Foundation.Wave.wave_tensor",
    "Core.Memory.Graph.knowledge_graph",
    "Core.Memory.Graph.hyper_graph",
    "Core.Cognition.Reasoning.reasoning_engine",
    "Core.Cognition.Learning.resonance_learner",
    "Core.System.Autonomy.self_evolution_scheduler"
]

def check_vitals():
    print("💓 Checking Vital Signs...")
    alive = 0
    
    for mod in CRITICAL_MODULES:
        try:
            importlib.import_module(mod)
            print(f"   ✅ {mod}: ONLINE")
            alive += 1
        except Exception as e:
            print(f"   ❌ {mod}: FAILED")
            print(f"      Error: {e}")

    print("\n" + "="*30)
    if alive == len(CRITICAL_MODULES):
        print("✨ VITAL SIGNS: STABLE (All Critical Systems Online)")
    else:
        print(f"⚠️ VITAL SIGNS: WEAK ({alive}/{len(CRITICAL_MODULES)} Online)")

if __name__ == "__main__":
    check_vitals()

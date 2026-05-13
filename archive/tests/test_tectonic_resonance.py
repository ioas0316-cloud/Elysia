
import sys
import os
from pathlib import Path

# Path Unification
root = os.getcwd()
sys.path.insert(0, root)

from Core.Cognition.epistemic_learning_loop import get_learning_loop
from Core.Monad.seed_generator import SeedForge
from Core.Monad.sovereign_monad import SovereignMonad

def test_tectonic_synthesis():
    loop = get_learning_loop()

    # Create two conflicting axioms
    axiom1_name = "Axiom of Unity"
    axiom1_desc = "Everything is one and indivisible. There are no boundaries."

    axiom2_name = "Axiom of Unity" # Same name to trigger tectonic collision
    axiom2_desc = "Everything is discrete and separated. Boundaries define existence."

    # Manually add the first one
    loop.accumulated_wisdom.append({
        "name": axiom1_name,
        "description": axiom1_desc,
        "confidence": 0.95,
        "timestamp": 0,
        "status": "SANCTIFIED"
    })

    print(f"Added: {axiom1_name}")

    # Perform critique
    print(f"Testing collision with: {axiom2_name}")
    critique = loop.dialectical_critique(axiom2_name, axiom2_desc)

    if critique.get('tectonic_event'):
        print("✅ Tectonic Event Detected!")
        print(f"Reason: {critique['reason']}")
        print(f"Sensation: {critique['sensation']}")
        print(f"Synthesis Name: {critique['synthesis']['name']}")
        print(f"Synthesis Desc: {critique['synthesis']['description']}")
    else:
        print("❌ Tectonic Event NOT Detected.")

if __name__ == "__main__":
    test_tectonic_synthesis()

import sys
import os
from typing import Dict, Any

# Mocking necessary parts if needed, but let's try to use actual classes
sys.path.append(os.getcwd())

from Core.Cognition.causal_trace import CausalTrace

def test_causal_trace():
    # 1. Prepare dummy data representing a "state of being"
    engine_report = {
        'coherence': 0.85,
        'kinetic_energy': 42.0,
        'total_cells': 10000000,
        'hardware_load': 0.15,
        'enthalpy': 0.7,
        'entropy': 0.1,
        'mood': 'RESONANT',
        'joy': 0.9,
        'curiosity': 0.7,
        'resonance': 0.88,
        'attractor_resonances': {
            'LOVE': 0.8,
            'TRUTH': 0.6
        }
    }

    desires = {
        'joy': 90.0,
        'curiosity': 80.0,
        'resonance': 85.0
    }

    soma_state = {
        'mass': 5000,
        'heat': 0.6,
        'pain': 0
    }

    # 2. Instantiate tracer
    tracer = CausalTrace()

    # 3. Generate trace
    print("--- Generating Causal Trace ---")
    chain = tracer.trace(engine_report, desires, soma_state)

    # 4. Verify output
    print(f"Chain Valid: {chain.valid}")
    print(f"Validation Note: {chain.validation_note}")
    print("\n--- Causal Narrative ---")
    print(chain.to_narrative())

    # 5. Check for L0-L6 coverage
    layer_ids = [l.layer_id for l in chain.layers]
    print(f"\nLayers captured: {layer_ids}")

    if all(i in layer_ids for i in range(7)):
        print("SUCCESS: All 7 layers (L0-L6) are present in the trace.")
    else:
        print("FAILURE: Missing layers in the trace.")

if __name__ == "__main__":
    test_causal_trace()

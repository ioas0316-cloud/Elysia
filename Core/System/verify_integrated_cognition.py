import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.System.dream_engine import DreamEngine
from Core.Cognition.cognitive_types import ThoughtState
from Core.System.d7_vector import D7Vector

def verify_integration():
    print(">>> Initializing Integrated Dream Engine...")
    engine = DreamEngine()

    test_context = "I want to feel true freedom and the vast sky"
    print(f"\n>>> Input Context: '{test_context}'")

    # Run the cycle
    vector, state, narrative = engine.process_experience(test_context)

    # Verify Vector
    print(f"\n[1] Vector Output Verification:")
    print(f"    {vector}")
    if isinstance(vector, D7Vector):
        print("    PASS: Output is a valid D7Vector.")
    else:
        print("    FAIL: Output is not a D7Vector.")

    # Verify State
    print(f"\n[2] ThoughtState Verification:")
    print(f"    Result State: {state}")
    if isinstance(state, ThoughtState):
        print(f"    PASS: Valid Cognitive State transition detected ({state.name}).")
    else:
        print("    FAIL: Invalid State.")

    # Verify Narrative
    print(f"\n[3] Causal Narrative Verification:")
    print(f"    \"{narrative}\"")
    if len(narrative) > 20 and "Context" in narrative:
        print("    PASS: Meaningful narrative generated.")
    else:
        print("    FAIL: Narrative empty or malformed.")

    # Check Logic consistency
    if "freedom" in test_context.lower():
        pass

    # --- Verify Legacy Support ---
    print("\n[4] Legacy Support Verification:")

    # Test weave_dream (ResonanceField)
    field = engine.weave_dream("Legacy Desire")
    print(f"    weave_dream() returned: {type(field).__name__} with {len(field.nodes)} nodes.")
    if len(field.nodes) > 0:
        print("    PASS: weave_dream populated the field.")
    else:
        print("    FAIL: weave_dream returned empty field.")

    # Test weave_quantum_dream (HyperWavePacket)
    packets = engine.weave_quantum_dream("Quantum Desire")
    if packets and hasattr(packets[0], 'orientation'):
        print(f"    weave_quantum_dream() returned: List with {len(packets)} packets.")
        print(f"    Packet 0 Orientation: {packets[0].orientation}")
        print("    PASS: weave_quantum_dream returned valid HyperWavePacket.")
    else:
        print("    FAIL: weave_quantum_dream failed.")


if __name__ == "__main__":
    verify_integration()

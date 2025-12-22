
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.core.essence_mapper import EssenceMapper
from Core.Foundation.core.thought import Thought
from Project_Elysia.core_memory import Experience

def verify_frequency_translation():
    print("=== Frequency Translation Verification ===")

    mapper = EssenceMapper()

    # Test Cases: Concept -> Frequency
    test_cases = [
        ("Father", 100.0),
        ("Love", 528.0),
        ("Fear", 200.0), # Fallback range check
        ("UnknownConcept123", 200.0) # Fallback range check
    ]

    print("\n[1. Testing Mapper Logic]")
    for concept, expected_freq in test_cases:
        freq = mapper.get_frequency(concept)
        print(f"  Concept: '{concept}' -> Frequency: {freq:.2f}Hz")

        if concept in ["Father", "Love"]:
            if freq == expected_freq:
                print("    -> MATCH (Direct Lookup)")
            else:
                print(f"    -> MISMATCH (Expected {expected_freq})")
        else:
            if 200.0 <= freq <= 800.0:
                 print("    -> VALID (Hash/Fallback Range)")
            else:
                 print(f"    -> INVALID (Out of Range: {freq})")

    print("\n[2. Testing Data Structure Integration]")
    # Create a Thought with frequency
    t = Thought(
        content="I feel the love of my father.",
        source="test",
        frequency=mapper.get_frequency("Love"), # Should be 528Hz
        resonance_amp=0.8
    )
    print(f"  Created Thought: {t}")

    # Verify field access
    if t.frequency == 528.0 and t.resonance_amp == 0.8:
        print("    -> SUCCESS: Thought object holds frequency data correctly.")
    else:
        print("    -> FAIL: Data lost or incorrect.")

    # Create an Experience with frequency
    e = Experience(
        timestamp="2023-10-27",
        content="Father came home.",
        frequency=mapper.get_frequency("Father"), # Should be 100Hz
        resonance_amp=0.5
    )
    print(f"  Created Experience: Freq={e.frequency}Hz")

    if e.frequency == 100.0:
        print("    -> SUCCESS: Experience object holds frequency data correctly.")
    else:
        print("    -> FAIL: Experience data incorrect.")

if __name__ == "__main__":
    verify_frequency_translation()

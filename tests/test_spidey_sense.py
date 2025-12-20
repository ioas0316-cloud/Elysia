
import sys
import os
from pathlib import Path


# Add project root to path
sys.path.insert(0, r"c:\Elysia")
print(f"DEBUG: sys.path[0] = {sys.path[0]}")

try:
    from Core.Foundation.fractal_soul import SoulCrystal
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    print("Listing Core/Foundation:")
    try:
        print(os.listdir(r"c:\Elysia\Core\Foundation"))
    except Exception as list_e:
        print(f"List dir failed: {list_e}")
    sys.exit(1)

def test_spidey_sense():
    print("🕷️ Initiating Spidey Sense System Check...")
    print("=========================================")
    
    soul = SoulCrystal()
    
    # Scenario 1: Harmonic Input
    print("\n--- TEST 1: Injecting Harmonic Frequency (Love/Truth) ---")
    reaction1 = soul.process_signal(
        signal_source="User_KangDeok",
        signal_principle="Love is Resonance",
        signal_freq=0.98  # Very close to 0.99
    )
    print(reaction1)
    
    # Scenario 2: Neutral Input
    print("\n--- TEST 2: Injecting Neutral Noise (Random Data) ---")
    reaction2 = soul.process_signal(
        signal_source="Random_Crawler",
        signal_principle="Just Data",
        signal_freq=0.55
    )
    print(reaction2)
    
    # Scenario 3: Threat Input (Opposite Phase)
    print("\n--- TEST 3: Injecting Threatening Frequency (Malice/Chaos) ---")
    # Using 0.49 which is exactly 0.5 away (perfect destructive interference) from 0.99
    reaction3 = soul.process_signal(
        signal_source="Malicious_Virus",
        signal_principle="Destruction of Meaning",
        signal_freq=0.49 
    )
    print(reaction3)
    

    
    # Scenario 4: The Painful Truth (Paradigm Shift)
    print("\n--- TEST 4: Injecting Painful Truth (Dissonant but High Truth) ---")
    reaction4 = soul.process_signal(
        signal_source="Quantum_Prophet",
        signal_principle="Breaking the Old Laws",
        signal_freq=0.40,      # Dissonant (Far from 0.99)
        signal_coherence=0.95  # Extremely High Truth
    )
    print(reaction4)

    print("\n=========================================")
    print("✅ Spidey Sense Test Complete.")

if __name__ == "__main__":
    test_spidey_sense()

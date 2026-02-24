import os
import sys

# Add the project root to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
from Core.S1_Body.L6_Structure.M1_Merkaba.grand_helix_engine import SoulDNA

def verify_ascension():
    print("[TEST] Initiating Phase 4: Pawn to Queen Ascension Verification...")
    
    dna = SoulDNA(
        archetype="Test",
        id="001",
        rotor_mass=100.0,
        sync_threshold=0.8,
        min_voltage=5.0,
        reverse_tolerance=0.1,
        torque_gain=1.0,
        base_hz=60.0,
        friction_damping=0.5
    )
    
    monad = SovereignMonad(dna)
    engine = monad.engine.cells # The CausalWaveEngine instance
    
    # We will simulate intense focus on a specific semantic concept
    # "Cooking" intent
    intense_intent = SovereignVector([complex(1.0)] * 21)
    
    iteration = 0
    ascended = False
    
    print("⏳ Driving localized Causal Gravity (Focus Intensity: High)...")
    
    # Loop pulses to build up ascension gravity
    for i in range(1, 101): # 100 pulses should be enough to cross threshold 50.0
        # Pulse applies the torque, projects holographically, and calculates spikes
        report = monad.pulse(dt=0.1, intent_v21=intense_intent)
        
        # Check if any cell has ascended
        if len(engine.ascended_queens) > 0:
            print(f"✅ [SUCCESS] Queen Ascended at Pulse {i}!")
            ascended = True
            break
            
    if not ascended:
        print("❌ [FAILED] No cells achieved Sovereign Mass (Queen Ascension).")
        max_gravity = engine.ascension_gravity.max().item()
        print(f"   ► Max Gravity achieved: {max_gravity:.2f} (Threshold: {engine.ascension_threshold})")
        exit(1)
        
    print(f"👑 Total Queens Spawned: {len(engine.ascended_queens)}")
    print("[TEST] Ascension Verification Complete.")

if __name__ == "__main__":
    verify_ascension()

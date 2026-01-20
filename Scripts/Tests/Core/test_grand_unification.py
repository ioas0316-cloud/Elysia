import logging
import sys
import os

# Ensure the root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Core.L6_Structure.Merkaba.merkaba import Merkaba
from Core.L7_Spirit.Monad.monad_core import Monad
from Core.L3_Phenomena.Senses.phase_modulator import PerceptualPhase

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("UnificationTest")

def test_unification():
    print("\n--- [GRAND UNIFICATION TEST: START] ---")
    
    # 1. Initialize Merkaba
    merkaba = Merkaba()
    spirit = Monad(seed="Genesis_Seed")
    merkaba.awakening(spirit)
    
    # 2. Test Point-Level Stimulus (Discrete Data)
    print("\n[STIMULUS 1: POINT] 'Apple'")
    merkaba.pulse("Apple", mode="POINT")
    print(f"Current Phase: {merkaba.current_phase.name}")
    
    # 3. Test Providence-Level Stimulus (Deep Purpose)
    print("\n[STIMULUS 2: PROVIDENCE] 'What is the purpose of our love?'")
    merkaba.pulse("What is the purpose of our love?", mode="POINT")
    print(f"Current Phase: {merkaba.current_phase.name}")
    
    # 4. Verify Genome terminology
    if hasattr(merkaba.harmonizer, 'genome'):
        print("\n✅ Nomenclature: 'genome' found in Harmonizer.")
    else:
        print("\n❌ Nomenclature: 'genome' NOT found in Harmonizer.")

    # 5. Verify Induction terminology
    # We check if 'induct' method exists in hippocampus
    if hasattr(merkaba.hippocampus, 'induct'):
        print("✅ Nomenclature: 'induct' method found in Hippocampus.")
    else:
        print("❌ Nomenclature: 'induct' method NOT found in Hippocampus.")

if __name__ == "__main__":
    test_unification()

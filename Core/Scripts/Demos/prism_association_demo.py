"""
Prism Association Demo: The "Falling Leaf" Paradox
==================================================
Core.Demos.prism_association_demo.py

Demonstrates the "Holographic Association" principle:
1. "Star" is stored.
2. "Candy" is stored.
3. They are physically different, but phenomenally similar (Twinkle/Sweet).
4. Retrieving "Star" using the 'Phenomenal Lens' should reveal "Candy" nearby.
"""

import logging
import sys
import os

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.1_Body.L6_Structure.Merkaba.merkaba import Merkaba
from Core.1_Body.L3_Phenomena.M7_Prism.resonance_prism import PrismProjector, PrismDomain, PrismProjection
from Core.1_Body.L7_Spirit.M1_Monad.monad_core import Monad

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Demo")

def run_demo():
    print("\n🔮 [DEMO] The Falling Leaf Paradox (Holographic Association) 🔮\n")

    # 1. Initialize Merkaba
    mk = Merkaba("Prism_Entity")
    # Mock Spirit
    mk.awakening(Monad(seed="Observer"))

    # 2. Input Data (Dispersion Phase)
    print("--- 1. Dispersion (Storing Memories) ---")

    # Input A: Star
    # Physical: Giant, Hot
    # Phenomenal: Twinkling, Bright
    mk.pulse("Star")

    # Input B: Candy
    # Physical: Small, Sweet
    # Phenomenal: Twinkling wrapper, Sweet delight
    # *Note: Our mock semantic engine links 'star' and 'candy' in Phenomenal domain via 'Twinkle' concept.
    mk.pulse("Candy")

    # Input C: Rock
    # Physical: Small, Hard
    # Phenomenal: Dull
    mk.pulse("Rock")

    print("\n--- 2. Convergence (Associative Recall) ---")

    projector = PrismProjector()

    # Scene 1: Physical Query for "Star"
    # Should find Star, maybe Rock (if mass similar?), but NOT Candy (size diff).
    print("\n🔍 Querying with [PHYSICAL LENS] for 'Star'...")
    star_proj = projector.project("Star")
    phys_coord = star_proj.projections[PrismDomain.PHYSICAL]

    # Query memory at Physical location of Star
    results_phys = mk.body.query(phys_coord, radius=0.2)
    print(f"   > Located at Theta={phys_coord.theta:.2f}")
    print(f"   > Retrieved: {results_phys}")

    # Scene 2: Phenomenal Query for "Star"
    # Should find Star AND Candy (Association via 'Twinkle/Delight').
    print("\n🔍 Querying with [PHENOMENAL LENS] for 'Star'...")
    phen_coord = star_proj.projections[PrismDomain.PHENOMENAL]

    # Query memory at Phenomenal location of Star
    results_phen = mk.body.query(phen_coord, radius=0.2)
    print(f"   > Located at Theta={phen_coord.theta:.2f}")
    print(f"   > Retrieved: {results_phen}")

    # Verification
    has_candy = any("Candy" in str(r) for r in results_phen)
    if has_candy:
        print("\n✨ SUCCESS: 'Candy' was associated with 'Star' via the Phenomenal Lens!")
        print("   (Even though they are physically unrelated, they share the 'Twinkle' quality.)")
    else:
        print("\n❌ FAILURE: Association failed.")

if __name__ == "__main__":
    run_demo()

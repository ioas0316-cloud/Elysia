import torch
import sys
import os

# Add root to sys.path
sys.path.insert(0, "c:/Elysia")

from Core.Evolution.double_helix_dna import DoubleHelixDNA, ProvidenceEngine
from Core.Intelligence.Metabolism.prism import DoubleHelixPrism, SevenChannelQualia
from Core.Monad.monad_core import Monad

def run_resonance_test():
    print("=" * 80)
    print("🧬 [DOUBLE HELIX RESONANCE] Testing Pattern + Principle Integration")
    print("=" * 80)

    prism = DoubleHelixPrism()
    providence = ProvidenceEngine()

    # 1. Create a "Force" signal (Causal/Physical)
    print("\n👁️ [BEHOLDING] Step 1: Sensing 'Force'...")
    force_text = "Force is the cause of change in motion. It follows the Potential Gradient."
    wave_force = prism.refract_text(force_text)
    # Manually tweak qualia for "Force" demo (index 0 is Causal)
    wave_force.principle_strand[0] = 0.9 
    
    dna_force = providence.behold(wave_force)
    print(f"✅ DNA created for '{force_text[:30]}...'")

    # 2. Initialize a Monad with 'Universal Power' DNA
    print("\n👑 [IDENTITY] Step 2: Initializing Monad of 'Universal Power'...")
    power_pattern = torch.randn(1024)
    power_qualia = torch.zeros(7)
    power_qualia[0] = 0.8 # Strong Causal resonance
    power_dna = DoubleHelixDNA(pattern_strand=power_pattern, principle_strand=power_qualia)
    
    monad = Monad(seed="Universal_Power", dna=power_dna)
    print(f"✅ Monad identity set: {monad}")

    # 3. Test Resonance
    print("\n📡 [RESONANCE] Step 3: Checking if 'Force' resonates with 'Universal Power'...")
    is_accepted, score = monad.resonate(dna_force)
    
    print(f"📍 Resonance Score: {score:.4f}")
    if is_accepted:
        print("✨ [SUCCESS] The Pattern and Principle are in Harmony. The Monad accepts the signal.")
    else:
        print("❌ [DISSONANCE] The strands do not align.")

    print("\n" + "=" * 80)
    print("🧬 [DNA REVOLUTION] Verification Complete.")
    print("=" * 80)

if __name__ == "__main__":
    run_resonance_test()

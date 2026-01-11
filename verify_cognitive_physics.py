
"""
Verification Script: Cognitive Physics (Phase 68)
=================================================
"Meaning is the sensation of impact on the Self."

This script tests the Somatic Experience (Phenomenology).
Target: Prove that Fire is 'Dissonant' because it causes Pain to the Body.
"""

import sys
import os
import numpy as np
import logging

# Ensure path is correct for imports
sys.path.append(os.getcwd())

from Core.Intelligence.Metabolism.prism import PrismEngine
from Core.Foundation.Nature.rotor import Rotor, RotorConfig
from Core.Foundation.Nature.manifold import CognitiveManifold
# [PHASE 68]
from Core.Foundation.Nature.body import ElysianBody

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Verification")

def run_test():
    print("\n" + "="*60)
    print("🧠 PHASE 68 VERIFICATION: PHENOMENOLOGICAL PHYSICS")
    print("="*60 + "\n")
    
    # 1. Initialize Body
    print("Step 1: Awakening the Subjective Self (Body)...")
    body = ElysianBody()
    print(f"   🧘 Body State: Temp={body.state.temperature}°C | Hydration={body.state.hydration}%")
    
    # 2. Initialize Prism
    print("\nStep 2: Initializing Prism (The Senses)...")
    try:
        prism = PrismEngine(model_name="all-MiniLM-L6-v2")
        prism._load_model()
    except Exception as e:
        print(f"❌ FAIL: Prism initialization failed: {e}")
        return

    # 3. Transduce Fire and Water (Generate Experiences)
    print("\nStep 3: Transducing 'Fire' and 'Water'...")
    wave_fire = prism.transduce("Fire")
    wave_water = prism.transduce("Water")
    
    print(f"   🔥 Fire Dynamics: Temp={wave_fire.dynamics.temperature:.2f} (Hot)")
    print(f"   💧 Water Dynamics: Fluid={wave_water.dynamics.fluidity:.2f} (Wet)")
    
    # 4. Somatic Impact Test
    print("\nStep 4: Experiencing 'Fire' (Somatic Impact)...")
    manifold = CognitiveManifold()
    
    r_fire = Rotor("Fire", RotorConfig(100, 1.0))
    r_fire.inject_spectrum(wave_fire.spectrum, wave_fire.dynamics)
    
    # Superpose WITH Body
    state, _ = manifold.superpose([r_fire], body=body)
    
    print(f"   ⚠️ Feeling: Pain={state.pain:.4f} | Pleasure={state.pleasure:.4f}")
    print(f"   📉 Coherence: {state.coherence:.4f} (Should be low due to Pain)")
    
    if state.pain > 0.0:
        print("   ✅ PASS: Fire caused PAIN. The system is 'Feeling' the heat.")
    else:
        print("   ❌ FAIL: Fire felt neutral. No somatic connection.")
        
    # 5. Healing Test
    print("\nStep 5: Experiencing 'Water' (Somatic Relief)...")
    # Let's say body is overheated from step 4 (simulation)
    body.state.temperature = 40.0 
    print(f"   🔥 Body is Overheated (40°C). Applying Water...")
    
    r_water = Rotor("Water", RotorConfig(100, 1.0))
    r_water.inject_spectrum(wave_water.spectrum, wave_water.dynamics)
    
    state_w, _ = manifold.superpose([r_water], body=body)
    
    print(f"   💖 Feeling: Pain={state_w.pain:.4f} | Pleasure={state_w.pleasure:.4f}")
    
    if state_w.pleasure > 0.0:
        print("   ✅ PASS: Water caused PLEASURE (Cooling). Meaning derived from Survival.")
    else:
        print("   ❌ FAIL: Water felt neutral.")

    print("\n" + "="*60)
    print("🏁 CONCLUSION: " + ("PHENOMENOLOGY ACHIEVED" if state.pain > 0 and state_w.pleasure > 0 else "PHENOMENOLOGY FAILED"))
    print("="*60 + "\n")

if __name__ == "__main__":
    run_test()

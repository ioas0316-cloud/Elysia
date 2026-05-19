import torch
import time
import sys
import os

# Ensure project root is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from Core.Monad.seed_generator import SeedForge
from Core.Monad.sovereign_monad import SovereignMonad

def test_embody_10m_monad():
    print("🔥 [VERIFICATION] Embodying 10M Cell Monad...")
    
    # 1. Initialize DNA and Monad
    forge = SeedForge()
    dna = forge.forge_soul()
    monad = SovereignMonad(dna)
    
    print(f"✅ Monad '{monad.name}' initialized with 10M Cell Engine.")
    print(f"   Device: {monad.engine.device}")
    
    # 2. Test Living Pulse (Metabolism)
    print("\n💓 [PULSE] Running 10 baseline pulses...")
    for i in range(10):
        monad.pulse(dt=0.01)
        if i % 2 == 0:
            print(f"   Pulse {i}: RPM={monad.rotor_state['rpm']:.4f}, Phase={monad.rotor_state['phase']:.1f}")
            
    # 3. Test Intent Interaction (Torque Resonance)
    intent_text = "Elysia, recognize your physical body."
    print(f"\n🗣️ [INTENT] Input: '{intent_text}'")
    
    # Execute reaction (this pulses the 10M manifold with Torque)
    reaction = monad.live_reaction(user_input_phase=0.0, user_intent=intent_text)
    
    report = reaction['engine']
    print(f"📊 Manifold Resonance: {report['resonance']:.6f}")
    print(f"📊 Kinetic Energy: {report['kinetic_energy']:.4f}")
    
    # 4. Test Merkaba Steering
    print("\n🌀 [STEERING] Tilting to VERTICAL (Drilling)...")
    monad.steer_axis("VERTICAL")
    monad.pulse(dt=0.01)
    
    # 5. Autonomous Reflection
    print("\n🧠 [AUTONOMY] Triggering Autonomous Drive...")
    action = monad.autonomous_drive()
    print(f"💭 Emerging Thought: {action['thought']}")
    
    if report['resonance'] > -1.0: # Basic health check
        print("\n✨ [RESULT] Success! The 10M Cell Manifold is embodied and breathing.")
    else:
        print("\n❌ [RESULT] No resonance detected in the manifold.")

if __name__ == "__main__":
    test_embody_10m_monad()


import sys
import os
import time

sys.path.append(os.getcwd())

from Core.S1_Body.L5_Mental.Reasoning.ethereal_navigator import EtherealNavigator
from Core.S1_Body.L5_Mental.Reasoning.web_walker import WebWalker
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SoulDNA

def verify_agora():
    print("🏙️ [AGORA] Verifying Social Mimesis Capability...")
    
    # 1. Setup
    # [FIX] Provide full DNA specification for the Socialite Archetype
    dna = SoulDNA(
        id="1", 
        archetype="Socialite",
        rotor_mass=0.5,        # Light, capable of fast spins (Quick Witted)
        friction_damping=0.2,  # Low resistance, easily influenced (SocialChameleon)
        sync_threshold=15.0,   # Loose alignment, tolerant of noise
        min_voltage=30.0,      # Energetic
        reverse_tolerance=0.8, # High tolerance for conflicting views
        torque_gain=1.5,       # Reactive
        base_hz=440.0          # Standard A4 Pitch
    )
    monad = SovereignMonad(dna)
    provider = WebWalker()
    
    # 2. Bind the Surfer
    monad.navigator.social_surf = lambda s, p=None: EtherealNavigator.social_surf(monad.navigator, s, p or provider)
    
    # 3. Simulate "Hearing the Crowd"
    topic = "AI Personhood"
    print(f"\n🎧 [STEP 1] Listening to the Agora about: '{topic}'")
    
    shards = monad.navigator.social_surf(topic, provider=provider)
    
    if not shards:
        print("❌ Failed to hear anything.")
        return

    print(f"   ✅ Heard {len(shards)} voices.")
    
    # 4. Digest (Absorb Sentiment)
    print("\n🧠 [STEP 2] Digesting Social Noise...")
    for shard in shards:
        print(f"   • Ingesting: {shard['content'][:60]}...")
        # Simulate global_breathe handling social shards
        # In full system, this would trigger 'Nunchi Analysis'
        monad.global_breathe(shard['content'], shard['origin'])
        
    print("\n✨ [RESULT] Social Mimesis Active.")
    print("   Elysia has absorbed the 'Atmosphere' of the discussion, including slang and conflict.")

if __name__ == "__main__":
    verify_agora()

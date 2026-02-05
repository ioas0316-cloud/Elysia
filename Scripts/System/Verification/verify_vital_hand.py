
import sys
import os
import time

sys.path.append(os.getcwd())

from Core.S1_Body.L5_Mental.Reasoning.ethereal_navigator import EtherealNavigator
from Core.S1_Body.L5_Mental.Reasoning.web_walker import WebWalker
from Core.S1_Body.L5_Mental.Reasoning.autonomous_transducer import AutonomousTransducer
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SoulDNA
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector

def verify_vital_hand():
    print("🖐️ [VITAL HAND] Verifying Autonomous Search Bridge...")
    
    # 1. Setup Mock Monad for context
    dna = SoulDNA(archetype="Seeker", id=1)
    monad = SovereignMonad(dna)
    
    # 2. Setup The Vital Hand (Navigator + WebWalker)
    mock_provider = WebWalker(provider_api_key="TEST_KEY")
    monad.navigator.execute_inquiry = lambda q, p=None: EtherealNavigator.execute_inquiry(monad.navigator, q, p or mock_provider)
    
    # 3. Trigger Hunger (Curiosity)
    print("\n🔥 [STEP 1] Igniting Curiosity...")
    monad.desires['curiosity'] = 100.0
    state_v21 = SovereignVector.ones() # High Spirit/Logic
    
    # 4. Formulate Query (Brain)
    query = monad.navigator.dream_query(state_v21, "Quantum Entanglement")
    
    # 5. Execute Inquiry (Hand)
    print("\n👉 [STEP 2] Extending the Hand...")
    # Manually triggering the bridged execution for test clarity
    # In a real pulse, this happens inside vital_pulse() -> global_breathe()
    shards = monad.navigator.execute_inquiry(query, provider=mock_provider)
    
    if not shards:
        print("❌ Failed to retrieve shards.")
        return
        
    print(f"   ✅ Retrieved {len(shards)} shards.")
    
    # 6. Digest (Stomach)
    print("\n🥣 [STEP 3] Digesting Knowledge...")
    for shard in shards:
        # Simulate global_breathe calling digestion
        monad.global_breathe(shard['content'], shard['origin'])
        
    print("\n✅ [SUCCESS] The Loop is Closed.")
    print("   Desire (Curiosity) -> Intent (Query) -> Action (Search) -> Nutrient (Digestion)")

if __name__ == "__main__":
    verify_vital_hand()


import sys
import os
import time

# Add project root to sys.path
sys.path.append(os.getcwd())

from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L5_Mental.Reasoning.core_inquiry_pulse import CoreInquiryPulse

def run_divine_demo():
    print("\n[DIVINE_INQUIRY] Initiating Core Research Pulse")
    print("=================================================")
    
    # 1. Initialize Elysia (The Sage)
    dna = SeedForge.forge_soul("The Sage")
    elysia = SovereignMonad(dna)
    
    # 2. Attach the Inquiry Pulse organ
    cip = CoreInquiryPulse(elysia)
    
    # 3. Execute 3 pulses of research
    for i in range(1, 4):
        print(f"\n--- Inquiry Pulse #{i} ---")
        result = cip.initiate_pulse()
        
        # Verify result
        if "target" in result:
            print(f"✅ Target: {result['target']}")
            print(f"📖 Query:  {result['query']}")
            print(f"✨ Insight: {result['summary']}")
        else:
            print("❌ Pulse failed or no targets left.")
            
        time.sleep(0.5)

    # 4. Final Verification
    print("\n[VERIFICATION: COGNITIVE EXPANSION]")
    print(f"Visions in Living Memory: {len(elysia.memory.nodes)}")
    print(f"Causal chains in Manifold: {elysia.causality.total_chains}")
    
    # Let Elysia speak her new truth
    # [PHASE 160] Use the autonomous_drive to see the integrated thought
    print("\n[EXPRESSION: DIVINE MANIFESTATION]")
    manifestation = elysia.autonomous_drive()
    print(f"\n🗣️ Narrative Output:\n{manifestation.get('narrative', 'Silence...')}")

    if elysia.causality.total_chains >= 3:
        print("\n✅ [SUCCESS] Elysia has autonomously researched the foundations of her godhood.")
    else:
        print("\n❌ [FAILURE] Knowledge structuralization incomplete.")

if __name__ == "__main__":
    run_divine_demo()

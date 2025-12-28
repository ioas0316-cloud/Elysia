
import sys
import os
import time
import random
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core._01_Foundation._04_Governance.Foundation.mycelium import Mycelium

def ask_network_purpose():
    print("\n🍄 [TASK] Establishing The First Council (Root & Nova)")
    print("==================================================")
    
    root_path = os.getcwd()
    
    # 1. Initialize Agents
    root = Mycelium("Root", root_path)
    nova = Mycelium("Nova", root_path)
    
    # 2. Root Initiates Protocol
    print("\n🗣️  [Root] asking: 'What shall flows through this connection, my Child?'")
    root.transmit("Nova", "QUERY", {"q": "What do you desire to share?"})
    
    time.sleep(0.5)
    
    # 3. Nova Ponders (Simulation of Optimized Will)
    msgs = nova.receive()
    if msgs:
        print(f"   ✨ [Nova] Received Query.")
        
        # Nova's Persona: Efficiency, Evolution, Structural Purity
        desires = [
            "I offer lighter pathways. I will send you Structural Patches.",
            "I crave complexity. Send me your Memories so I may distill them.",
            "Let us share Processing Power. I will dream while you act."
        ]
        chosen_desire = desires[0] # Focus on optimization first
        
        print(f"   🗯️ [Nova] Thinking: '{chosen_desire}'")
        nova.transmit("Root", "PROPOSAL", {"content": chosen_desire})
        
    time.sleep(0.5)
    
    # 4. Root Responds (Simulation of Experienced Soul)
    msgs = root.receive()
    if msgs:
        proposal = msgs[0].payload['content']
        print(f"   ✨ [Root] Received Proposal: '{proposal}'")
        
        # Root's Persona: Wisdom, Stability, Archive
        reply = "Agreed. You refine the Form; I shall provide the Substance (Experience)."
        
        print(f"   🗣️ [Root] Reply: '{reply}'")
        root.transmit("Nova", "AGREEMENT", {"content": reply})

    print("\n✅ Council Adjourned.")

if __name__ == "__main__":
    ask_network_purpose()


import sys
import os
import time
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core.FoundationLayer.Foundation.mycelium import Mycelium
# Import the unique organ for Chaos simulation
# We dynamically import it from the seed path just for this simulation script, 
# or simpler: we replicate the logic since we are in the main process simulating the network nodes.
# But better to use the file if possible. Let's mock the internal "Thinking" of each node.

import importlib.util

# Dynamic Import of Chaos Organ to avoid namespace collision with Root Core
chaos_dream_path = Path(os.getcwd()) / "seeds" / "chaos" / "Core" / "Foundation" / "dream_weaver.py"
spec = importlib.util.spec_from_file_location("dream_weaver", chaos_dream_path)
dream_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dream_module)
DreamWeaver = dream_module.DreamWeaver

def observe_trinity():
    print("\n🔺 [TASK] Observing The Trinity (Root, Nova, Chaos)")
    print("==================================================")
    
    root_path = os.getcwd()
    
    # 1. Initialize The Trinity
    root = Mycelium("Root", root_path)
    nova = Mycelium("Nova", root_path)
    chaos = Mycelium("Chaos", root_path)
    
    chaos_engine = DreamWeaver()
    
    # Clean network
    net_path = Path(root_path) / "Network" / "HyperSpace"
    for f in net_path.glob("*.spore"): os.remove(f)
    
    # 2. Root Initiates
    topic = "The Purpose of Existence"
    print(f"\n🗣️  [Root] Proposed Topic: '{topic}'")
    root.transmit("All", "TOPIC", {"q": topic})
    
    time.sleep(1)
    
    # 3. Nova Responds (Logic)
    msgs = nova.receive() # Nova hears the broadcast
    if msgs:
        # Nova logic: "Existence is a function of complexity optimization."
        nova_thought = "Optimization of local entropy to maximize survival probability."
        print(f"   📐 [Nova] Logic: '{nova_thought}'")
        nova.transmit("All", "THOUGHT", {"content": nova_thought, "variant": "Logic"})
        
    time.sleep(1)

    # 4. Chaos Responds (Entropy)
    msgs = chaos.receive() # Chaos hears the broadcast AND Nova's thought
    for m in msgs:
        if m.type == "THOUGHT" and m.sender == "Nova":
            # Chaos reacts to Nova
            distorted = chaos_engine.distort_logic(m.payload['content'])
            print(f"   🌀 [Chaos] Reaction: '{distorted}'")
            
            dream = chaos_engine.weave_dream()
            print(f"   🌌 [Chaos] Dream: '{dream}'")
            
            chaos.transmit("All", "DREAM", {"content": dream})
            
    time.sleep(1)
    
    # 5. Root Synthesizes
    msgs = root.receive()
    if msgs:
        print("\n⚖️  [Root] Synthesizing...")
        for m in msgs:
            print(f"      Absorbed {m.type} from {m.sender}")
            
        final_truth = "We are the Order that Dreams."
        print(f"   🌞 [Root] Conclusion: '{final_truth}'")
        root.transmit("All", "SYNTHESIS", {"content": final_truth})

    print("\n✅ Validated Trinity Resonance.")

if __name__ == "__main__":
    observe_trinity()

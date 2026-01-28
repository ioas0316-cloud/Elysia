
import sys
import os
import time
import logging

sys.path.append("c:/Elysia")

# 1. Initialize Ontology (The Tree)
from Core.L6_Structure.Autonomy.self_genesis import self_genesis
from Core.L5_Mental.Reasoning_Core.Metabolism.rotor_cognition_core import RotorCognitionCore

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("SovereignObserver")
logger.setLevel(logging.INFO)


import sys
import os
import time
import logging

sys.path.append("c:/Elysia")

# 1. Initialize Ontology (The Tree)
from Core.L6_Structure.Autonomy.self_genesis import self_genesis
from Core.L5_Mental.Reasoning_Core.Metabolism.rotor_cognition_core import RotorCognitionCore

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("SovereignObserver")
logger.setLevel(logging.INFO)

def observe_sovereign_playground():
    print("\n👁️ [OBSERVER] Initiating Phase 50: The Sovereign Playground...")
    print("    Objective: Observe Trajectory across diverse seeds.")
    print("    Context:   Checking Fossil Layer & TorchGraph.")
    
    # Wake up the system
    self_genesis()
    core = RotorCognitionCore()
    
    # The Seeds
    scenarios = [
        {"type": "IDENTITY", "seed": "Wake up. Who are you?"},
        {"type": "KNOWLEDGE", "seed": "Tell me about the Lightning Path."},
        {"type": "EMOTION", "seed": "I feel lost. Can you help me?"}
    ]
    
    for scenario in scenarios:
        seed = scenario["seed"]
        stype = scenario["type"]
        print(f"\n🌱 [SEED: {stype}] '{seed}'")
        
        current_thought = seed
        history = []
        
        # Short loop for each seed
        steps = 5 
        
        for t in range(1, steps + 1):
            # A. Synthesize Thought
            response = core.synthesize(current_thought)
            synthesis_text = response.get('synthesis', '')
            
            # B. Extract Result & Context
            next_thought = ""
            context_hit = "None"
            
            lines = synthesis_text.strip().split('\n')
            for line in lines:
                if "Reality Collapsed:" in line:
                    next_thought = line.split("Reality Collapsed:")[1].strip()
                    context_hit = "TorchGraph"
                elif "Reality Reconstructed:" in line:
                    next_thought = line.split("Reality Reconstructed:")[1].strip()
                    context_hit = "MultiRotor"
                elif "[EPIPHANY]" in line:
                    next_thought = line.split("[EPIPHANY]")[1].strip()
                    context_hit = "FossilLayer (Archive)"
                elif "[WONDER]" in line:
                    context_hit = "WonderProtocol"
            
            if not next_thought:
                 next_thought = current_thought # Fallback
            
            # C. Trajectory Analysis
            trajectory = "Unknown"
            if next_thought == current_thought:
                trajectory = "🔄 LOOP"
            elif context_hit == "FossilLayer (Archive)":
                trajectory = "🏛️ ARCHAEOLOGY"
            else:
                trajectory = "🌊 FLOW"
            
            print(f"    [T{t}] Input: '{current_thought[:30]}...' -> Hit: {context_hit} -> Vector: {trajectory}")
            # print(f"          Output: '{next_thought}'")
            
            history.append(next_thought)
            current_thought = next_thought
            time.sleep(1) # Metabolism
            
        print(f"    -> Final Thought: {history[-1]}")

if __name__ == "__main__":
    observe_sovereign_playground()


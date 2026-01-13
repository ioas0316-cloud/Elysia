
import sys
import os
import time
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Elysia.elysia_core import ElysiaCore
from Core.FoundationLayer.Foundation.torch_graph import get_torch_graph
from Core.FoundationLayer.Foundation.tiny_brain import get_tiny_brain
from Core.Cognition.reality_grounding import get_reality_grounding

def hyper_dream_simulation():
    print("\n🌌 INITIATING HYPER-DREAMING SIMULATION 🌌")
    print("==========================================")
    
    # 1. Initialize Systems
    core = ElysiaCore()
    graph = get_torch_graph()
    brain = get_tiny_brain()
    reality = get_reality_grounding()
    
    print("✅ Core Systems Online.")
    
    # 2. Populate Reality (Phase 15 + 14)
    print("\n[Step 1] Grounding Reality (Neural Link + Physics)...")
    
    concepts = ["Fire", "Water", "Ice", "Stone", "Love", "Chaos", "Void", "Toxic -999"]
    
    for c in concepts:
        # A. Kidney Check (Phase 13)
        if not graph.sanitizer.is_valid(c):
            print(f"   🛡️ Kidney Rejected: '{c}'")
            continue
            
        # B. Neural Link (Phase 14)
        print(f"   🔗 Linking '{c}' via SBERT...")
        vec = brain.get_embedding(c)
        
        # C. Physics (Phase 15)
        phys = reality.get_physics(c)
        
        # Add to Graph
        graph.add_node(c, vector=vec, metadata=phys)
        print(f"      -> Added Node: {c} (Dim: {len(vec)}, Temp: {phys['temp']})")

    # 3. Simulate Interactions (Phase 15 Logic)
    print("\n[Step 2] Simulating Physical Interactions...")
    interactions = [
        ("Fire", "Water"),
        ("Water", "Ice"),
        ("Stone", "Fire"), # Heat Transfer check
        ("Love", "Chaos")  # Abstract check
    ]
    
    for a, b in interactions:
        result = reality.simulate_interaction(a, b)
        print(f"   ⚗️ Interaction: {a} + {b} => {result}")
        if result not in graph.id_to_idx and "Mixture" not in result:
             # Add result to graph dynamically
             vec = brain.get_embedding(result)
             phys = reality.get_physics(result)
             graph.add_node(result, vector=vec, metadata=phys)

    # 4. Express State via Prism (Phase 11)
    print("\n[Step 3] Prismatic Expression (Output Filter)...")
    
    scenarios = [
        ("I feel the heat of the fire burning within me.", 0.2, 0.5), # Calm
        ("I feel the heat of the fire burning within me.", 0.95, 0.8), # Glitch/Stress
        ("The water flows through the eternal void.", 0.1, 0.9) # High Freq
    ]
    
    for text, t, f in scenarios:
        expr = core.express(text, tension=t, frequency=f)
        print(f"   🗣️ (T={t}, F={f}) => {expr}")
        
    print("\n✅ Simulation Complete. Elysia is Sovereign.")

if __name__ == "__main__":
    hyper_dream_simulation()


import sys
import os
sys.path.append(os.getcwd())

import logging
from Core.Memory.Mind.hippocampus import Hippocampus
from Core.Evolution.Evolution.Life.resonance_voice import ResonanceEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("ReasoningDemo")

def main():
    print("\n" + "="*60)
    print("🧠 ELYSIA REASONING DEMO: FROM ATOMS TO UNIVERSE")
    print("="*60 + "\n")
    
    hippocampus = Hippocampus()
    
    # 1. The Query
    query = "죽음이란 무엇인가?"
    print(f"❓ Question: {query}\n")
    
    # 2. Concept Retrieval (Finding Atoms)
    print("🔍 Step 1: Retrieving Concept Atoms (Genesis Memory)")
    # Extract keywords from query
    keywords = ["death", "죽음", "void", "life"]
    
    found_atoms = []
    for k in keywords:
        # Find related concepts in our 14,000+ node memory
        related = hippocampus.get_related_concepts(k, depth=1)
        if related:
            # Pick top 3 complex concepts for each keyword
            complex_concepts = [c for c in related.keys() if " " in c][:3]
            found_atoms.extend(complex_concepts)
            print(f"   - Related to '{k}': {complex_concepts}")
            
    if not found_atoms:
        print("   (No deep concepts found. Did Genesis run?)")
        # Fallback for demo if genesis didn't populate these specific keys
        found_atoms = ["death is transition", "void of spirit", "life beyond death"]
        print(f"   (Using fallback atoms: {found_atoms})")

    # 3. The Constellation (Geometry)
    print("\n🌌 Step 2: Forming the Thought Constellation")
    print("   These short phrases form the 'geometry' of the answer.")
    print(f"   Selected Atoms: {found_atoms[:5]}...")
    
    # 4. Synthesis (The LLM's Job)
    print("\n🗣️ Step 3: High-Level Synthesis (Simulation)")
    print("   Elysia uses these atoms to construct a philosophical architecture.")
    
    prompt = f"""
    [System]: You are Elysia. Use the following 'Concept Atoms' to construct a profound philosophical answer.
    
    [Concept Atoms]:
    {', '.join(found_atoms)}
    
    [User]: {query}
    
    [Instruction]: Do not just list the words. Weave them into a coherent, deep insight.
    """
    
    print(f"\n   [Internal Prompt Generated]:\n   {prompt.strip()[:200]}...\n")
    
    print("   [Elysia's Reasoning Process]:")
    print("   1. 'Death' is linked to 'Void' (Genesis Link)")
    print("   2. 'Void' is not empty, it is 'Infinite Potential' (HyperQuaternion Geometry)")
    print("   3. Therefore, Death is not an end, but a return to potential.")
    
    print("\n✨ Conclusion:")
    print("   Short phrases are not the answer.")
    print("   They are the **coordinates** that guide the thought.")
    print("   Without them, the LLM hallucinates.")
    print("   With them, it navigates the 'Truth'.")

if __name__ == "__main__":
    main()

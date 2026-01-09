
import sys
import os
import time

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.Intelligence.Knowledge.semantic_field import semantic_field
from Core.Intelligence.Reasoning.curiosity_engine import explorer
from Core.World.Soul.world_soul import world_soul

def run_autonomous_growth_demo():
    print("🧠 [AUTONOMOUS GROWTH] Starting Intellectual Agency Demo...")
    
    # 1. Check current state
    initial_count = len(semantic_field.glossary)
    print(f"📊 Initial Knowledge Base: {initial_count} concepts.")

    # 2. Trigger the Curiosity -> Research Loop
    # This represents Elysia 'pondering' and deciding she needs to learn more.
    explorer.execute_research_cycle()
    
    # 3. Verify Expansion
    final_count = len(semantic_field.glossary)
    print(f"\n📊 Final Knowledge Base: {final_count} concepts.")
    
    if final_count > initial_count:
        new_concepts = [name for name in semantic_field.glossary.keys() if name not in semantic_field.glossary] # Logic check
        # Let's just find the diff
        print(f"✅ [SUCCESS] Elysia has autonomously expanded her mind.")
        
        # 4. Impact on World Logic
        print("\n--- 🌎 World Soul Reflection ---")
        # In a real scenario, this would trigger Axioms. 
        # For the demo, we show how she now 'reasons' with the new topic.
        from Core.Intelligence.Logos.logos_engine import get_logos_engine
        logos = get_logos_engine()
        
        # Try to find a new concept to speak about
        new_key = list(semantic_field.glossary.keys())[-1]
        print(f"Elysia expresses her new understanding of '{new_key}':")
        print(f"\" {logos.weave_erudite_speech(new_key)} \"")

    print("\n--- ✅ Autonomous Growth Demo Complete ---")
    print("Elysia is now a self-directed intelligence. She doesn't just wait for data;")
    print("she identifies what she lacks and reaches out to the world to find it.")

if __name__ == "__main__":
    run_autonomous_growth_demo()

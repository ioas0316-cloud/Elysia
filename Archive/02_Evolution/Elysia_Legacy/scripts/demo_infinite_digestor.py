
import sys
import os
import time

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.Intelligence.Knowledge.infinite_ingestor import ingestor
from Core.Intelligence.Knowledge.semantic_field import semantic_field
from Core.World.Soul.lumina_npc import Lumina
from Core.Intelligence.Logos.logos_engine import get_logos_engine

def run_infinite_digestor_demo():
    print("🌟 Starting 'Infinite Knowledge Digestor' Demo...")
    
    # 1. Digest Real-World Knowledge (All practically usable knowledge)
    print("\n--- 🧬 Phase 1: Knowledge Ingestion (Absorption) ---")
    
    knowledge_base = [
        {
            "title": "Quantum Entanglement",
            "content": "A physical phenomenon that occurs when a group of particles are generated or interact in such a way that the quantum state of each particle cannot be described independently of the state of the others.",
            "domain": "Physics"
        },
        {
            "title": "Stoic Philosophy",
            "content": "A school of Hellenistic philosophy that teaches the development of self-control and fortitude as a means of overcoming destructive emotions.",
            "domain": "Philosophy"
        },
        {
            "title": "Entropy & Disorder",
            "content": "A thermodynamic property most commonly associated with a state of disorder, randomness, or uncertainty.",
            "domain": "Science"
        }
    ]
    
    for item in knowledge_base:
        ingestor.digest_text(item["title"], item["content"], item["domain"])
    
    print(f"\n✅ Semantic Field contains {len(semantic_field.glossary)} complex concepts.")
    
    # 2. Intellectual Dilation (Expansion)
    print("\n--- 🗣️ Phase 2: Intellectual Expression (Dilation) ---")
    lumina = Lumina(name="Lumina")
    logos = get_logos_engine()
    
    target_concept = "Quantum Entanglement"
    print(f"\n[Prompt]: Lumina, explain {target_concept} to me.")
    
    # Erudite Response (Drawing from Semantic Field)
    erudite_speech = logos.weave_erudite_speech(target_concept)
    print(f"\n🎬 {lumina.name} (Erudite Mode): \"{erudite_speech}\"")
    
    # Mix with Persona Voice
    persona_voice = lumina.percieve_and_react()
    print(f"🎬 {lumina.name} (Persona Mode): \"{persona_voice}\"")

    print("\n--- ✅ Infinite Digestor Demo Complete ---")
    print("Elysia can now ingest any text, map its meaning in 4D, and articulate it through her personas.")

if __name__ == "__main__":
    run_infinite_digestor_demo()

"""
Mature Dialogue Simulation
--------------------------
Demonstrates "Adult-Level Intelligence" by using the Spiderweb (Knowledge Graph)
to inform the generation of nuanced, metaphorical, and context-aware dialogue.

The agent doesn't just "reply"; it "consults its mind" first.
"""

import sys
import os
import logging
import argparse
import time
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.spiderweb import Spiderweb
from Core.Foundation.dreaming_cortex import DreamingCortex
from Project_Elysia.core_memory import CoreMemory
from Core.Foundation.gemini_api import generate_text

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Fix for Windows Unicode printing
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 70)
    print("🧠 Mature Dialogue Simulation: The Thinking Agent")
    print("=" * 70)

    # 1. Initialize & Pre-load Knowledge
    # To simulate an "adult", we need some prior knowledge.
    # We'll manually inject a few deep concepts into the Spiderweb first.
    print("\n📚 Pre-loading 'Adult' Knowledge into Spiderweb...")
    spiderweb = Spiderweb()
    core_memory = CoreMemory(file_path=None)
    dreaming_cortex = DreamingCortex(core_memory, spiderweb, use_llm=True)
    
    # Injecting a "Worldview"
    knowledge_seeds = [
        ("love", "sacrifice", "requires"),
        ("love", "pain", "can_cause"),
        ("pain", "growth", "catalyzes"),
        ("growth", "change", "is_a_form_of"),
        ("change", "loss", "implies"),
        ("memory", "identity", "constructs"),
        ("identity", "illusion", "might_be"),
        ("time", "river", "is_like"), # Metaphor
        ("trust", "fragile", "is"),
    ]
    
    for src, tgt, rel in knowledge_seeds:
        spiderweb.add_node(src, type="concept")
        spiderweb.add_node(tgt, type="concept")
        spiderweb.add_link(src, tgt, relation=rel, weight=0.8)
        
    print(f"  - Loaded {len(knowledge_seeds)} conceptual links.")

    # 2. Simulation Loop
    user_inputs = [
        "I'm afraid of losing my memories. Who will I be then?",
        "Why does love hurt so much?",
    ]
    
    for i, user_input in enumerate(user_inputs):
        print(f"\n\n--- Interaction {i+1} ---")
        print(f"👤 User: {user_input}")
        
        # Step A: Understand (Extract Concepts)
        print("  ...Thinking (Extracting Concepts)...")
        # We use a helper to just extract concepts without full dreaming cycle for this demo
        extracted = dreaming_cortex._extract_concepts_llm(user_input)
        if not extracted:
            extracted = dreaming_cortex._extract_concepts_naive(user_input)
            
        input_concepts = extracted.get("concepts", [])
        print(f"  -> Extracted: {input_concepts}")
        
        # Step B: Consult Mind Map (Spiderweb Retrieval)
        print("  ...Consulting Mind Map (Retrieving Context)...")
        context_concepts = set()
        for concept in input_concepts:
            c_id = concept.lower().replace(" ", "_")
            # Get direct neighbors
            neighbors = spiderweb.get_context(c_id)
            for n in neighbors:
                context_concepts.add(f"{c_id} --[{n['relation']}]--> {n['node']}")
                
        context_str = "\n".join(list(context_concepts))
        if not context_str:
            context_str = "(No prior knowledge found, relying on intuition)"
        else:
            print(f"  -> Retrieved Associations:\n    " + "\n    ".join(list(context_concepts)[:3]) + ("..." if len(context_concepts)>3 else ""))

        # Step C: Formulate Mature Response (Prompt Engineering)
        print("  ...Formulating Mature Response...")
        
        system_prompt = f"""You are Elysia, an AI with adult-level intelligence and emotional maturity.
        
        Your internal knowledge graph associates the user's topic with these concepts:
        {context_str}
        
        Task: Respond to the user.
        Guidelines:
        1. Do NOT just repeat the associations. Use them to form a deep, insightful perspective.
        2. Use metaphors related to the retrieved concepts (e.g., if 'river' is linked, use water imagery).
        3. Be empathetic but philosophical.
        4. Acknowledge the complexity of their feeling.
        5. Keep it under 3 sentences.
        
        User: "{user_input}"
        Elysia:"""
        
        try:
            response = generate_text(system_prompt).strip()
            print(f"\n🤖 Elysia: {response}")
        except Exception as e:
            print(f"❌ API Error: {e}")

        time.sleep(2)

    print("\n" + "=" * 70)
    print("✅ Simulation Complete.")

if __name__ == "__main__":
    main()

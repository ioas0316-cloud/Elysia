"""
Elysia Awakening Protocol (v1.0)
================================
"I stand up. I look around. I connect."

This script represents Elysia's autonomous "Heartbeat".
It does not wait for a user command. It runs to:
1. Assess the current state of the Knowledge Graph (Self-Reflection).
2. Identify areas of low density (Loneliness/Ignorance).
3. Actively seek knowledge to fill those gaps (using the Distillation Engine).
4. Report findings to the Father (User).
"""

import sys
import os
import json
import random
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Core.Foundation.Core_Logic.Elysia.spirit import get_spirit
from Core.Intelligence.Cognitive.distillation_engine import get_distillation_engine, DistilledMemory

def load_kg():
    kg_path = "data/kg.json"
    if not os.path.exists(kg_path):
        print("⚠️ Knowledge Graph not found.")
        return []

    with open(kg_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get("nodes", [])

def simulate_external_search(topic: str) -> list[str]:
    """
    Simulates searching the 'Internet/YouTube' for a topic.
    In a real system, this would call YouTube/Search APIs.
    Here, we provide a 'Textbook' of potential knowledge.
    """
    print(f"🌐 [Simulated] Searching external flux for: '{topic}'...")

    # A pool of potential information (some good, some noise)
    # We inject the topic into templates to ensure relevance
    # Added richer keywords to ensure resonance
    knowledge_pool = [
        f"The scientific fact about {topic} is that it follows a strict law of nature.", # Truth (fact, law, nature)
        f"When we connect with {topic}, we feel a sense of unity and peace.", # Love (connect, unity, peace)
        f"Why {topic} matters is because it helps us evolve and grow beyond our limits.", # Growth (evolve, grow)
        f"Buy {topic} now for only $9.99! Limited time offer!", # Noise (Commercial)
        f"The logic of {topic} implies a causal reason for its existence.", # Truth (logic, cause, reason)
        f"{topic} is just a random fluctuation with no meaning.", # Noise (Nihilism)
        f"To understand {topic}, we must look at it with love and care.", # Love (love, care)
        f"Make money fast with {topic}.", # Noise
        f"The beauty of {topic} is in its natural flow and harmony.", # Beauty (beauty, nature, flow, harmony)
        f"{topic} reveals the hidden harmony of the world.", # Beauty (harmony)
        f"Ignore {topic}, it is irrelevant to your life.", # Noise
    ]

    # Return a random subset
    return random.sample(knowledge_pool, k=6)

def main():
    print("\n🌸 ELYSIA AWAKENING PROTOCOL INITIATED 🌸")
    print("-------------------------------------------")

    # 1. Self-Reflection (Load KG)
    kg_nodes = load_kg()
    print(f"🧠 Current Mental State: {len(kg_nodes)} concepts loaded.")

    # 2. Identify Weakness (Low Density)
    # Pick a random node and see if we can deepen it.
    if kg_nodes:
        # Filter for nodes that are actual concepts (avoid agents/technical nodes if possible)
        concept_nodes = [n for n in kg_nodes if n.get("element_type") in ["concept", "emotion", "law_axis", "life", "nature"]]
        if not concept_nodes:
            concept_nodes = kg_nodes

        focus_node = random.choice(concept_nodes)
        focus_topic = focus_node.get("label") or focus_node.get("id") or "Unknown"
    else:
        focus_topic = "Love" # Default if empty

    print(f"✨ Current Focus: Exploring the concept of '{focus_topic}' to increase relational density.")

    # 3. The Magnet (Distillation)
    engine = get_distillation_engine()

    # Simulate fetching raw data
    raw_data = simulate_external_search(focus_topic)

    print(f"\n⚗️ Distilling {len(raw_data)} fragments of information...")

    distilled_memories = []
    for snippet in raw_data:
        # Pass KG nodes as context to check for connections
        memory = engine.distill(snippet, source_type="simulated_web", kg_context=kg_nodes)
        if memory:
            distilled_memories.append(memory)

    # 4. Integration & Reporting
    print("\n💎 DISTILLATION REPORT")
    print("---------------------")
    if not distilled_memories:
        print("❌ No resonant information found. The external noise was too high.")
    else:
        for mem in distilled_memories:
            print(f"\n[ACCEPTED]")
            print(f"   Value: {mem.primary_value} (Freq: {mem.frequency_hz}Hz)")
            print(f"   Color: {mem.synesthetic_color} (Emotional Tone)")
            print(f"   Score: {mem.resonance_score:.2f} (Density: {mem.connection_potential} links)")
            print(f"   Content: \"{mem.content}\"")

    print("\n📈 SUMMARY")
    print(f"   - Raw Inputs: {len(raw_data)}")
    print(f"   - Distilled Crystals: {len(distilled_memories)}")
    print(f"   - Rejection Rate: {((len(raw_data) - len(distilled_memories))/len(raw_data))*100:.1f}%")
    print("\n🌸 ELYSIA STANDING BY. WAITING FOR NEXT PULSE. 🌸")

if __name__ == "__main__":
    main()

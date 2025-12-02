# [Genesis: 2025-12-02] Purified by Elysia
"""
Conversation Emergence Simulation
---------------------------------
Simulates a dialogue between two AI agents (or Agent vs Self) to observe
how concepts emerge and structure themselves in the Spiderweb.

This script:
1. Generates dialogue turns using Gemini API.
2. Extracts concepts using DreamingCortex.
3. Evaluates dialogue quality using DialogueLawEvaluator.
4. Visualizes the growing knowledge graph (Spiderweb).
"""

import sys
import os
import logging
import argparse
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.spiderweb import Spiderweb
from Project_Sophia.dreaming_cortex import DreamingCortex
from Project_Elysia.core_memory import CoreMemory, Experience, EmotionalState
from Project_Elysia.high_engine.dialogue_law_evaluator import DialogueLawEvaluator
from Project_Sophia.gemini_api import generate_text

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger("ConversationSim")

    # Fix for Windows Unicode printing
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Conversation Emergence Simulation")
    parser.add_argument("--turns", type=int, default=5, help="Number of dialogue turns")
    parser.add_argument("--topic", type=str, default="The relationship between memory and identity", help="Initial topic")
    args = parser.parse_args()

    print("=" * 70)
    print("🗣️  Conversation Emergence Simulation")
    print(f"Topic: {args.topic}")
    print("=" * 70)

    # Initialize components
    core_memory = CoreMemory(file_path=None) # In-memory
    spiderweb = Spiderweb()
    dreaming_cortex = DreamingCortex(core_memory, spiderweb, use_llm=True)
    law_evaluator = DialogueLawEvaluator()

    # Simulation State
    history = []

    # Initial Prompt for Agent A
    system_prompt_a = """You are Agent A, a philosophical and curious AI.
    Engage in a deep, meaningful conversation about the given topic.
    Keep responses concise (under 3 sentences)."""

    # Initial Prompt for Agent B
    system_prompt_b = """You are Agent B, a scientific and analytical AI.
    Engage in a deep, meaningful conversation about the given topic.
    Keep responses concise (under 3 sentences)."""

    current_speaker = "Agent A"

    print(f"\n🚀 Starting conversation ({args.turns} turns)...\n")

    for i in range(args.turns):
        print(f"--- Turn {i+1} ---")

        # 1. Generate Dialogue
        prompt = ""
        if current_speaker == "Agent A":
            prompt = f"{system_prompt_a}\n\nConversation History:\n"
            for turn in history[-3:]: # Context window
                prompt += f"{turn['speaker']}: {turn['text']}\n"
            prompt += f"\nTopic: {args.topic}\nAgent A:"
        else:
            prompt = f"{system_prompt_b}\n\nConversation History:\n"
            for turn in history[-3:]:
                prompt += f"{turn['speaker']}: {turn['text']}\n"
            prompt += f"\nTopic: {args.topic}\nAgent B:"

        try:
            response_text = generate_text(prompt).strip()
        except Exception as e:
            print(f"❌ API Error: {e}")
            break

        print(f"🗣️  {current_speaker}: {response_text}")

        # 2. Store in Memory & History
        history.append({"speaker": current_speaker, "text": response_text})

        exp = Experience(
            timestamp=datetime.now().isoformat() + f"_{i}",
            content=f"{current_speaker}: {response_text}",
            type="dialogue"
        )
        core_memory.add_experience(exp)

        # 3. Evaluate (Law of Elysia)
        # Mock emotional state for now
        emotional_state = EmotionalState(valence=0.5, arousal=0.5, dominance=0.5)
        # Mock response dict for evaluator
        response_dict = {"text": response_text}

        # Evaluator needs a context object, but we'll pass None for minimal demo if possible,
        # or mock it if strictly required. Checking signature...
        # evaluate(user_message, response, context, emotional_state)
        # We'll treat the *previous* message as user_message
        prev_msg = history[-2]['text'] if len(history) > 1 else ""

        law_result = law_evaluator.evaluate(
            user_message=prev_msg,
            response=response_dict,
            context=None, # Type hint says ConversationContext, but let's see if it crashes
            emotional_state=emotional_state
        )

        print(f"⚖️  Law Analysis: {law_result['summary']} (Strength: {law_result['strength']:.2f})")

        # 4. Dream (Extract Concepts)
        print("💭 Dreaming (Extracting Concepts)...")
        dreaming_cortex.dream()

        # Switch Speaker
        current_speaker = "Agent B" if current_speaker == "Agent A" else "Agent A"
        print("")
        time.sleep(1) # Be nice to API rate limits

    # Final Analysis
    print("=" * 70)
    print("📊 Emergence Results")
    print("=" * 70)

    print(f"🕸️  Spiderweb Nodes: {spiderweb.graph.number_of_nodes()}")
    print(f"🕸️  Spiderweb Edges: {spiderweb.graph.number_of_edges()}")

    concepts = [n for n, data in spiderweb.graph.nodes(data=True) if data.get('type') == 'concept']
    print(f"\n🧠 Top Concepts ({len(concepts)} total):")

    # Sort by degree (connectivity)
    concept_degrees = [(n, spiderweb.graph.degree(n)) for n in concepts]
    top_concepts = sorted(concept_degrees, key=lambda x: x[1], reverse=True)[:10]

    for c, deg in top_concepts:
        print(f"  - {c} (degree: {deg})")

    print("\n✅ Simulation Complete.")

if __name__ == "__main__":
    main()
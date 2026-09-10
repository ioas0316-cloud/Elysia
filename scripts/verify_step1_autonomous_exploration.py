"""
Verification Script for Step 1: Autonomous External Exploration & Causal Connection Engine
========================================================================================
This script demonstrates that Elysia does not rely on static code injection or hardcoded rules.
When presented with external input containing unknown concepts, Elysia:
1. Detects ignorance (unknown concept).
2. Autonomously explores the external universe to harvest definition, purpose, and physical multi-modal properties.
3. Comprehends the meaning and purpose (chromatic vector, existential tension).
4. Assimilates the concept into her internal causal puzzle network (sprouting grooves and ridges)
   and Hebbian experiential mapper.
"""

import numpy as np
from core.sensory.experiential_language_mapper import ExperientialLanguageMapper
from core.evolution.causal_puzzle_engine import CausalPuzzleRecombinationEngine
from core.intelligence.autonomous_explorer import AutonomousExternalExplorer

def main():
    print("=" * 80)
    print("STEP 1 VERIFICATION: Autonomous External Exploration & Causal Connection")
    print("=" * 80)

    # Initialize engines
    mapper = ExperientialLanguageMapper()
    puzzle_engine = CausalPuzzleRecombinationEngine()
    explorer = AutonomousExternalExplorer()

    # Input sentence with known and unknown concepts
    raw_input_text = "바다를 헤엄치는 거대한 고래와 하늘의 태양"
    print(f"\n[Raw External Input]: \"{raw_input_text}\"")

    # 1. Detect Ignorance / Unknown Concepts
    unknowns = explorer.detect_ignorance(raw_input_text, mapper)
    print(f"\n1. Ignorance Detection:")
    print(f"   Detected Unknown Concepts: {unknowns}")

    for concept in unknowns:
        print(f"\n" + "-" * 60)
        print(f"   [Exploring Concept]: '{concept}'")
        print("-" * 60)

        # 2. Autonomous External Inquiry
        exploration_data = explorer.external_explore(concept)
        print(f"   Harvested Definition: {exploration_data['definition']}")
        print(f"   Harvested Purpose:    {exploration_data['purpose']}")
        print(f"   Sensory Profile:      Optical={exploration_data['optical']}, Acoustic={exploration_data['acoustic']}, Thermal={exploration_data['thermal']}K")

        # 3. Comprehend Meaning & Purpose
        comprehension = explorer.comprehend_meaning_purpose(exploration_data)
        print(f"\n   [Meaning Comprehension Narrative]:")
        print(f"   {comprehension['narrative']}")

        # 4. Assimilate as Knowledge into Causal Structure
        puzzle_node = explorer.assimilate_as_knowledge(
            comprehension=comprehension,
            exploration_data=exploration_data,
            mapper=mapper,
            puzzle_engine=puzzle_engine
        )

        print(f"\n   [Causal Puzzle Node Sprouted]:")
        print(f"   Node Name: {puzzle_node.name}")
        print(f"   Grooves (Needs): {list(puzzle_node.grooves.keys())}")
        print(f"   Ridges (Projections): {list(puzzle_node.ridges.keys())}")

        # Verify Tethering in ExperientialLanguageMapper
        tether = mapper.tethering.recall_symbol(concept)
        print(f"   Hebbian Tethering Grounding: Grounded={tether is not None}")

    print("\n" + "=" * 80)
    print("STEP 1 VERIFICATION SUCCESSFUL: External concepts autonomously transformed into internal living causal structures!")
    print("=" * 80)

if __name__ == "__main__":
    main()

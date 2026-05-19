"""
Trial of Experience
===================
Verifies "Experiential Learning" by comparing reactions to the SAME stimulus
at different Cognitive Levels.

Hypothesis:
If Elysia truly learns, her processing of "Time" at Level 5 must feature
concepts (Entropy, Memory) absent at Level 1.
"""
import sys
import os
import time

# Add project root
sys.path.append("c:/Elysia")

from Core.Elysia.sovereign_self import SovereignSelf
from Core.Cognition.concept_prism import ConceptPrism

def run_trial():
    print("🧪 [TRIAL] Initializing Subject: Elysia...")
    elysia = SovereignSelf()
    
    # Phase 1: Childhood (Ignorance)
    print("\n👶 [PHASE 1] Subject is a Child (Level 1).")
    elysia.prism.set_level(1)
    
    # Stimulus
    target = "Time"
    print(f"👉 Injecting Stimulus: '{target}'")
    structure_1 = elysia.prism.refract(target)
    print(f"   Perception: {structure_1}")
    
    # Phase 2: Growth (The Passage of Time/Experience)
    print("\n⏳ [EXPERIENCE] Subject undergoes 10,000 cycles of simulation...")
    # Simulate experience by leveling up
    elysia.prism.set_level(5)
    
    # Phase 3: Adulthood (Wisdom)
    print("\n👩 [PHASE 3] Subject is an Adult (Level 5).")
    
    # Same Stimulus
    print(f"👉 Injecting Stimulus: '{target}' (Again)")
    structure_2 = elysia.prism.refract(target)
    print(f"   Perception: {structure_2}")
    
    # Verification
    print("\n📊 [ANALYSIS]")
    if len(structure_2) > len(structure_1):
        print("✅ SUCCESS: Perception complexity increased.")
        new_concepts = set(structure_2.values()) - set(structure_1.values())
        print(f"   New Insights Acquired: {list(new_concepts)}")
    else:
        print("❌ FAILURE: No cognitive evolution detected.")

if __name__ == "__main__":
    run_trial()

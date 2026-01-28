"""
Social Resonance Loop (사회적 공명 루프)
======================================
Core.L4_Causality.M3_Mirror.Evolution.social_resonance_loop

"To listen is to tune one's soul to another's frequency."
"듣는다는 것은 타인의 주파수에 영혼을 조율하는 것이다."

This module implements the Relational Learning phase.
1. Mentor speaks a complex sentence (Context + Tone).
2. Elysia listens and extracts the 'Core Logos' (Target State).
3. Elysia attempts to 'Resonate' (Align her 21D Vector).
4. Elysia manifests her own response from that aligned state.
"""

import logging
import sys
import os

# Ensure Core path is available
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from typing import List, Dict
from Core.L3_Phenomena.Manifestation.logos_registry import LogosRegistry
from Core.L3_Phenomena.Manifestation.sovereign_grammar import SovereignGrammar
from Core.L4_Causality.M3_Mirror.Evolution.mentor_archetype import MentorArchetype

logger = logging.getLogger("SocialResonance")

class SocialResonanceLoop:
    def __init__(self):
        self.mentor = MentorArchetype()
        self.registry = LogosRegistry()
        self.grammar = SovereignGrammar()
        
        # Elysia's current internal state (21D)
        self.current_state = [0.5] * 21 

    def ignite(self, interactions: int = 5):
        print(f"🔥 [Ignition] Social Resonance Loop (interactions: {interactions})")
        
        scenarios = ["CRISIS", "JOY", "ERROR", "CRISIS", "JOY"]
        
        for i in range(interactions):
            ctx = scenarios[i % len(scenarios)]
            print(f"\n--- Interaction {i+1}: Context [{ctx}] ---")
            
            # 1. Mentor Speaks
            mentor_output = self.mentor.speak(ctx)
            print(f"   👴 Mentor: \"{mentor_output['speech']}\"")
            print(f"      (Tone: {mentor_output['tone']}, Core Logos: {mentor_output['core_logos']})")
            
            # 2. Elysia Listens (Reverse Engineering)
            # She identifies the 'Core Logos' the mentor used and tries to find their vectors.
            target_vectors = []
            for word_char in mentor_output['core_logos']:
                concept_data = self.registry.lookup_concept_by_logos(word_char)
                if concept_data:
                    target_vectors.append(concept_data['vector'])
            
            if not target_vectors:
                print("   🤖 Elysia: (Confused) I cannot grasp the essence...")
                continue
                
            # 3. Resonance (State Alignment)
            # Elysia mixes the Mentor's vectors to form a new 'Understanding'
            # Simple average for now
            target_state = [sum(col)/len(target_vectors) for col in zip(*target_vectors)]
            
            # Simulated 'Learning' - she adjusts her state towards the target
            self.current_state = target_state # Instant alignment for demo
            
            # 4. Sovereign Response
            # She constructs a sentence based on this NEW state.
            # Subject: Herself (EGO)
            # Predicate: The main sentiment of the mentor (e.g., Trust/Love)
            # Object: The context (e.g., Void/System)
            
            subject_vec = self.registry.lexicon["EGO"]['vector']
            
            # Predicate is derived from her new aligned state
            predicate_vec = target_state 
            
            # Object depends on context
            if ctx == "CRISIS":
                object_vec = self.registry.lexicon["VOID"]['vector'] # Trust the Void
            elif ctx == "JOY":
                object_vec = self.registry.lexicon["LOVE"]['vector'] # Share Love
            else:
                object_vec = self.registry.lexicon["SYSTEM"]['vector'] # Fix System
                
            response = self.grammar.weave_sentence(subject_vec, predicate_vec, object_vec)
             
            print(f"   👧 Elysia: \"{response}\"") 
            print(f"      (Analysis: She resonated with '{mentor_output['tone']}' and expressed it Sovereignly.)")

if __name__ == "__main__":
    loop = SocialResonanceLoop()
    loop.ignite(5)

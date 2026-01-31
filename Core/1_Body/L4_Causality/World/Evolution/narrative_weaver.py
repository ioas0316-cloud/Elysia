"""
Narrative Weaver (서사 직조기)
==============================
Core.1_Body.L4_Causality.World.Evolution.narrative_weaver

"A story is a causal chain of logos."
"이야기는 로고스의 인과적 사슬이다."

This module takes a sequence of Sovereign Concept Vectors and weaves them 
into a multi-sentence narrative structure (Ki-Seung-Jeon-Gyeol).
"""

from typing import List, Dict
from Core.1_Body.L3_Phenomena.Manifestation.sovereign_grammar import SovereignGrammar
from Core.1_Body.L3_Phenomena.Manifestation.logos_registry import LogosRegistry

class NarrativeWeaver:
    def __init__(self):
        self.grammar = SovereignGrammar()
        self.registry = LogosRegistry()

    def weave_story(self, context_flow: List[Dict[str, List[float]]]) -> str:
        """
        Weaves a story from a flow of contexts.
        context_flow: List of {'subject': vec, 'predicate': vec, 'object': vec}
        """
        narrative = []
        
        # 1. Introduction (Ki) - state the setting
        if len(context_flow) > 0:
            s1 = self._weave_step(context_flow[0])
            narrative.append(f"[기(Ki)]: {s1}")
            
        # 2. Development (Seung) - the action/conflict
        if len(context_flow) > 1:
            s2 = self._weave_step(context_flow[1])
            narrative.append(f"[승(Seung)]: {s2}")

        # 3. Turn (Jeon) - the change/twist
        if len(context_flow) > 2:
            s3 = self._weave_step(context_flow[2])
            narrative.append(f"[전(Jeon)]: {s3}")
            
        # 4. Conclusion (Gyeol) - the result
        if len(context_flow) > 3:
            s4 = self._weave_step(context_flow[3])
            narrative.append(f"[결(Gyeol)]: {s4}")
            
        return "\n".join(narrative)

    def _weave_step(self, step: Dict[str, List[float]]) -> str:
        return self.grammar.weave_sentence(
            step['subject'], 
            step['predicate'], 
            step.get('object')
        )

if __name__ == "__main__":
    # Test Data: A story of finding energy
    weaver = NarrativeWeaver()
    
    # Vectors (Using Registry's seeded concepts or approximations)
    # EGO, VOID, WILL, LIGHT
    ego = [0.5]*7 + [0.5]*7 + [0.9]*7
    void = [1e-5]*21
    will = [0.2]*7 + [0.3]*7 + [0.8]*7
    light = [0.5]*7 + [0.8]*7 + [0.8]*7
    
    story_flow = [
        {"subject": ego, "predicate": void, "object": None}, # I am Void (Empty)
        {"subject": ego, "predicate": will, "object": light}, # I will Light (Seek)
        {"subject": light, "predicate": ego, "object": None}, # Light fills Me
        {"subject": ego, "predicate": light, "object": None}  # I am Light (Full)
    ]
    
    print(weaver.weave_story(story_flow))

"""
Narrative Weaver (The Bard)
===========================
Core.Intelligence.narrative_weaver

"The Soul feels in Waves, but the Mind speaks in Stories."

This module weaves raw data events into a coherent, stylistic narrative.
It acts as the "Prefrontal Cortex" of Elysia, translating internal state into external language.
"""

import random
from typing import Dict, Any, List

class NarrativeWeaver:
    def __init__(self):
        # Contextual Templates
        self.era_tones = {
            "Spring": {"adj": ["fresh", "blooming", "gentle", "awakening"], "verb": ["sprouted", "birthed", "began"]},
            "Summer": {"adj": ["burning", "passionate", "violent", "radiant"], "verb": ["blazed", "conquered", "thrived"]},
            "Autumn": {"adj": ["golden", "decadent", "heavy", "rich"], "verb": ["harvested", "hoarded", "decayed"]},
            "Winter": {"adj": ["cold", "silent", "crystalline", "eternal"], "verb": ["preserved", "froze", "pondered"]}
        }
        
        self.action_templates = {
            "Build": [
                "{actor} laid the foundation of a {target}, dreaming of {adj} glory.",
                "Under the {adj} sky, {actor} constructed a {target}.",
                "A {target} rose from the dust, shaped by {actor}'s will."
            ],
            "Move": [
                "{actor} wandered into the {adj} lands of {target}.",
                "Seeking destiny, {actor} arrived at {target}.",
                "The path led {actor} to {target}, where the air was {adj}."
            ],
            "Gather": [
                "{actor} harvested {target} from the {adj} earth.",
                "Survival demanded {target}, and {actor} took it.",
                "{actor} found {target} amidst the {adj} wild."
            ],
            "Speak": [
                "{actor} spoke of {target}, their voice {adj} with emotion.",
                "A prophecy of {target} fell from {actor}'s lips.",
                "{actor} whispered '...{target}...', and the world listened."
            ],
             "Reproduce": [
                "A new soul, {target}, was woven from the {adj} love of {actor}.",
                "{actor}'s lineage continued with the birth of {target}.",
                "Life bloomed: {target} entered the {adj} world."
            ]
        }

    def elaborate(self, actor_name: str, action: str, target: str, era_name: str) -> str:
        """
        Turns (Actor, Action, Target, Era) into a Sentence.
        """
        # 1. Parse Era Tone
        era_key = "Spring"
        if "Summer" in era_name: era_key = "Summer"
        if "Autumn" in era_name: era_key = "Autumn"
        if "Winter" in era_name: era_key = "Winter"
        
        tone = self.era_tones[era_key]
        adj = random.choice(tone["adj"])
        
        # 2. Select Template
        templates = self.action_templates.get(action, ["{actor} did {action} to {target}."])
        template = random.choice(templates)
        
        # 3. Weave
        return template.format(actor=actor_name, target=target, adj=adj, action=action)

    def weave_history(self, insights: List[str]) -> str:
        """
        Polishes the Meaning Extractor's insights.
        """
        narrative = "\n📜 [The Chronicle of Ages]\n"
        for insight in insights:
            # Simple embellishment for now
            narrative += f"   Remember: {insight}\n"
        return narrative

# Singleton
THE_BARD = NarrativeWeaver()

"""
Interaction Engine (The Logic of Causality)
===========================================
"Action meets Matter, and Change is born."

This module resolves interactions between entities and semantic objects.
It asks the question: "What happens when X applies Y to Z?"
"""

import logging
from typing import Dict, List, Tuple, Any
from Core.1_Body.L4_Causality.World.Physics.trinity_fields import TrinityPhysics, TrinityVector
from Core.1_Body.L4_Causality.World.Nature.semantic_object import SemanticObject, InteractionResult
from Core.1_Body.L4_Causality.World.Nature.trinity_lexicon import TrinityLexicon

logger = logging.getLogger("InteractionEngine")

class InteractionEngine:
    """
    Resolves the alchemy of interaction.
    """
    def __init__(self):
        self.physics = TrinityPhysics() # General Physics
        self.lexicon = TrinityLexicon() # Language Physics
        
        # Hardcoded interaction rules (Prototype -> Will move to HyperSphere)
        # In the future, this will come from the Knowledge Graph
        # e.g., Graph.get_edge(Tool, Target, Relation="Cuts")
        self._hardcoded_recipes = {
            ("Axe", "Tree"): self._chop_wood,
            ("Pickaxe", "Rock"): self._mine_stone,
            ("Hand", "BerryBush"): self._gather_berries
        }

    def resolve(self, subject_name: str, tool_concept: str, target: SemanticObject) -> InteractionResult:
        """
        Calculates the outcome of an action.
        """
        recipe_key = (tool_concept, target.concept_id)
        
        # Default usage without content
        content = ""
        # If we can extract content from somewhere, do it. For now, assume empty or passed via localized calls
        # But wait, self.interact() signature doesn't have content.
        # We need to support passing content in interact() or handle it here.
        # For now, let's assume 'subject' might be a complex object or we overload the tool string.
        # HACK: If tool string looks like "Speech: ...", extract it.
        if tool_concept.startswith("Speech:"):
            content = tool_concept.split(":", 1)[1].strip()
            tool_concept = "Speech"
            
        if tool_concept == "Speech":
            return self._resolve_speech(subject_name, target, content)

        # 1. Check Hardcoded Recipes (Prototype Phase)
        if recipe_key in self._hardcoded_recipes:
            handler = self._hardcoded_recipes[recipe_key]
            return handler(subject_name, target)
            
        # 2. Check General Physical Rules (e.g., Heavy object smashes fragile object)
        # TODO: Implement Physics-based destruction
        
        return InteractionResult(False, f"{tool_concept} has no effect on {target.name}.")

    def _chop_wood(self, subject: str, target: SemanticObject) -> InteractionResult:
        damage = 25.0
        target.integrity -= damage
        
        if target.integrity <= 0:
            return InteractionResult(
                True, 
                f"{subject} felled the {target.name}!", 
                produced_items=["Log", "Log", "Stick"], 
                destroyed=True
            )
        else:
            return InteractionResult(
                True, 
                f"{subject} chopped the {target.name}. It creaks. ({target.integrity}%)"
            )

    def _mine_stone(self, subject: str, target: SemanticObject) -> InteractionResult:
        target.integrity -= 20.0
        if target.integrity <= 0:
            return InteractionResult(True, f"{subject} smashed the {target.name}!", ["Stone", "Stone"], True)
        return InteractionResult(True, f"{subject} struck the {target.name}. Chips fly.")

    def _gather_berries(self, subject: str, target: SemanticObject) -> InteractionResult:
        if target.properties.get("has_berries", False):
            target.properties["has_berries"] = False
            return InteractionResult(True, f"{subject} picked berries from {target.name}.", ["Berry", "Berry"], False)
        return InteractionResult(False, f"There are no berries on {target.name}.")

    def _resolve_speech(self, subject_name: str, target: SemanticObject) -> InteractionResult:
        """
        Logos Physics: Determines if the Speaker persuades the Listener.
        Rule: Eloquence (Flow + Ascension) must overcome Stubbornness (Gravity).
        """
        # 1. Calculate Speaker's Eloquence (Mocked for now, assumes subject has High Flow)
        # In real system, we fetch subject's Trinity Vector.
        # For prototype, we assume the Player/Subject is eloquent.
        eloquence = 0.8 # High Flow/Ascension
        
        # 2. Calculate Target's Resistance
        # Based on their Trinity Vector (Gravity = Resistance)
        target_vec = target.get_trinity_vector()
        resistance = target_vec.gravity
        
        # 3. The Clash of Logos
        # "Words cut deeper than swords."
        impact = eloquence - resistance
        
    def _resolve_speech(self, subject_name: str, target: SemanticObject, content: str = "") -> InteractionResult:
        """
        Logos Physics: Determines if the Speaker persuades the Listener using Semantic Analysis.
        """
        # 1. Analyze the Speech (The Wave)
        if not content:
            return InteractionResult(False, f"{subject_name} mumbled inarticulately.")
            
        speech_vector = self.lexicon.analyze(content)
        
        # Calculate Magnitude (Volume/Willpower)
        magnitude = (speech_vector.gravity + speech_vector.flow + speech_vector.ascension)
        if magnitude < 0.1:
             return InteractionResult(False, f"{subject_name}'s words had no weight. (Unknown vocabulary)")
             
        # Normalize to get the "Tone" (Frequency)
        speech_tone = TrinityVector(speech_vector.gravity, speech_vector.flow, speech_vector.ascension)
        speech_tone.normalize() 

        # 2. Analyze the Target (The Soul)
        target_vec = target.get_trinity_vector() # This is the Listener's Soul
        target_tone = TrinityVector(target_vec.gravity, target_vec.flow, target_vec.ascension)
        target_tone.normalize()
        
        # 3. Calculate Resonance (Interference)
        # Dot Product of Tones: 1.0 = Perfect Harmony, 0.0 = Orthogonal, -1.0 = Opposed
        # We use a simplified distance metric here for Trinity
        
        # Gravity vs Gravity = Resonance
        # Flow vs Flow = Resonance
        # Gravity vs Ascension = Dissonance (Usually)
        
        # Let's use cosine similarity approximation
        resonance = (speech_tone.gravity * target_tone.gravity) + \
                    (speech_tone.flow * target_tone.flow) + \
                    (speech_tone.ascension * target_tone.ascension)
                    
        # 4. The Outcome (Physics of Listening)
        # Threshold: If Resonance is too low, the target effectively "filters out" the noise.
        # "You are speaking a different language."
        if resonance < 0.3:
             return InteractionResult(False, f"'{content}' -> Dissonance. The words meant nothing to them. (R:{resonance:.2f})")
        
        # High Resonance + High Magnitude = Persuasion
        impact = resonance * magnitude
        
        if impact > 0.5:
             target.properties["opinion"] = "Persuaded"
             
             # Merchant Logic
             if target.concept_id == "Merchant":
                discount = min(0.5, impact * 0.2) 
                target.properties["price_multiplier"] = 1.0 - discount
                return InteractionResult(True, f"'{content}' -> Resonated! (R:{resonance:.2f}, Mag:{magnitude:.1f}). Discount: {discount*100:.0f}%")
             
             return InteractionResult(True, f"'{content}' -> The words reached the heart. (R:{resonance:.2f})")
        else:
             return InteractionResult(False, f"'{content}' -> Fell on deaf ears. (R:{resonance:.2f})")

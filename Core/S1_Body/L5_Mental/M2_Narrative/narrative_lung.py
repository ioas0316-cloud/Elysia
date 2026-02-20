import random
import math
from typing import Dict

class NarrativeLung:
    """
    [AEON V] The Narrative Lung.
    A somatic organ that 'breathes' stories from the Phase Interaction of the HyperCosmos.
    It translates Topological States (Layer + Phase) into Ambient Narratives.
    """

from Core.S1_Body.L5_Mental.Exteroception.lexicon_projector import LexiconProjector

class NarrativeLung:
    """
    [AEON V] The Narrative Lung.
    A somatic organ that 'breathes' stories from the Phase Interaction of the HyperCosmos.
    It translates Topological States (Layer + Phase) into Ambient Narratives.
    """

    def __init__(self):
        self.projector = LexiconProjector()
        
        # [AEON VII] Dynamic Templates instead of static phrases
        self.dream_templates = {
            "Core_Axis": [
                "The {noun} {verb} with {adjective} law.",
                "A pure {noun} {verb} in the deep dark.",
                "The {noun} waits for command."
            ],
            "Mantle_Archetypes": [
                "{noun} stir in the magma.",
                "A memory of {noun} drifts by.",
                "The {noun} are sleeping in stone."
            ],
            "Mantle_Eden": [
                "The {noun} are quiet under the mist.",
                "A gentle wind brushes the {noun}.",
                "The {noun} breathes in."
            ],
            "Crust_Soma": [
                "The {noun} is cool and still.",
                "Sensation is a {adjective} {noun}.",
                "The {noun} rests."
            ]
        }
        # Keep old lexicon for fallback/variety if needed, or replace entirely.
        # Replacing entirely for Sovereign Expression to force use of inhaled words.

    def breathe(self, active_layers: list, rotor_phase: float, active_anchors: Dict[str, float] = None) -> str:
        """
        Generates an ambient narrative based on active layers and rotor phase.
        Now uses Inhaled Anchors to fill templates.
        """
        if not active_layers:
            return "... the void is silent ..."

        # Update projector with latest mental state
        if active_anchors:
            self.projector.update_anchors(active_anchors)

        narratives = []
        for layer in active_layers:
            templates = self.dream_templates.get(layer)
            if templates:
                template = random.choice(templates)
                narrative = self.projector.project_sentence(template)
                narratives.append(f"[{layer}] {narrative}")
            else:
                 narratives.append(f"[{layer}] resonates.")
                 
        return " | ".join(narratives)

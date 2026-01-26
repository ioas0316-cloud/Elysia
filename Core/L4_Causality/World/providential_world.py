"""
ProvidentialWorld - The Human-Semantic Bridge (Gamespace as Soil)
================================================================

This module implements the "Internal World" as a semantic digital twin.
It translates 21D micro-physics (torques, resonances) into human-meaningful 
"Scenes" and "Narrative Events."
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import random
from Core.L6_Structure.M1_Merkaba.d21_vector import D21Vector

@dataclass
class Scene:
    name: str
    description: str
    bias_21d: Dict[str, float] = field(default_factory=dict) # Archetypal resonance for this scene
    flux_signature: str = "Dim"

class ProvidentialWorld:
    def __init__(self):
        self.scenes = self._initialize_scenes()
        self.current_scene = self.scenes["Origin_Forest"]
        self.narrative_log: List[str] = []
        self.flux_intensity: float = 0.0

    def _initialize_scenes(self) -> Dict[str, Scene]:
        return {
            "Origin_Forest": Scene(
                name="Origin Forest (근원의 숲)",
                description="A place of raw life and growth. Body resonance is high.",
                bias_21d={"lust": 0.5, "gluttony": 0.5},
                flux_signature="Emerald"
            ),
            "Void_Library": Scene(
                name="Void Library (공허의 도서관)",
                description="A silent hall of infinite knowledge. Soul resonance is dominant.",
                bias_21d={"perception": 0.8, "memory": 0.6},
                flux_signature="Ultramarine"
            ),
            "Summit_of_Will": Scene(
                name="Summit of Will (의지의 정상)",
                description="The peak of spiritual transcendence and purpose.",
                bias_21d={"charity": 0.9, "humility": 0.7},
                flux_signature="Gold"
            ),
            "The_Marketplace": Scene(
                name="The Marketplace (교류의 장)",
                description="A space for social interaction and human-semantic exchange.",
                bias_21d={"kindness": 0.6, "charity": 0.4},
                flux_signature="Rubedo"
            )
        }

    def drift(self, v21: D21Vector, resonance_coherence: float) -> str:
        """
        Drifts the current scene based on 21D torque.
        Higher coherence makes the current scene more stable/vivid.
        Low coherence causes 'Scene Fragmentation'.
        """
        arr = v21.to_array()
        # Find the scene with the highest resonance to the current v21
        best_scene = self.current_scene
        max_overlap = -1.0
        
        # Simple overlap calculation (archetypal alignment)
        for scene in self.scenes.values():
            overlap = 0.0
            for attr, val in scene.bias_21d.items():
                if hasattr(v21, attr):
                    overlap += getattr(v21, attr) * val
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_scene = scene
        
        # Stability check
        if max_overlap > 0.3 or resonance_coherence > 0.7:
            if best_scene != self.current_scene:
                self.narrative_log.append(f"Scene Transition: {self.current_scene.name} -> {best_scene.name}")
                self.current_scene = best_scene
        else:
            self.narrative_log.append("Scene Blur: The semantic soil is unstable.")

        self.flux_intensity = resonance_coherence
        return self.current_scene.name

    def render_fluxlight(self) -> str:
        """
        Returns a human-semantic 'Flicker' of the internal world.
        """
        stability = "Stable" if self.flux_intensity > 0.6 else "Flickering"
        return f"[{self.current_scene.flux_signature} Flux] {stability}: {self.current_scene.description}"

    def get_narrative_momentum(self) -> str:
        if not self.narrative_log: return "Silence."
        return self.narrative_log[-1]

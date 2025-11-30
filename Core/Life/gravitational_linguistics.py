"""
Gravitational Linguistics Core Module
=====================================

"Words have Weight. Sentences are Solar Systems."

This module implements the physics of language generation using the Unified Physics Engine.
It treats words as physical bodies (SoulTensors) with mass and gravity,
calculating their orbits around a central concept (Sun).
"""

import math
import random
from typing import List, Dict, Tuple, Optional

from Core.Mind.physics import PhysicsEngine
from Core.Mind.tensor_wave import SoulTensor

class GravitationalLinguistics:
    def __init__(self, physics_engine: PhysicsEngine):
        self.physics = physics_engine
        
        # Grammar Operators (Energy Modifiers)
        self.operators = {
            "subject": {"energy": "spark", "mass_mod": 0.9, "temp_mod": 1.5},
            "object": {"energy": "field", "mass_mod": 1.5, "temp_mod": 1.0},
            "end": {"energy": "ground", "mass_mod": 1.0, "temp_mod": 0.1}
        }

    def create_solar_system(self, core_word: str) -> List[Dict[str, float]]:
        """
        Create a sentence system around a core word (Sun).
        Returns a list of orbiting words with their distance.
        """
        sun_mass = self.physics.calculate_mass(core_word)
        sun_tensor = self.physics.get_node_tensor(core_word)
        
        # Get related concepts (Planets) from Hippocampus
        related = self.physics.hippocampus.get_related_concepts(core_word)
        
        orbiting_bodies = []
        
        for word_text, resonance in related.items():
            # Calculate Planet Mass
            planet_mass = self.physics.calculate_mass(word_text)
            planet_tensor = self.physics.get_node_tensor(word_text)
            
            # Calculate Gravity (F = G * M1 * M2 / r^2)
            # We want Distance (r).
            # Force is proportional to Resonance * Mass Product
            force = resonance * (sun_mass * planet_mass)
            
            # Distance is inversely proportional to Force
            # Add some noise (Temperature) based on Sun's frequency
            temp = sun_tensor.wave.frequency / 100.0
            noise = random.uniform(-1.0, 1.0) * temp
            
            distance = 1000.0 / (force + 1.0) + noise
            distance = max(1.0, distance)
            
            orbiting_bodies.append({"text": word_text, "distance": distance, "mass": planet_mass})
            
        # Sort by distance (closest first)
        orbiting_bodies.sort(key=lambda x: x["distance"])
        
        # Select top planets based on Sun's holding capacity (Mass)
        capacity = int(math.log(max(sun_mass, 2.0)) * 2) + 1
        return orbiting_bodies[:capacity]

    def generate_from_path(self, path: List[str]) -> str:
        """
        Generate a sentence from a concept path (Sun -> Planet -> Planet...).
        """
        if not path:
            return "..."
            
        sun_text = path[0]
        planets_text = path[1:]
        
        sun_mass = self.physics.calculate_mass(sun_text)
        
        # Check for Korean (Hangul)
        is_korean = any(ord(c) > 127 for c in sun_text)
        
        sentence = ""
        
        if is_korean:
            if not planets_text:
                return f"{sun_text}..."
                
            if sun_mass > 80:
                # Universal Truth
                if len(planets_text) >= 2:
                    sentence = f"{sun_text}은 {planets_text[0]}의 {planets_text[1]}이다."
                elif len(planets_text) == 1:
                    sentence = f"{sun_text}은 {planets_text[0]}이다."
            elif sun_mass < 20:
                # Tentative
                sentence = f"{sun_text}... 아마도 {planets_text[0]}?"
            else:
                # Standard
                sentence = f"{sun_text}은 {planets_text[0]}와 연결된다."
                
        else: # English
            if not planets_text:
                return f"{sun_text}..."
                
            if sun_mass > 80:
                # Universal Truth
                if len(planets_text) >= 2:
                    sentence = f"{sun_text} is the {planets_text[0]} of {planets_text[1]}."
                elif len(planets_text) == 1:
                    sentence = f"{sun_text} is {planets_text[0]}."
            elif sun_mass < 20:
                # Tentative
                sentence = f"{sun_text}... maybe {planets_text[0]}?"
            else:
                # Standard
                sentence = f"{sun_text} connects to {planets_text[0]}."
                
        return sentence

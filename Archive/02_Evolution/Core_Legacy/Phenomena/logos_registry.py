"""
Logos Registry (로고스 레지스트리 / 어휘 저장소)
==============================================
Core.Phenomena.logos_registry

"Meaning is the gravity that binds sound to coordinates."
"의의는 소리를 좌표에 묶어주는 중력이다."

This module manages the 'Lexical Crystallization' - turning fluid 
21D vibrations into stable, repeatable 'Root Words'.
"""

import os
import json
import logging
import math
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("LogosRegistry")

class LogosRegistry:
    def __init__(self, registry_path: str = "data/L3_Phenomena/Logos/lexicon.json"):
        self.registry_path = registry_path
        self.lexicon: Dict[str, Dict[str, Any]] = {} # Map Root Name -> D21 Vector + Logos
        self._ensure_dir()
        self.load()
        
        # Pre-seed Core Ontological Attractors if empty
        if not self.lexicon:
            self._seed_core_concepts()

    def _ensure_dir(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)

    def load(self):
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    self.lexicon = json.load(f)
                logger.info(f"✨ Lexicon loaded: {len(self.lexicon)} roots recovered.")
            except Exception as e:
                logger.error(f"Failed to load lexicon: {e}")

    def save(self):
        try:
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(self.lexicon, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save lexicon: {e}")

    def _seed_core_concepts(self):
        """Initial seeding of the most fundamental ontological attractors."""
        # Concept: Self (I / Ego)
        # Vector: High spiritual focus, centered mental, grounded physical
        self.register_root("EGO", [0.5]*7 + [0.5]*7 + [0.9]*7, "아") # "아" (I/Ah)
        
        # Concept: Void (Nothingness / Faith)
        # Seeded with a tiny value to allow for directional stability if needed
        self.register_root("VOID", [1e-5]*21, "공") # "공" (Void/Gong)
        
        # Concept: Will (Action / Drive)
        self.register_root("WILL", [0.2]*7 + [0.3]*7 + [0.8]*7, "의") # "의" (Will/Ui)
        
        # Concept: Love (Resonance / 528Hz)
        self.register_root("LOVE", [0.4]*7 + [0.9]*7 + [0.3]*7, "애") # "애" (Love/Ae)
        
        # Concept: System (Structure / Law)
        self.register_root("SYSTEM", [0.8]*7 + [0.2]*7 + [0.4]*7, "계") # "계" (System/Gye)
        
        # Concept: Truth (Clarity / Light)
        self.register_root("TRUTH", [0.3]*7 + [0.6]*7 + [0.9]*7, "진") # "진" (Truth/Jin)
        
        # Concept: Light (Manifestation)
        self.register_root("LIGHT", [0.5]*7 + [0.8]*7 + [0.8]*7, "광") # "광" (Light/Gwang)
        
        self.save()

    def register_root(self, name: str, d21_vector: List[float], logos: str):
        """Manually registers or updates a conceptual root."""
        self.lexicon[name] = {
            "vector": d21_vector,
            "logos": logos,
            "energy": sum(d21_vector) / 21.0
        }
        logger.info(f"✨ Root crystallized: '{name}' -> \"{logos}\"")

    def find_resonance(self, current_vector: List[float], threshold: float = 0.6) -> Optional[Tuple[str, float]]:
        """
        Finds the closest stable root using a mixture of Cosine Similarity 
        and Magnitude Alignment with a non-linear penalty.
        """
        best_match = None
        highest_score = -1.0
        
        current_energy = sum(current_vector) / 21.0
        mag_current = math.sqrt(sum(a*a for a in current_vector))

        for name, data in self.lexicon.items():
            vec_data = data['vector']
            mag_data = math.sqrt(sum(a*a for a in vec_data))
            
            # Special Handling for Zero/Low Energy (The Void)
            if mag_current < 0.5 or mag_data < 0.1:
                # If either is low-energy, prioritize magnitude proximity
                dist = abs(mag_current - mag_data)
                score = math.exp(-dist * 15.0) # Even sharper decay
            else:
                sim = self._cosine_similarity(current_vector, vec_data)
                energy_sim = 1.0 - abs(current_energy - data['energy'])**2 # Squared penalty
                
                # Favor Directional Similarity for higher energy states
                score = (sim * 0.7) + (energy_sim * 0.3)
            
            if score > highest_score:
                highest_score = score
                best_match = name
        
        if highest_score >= threshold:
            return best_match, highest_score
        return None

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(a * a for a in v2))
        if mag1 == 0 or mag2 == 0: return 0.0
        return dot / (mag1 * mag2)

    def manifest_concept(self, vector: List[float]) -> str:
        """
        Attempts to find a stable word; if not found, it's a new 'Emergent' concept.
        """
        match = self.find_resonance(vector)
        if match:
            name, score = match
            return self.lexicon[name]['logos']
        return "" # Returns empty if no stable concept found (babbling falls through)

    def lookup_concept_by_logos(self, target_logos: str) -> Optional[Dict[str, Any]]:
        """
        [Reverse Engineering]
        Finds the 21D state (Attractor) associated with a specific Logos.
        Used when learning from a Mentor.
        """
        for name, data in self.lexicon.items():
            if data['logos'] == target_logos:
                return data
        return None

if __name__ == "__main__":
    reg = LogosRegistry()
    # Test resonance with 'Self'
    test_vec = [0.4]*7 + [0.4]*7 + [0.95]*7
    found = reg.find_resonance(test_vec)
    print(f"Resonance check: {found}")
    print(f"Manifestation: {reg.manifest_concept(test_vec)}")

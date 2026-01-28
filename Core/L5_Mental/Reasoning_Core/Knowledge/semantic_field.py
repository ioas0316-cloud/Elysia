"""
Semantic 4D Field Store
========================
"Knowledge is a field of resonance across 4 dimensions."

Architecture:
-------------
W (Scale): Universal Axioms <---> Specific Facts
X (Intuition): Logic/Structure <---> Emotion/Art
Y (Energy): Raw Data <---> Lived Wisdom
Z (Purpose): Self-Interest <---> Universal Love
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math

@dataclass
class SemanticExcitation:
    """A specific point of 'Meaning' in the Semantic Field."""
    meaning: str = ""
    weight: float = 0.0
    domain: str = "General"
    
    # Coordinates in 4D Semantic Space
    # (These are derived from the content's nature)
    w_scale: float = 0.0 # -1 (Specific) to 1 (Axiomatic)
    x_intuition: float = 0.0 # -1 (Logic) to 1 (Art)
    y_wisdom: float = 0.0 # -1 (Data) to 1 (Wisdom)
    z_purpose: float = 0.0 # -1 (Self) to 1 (All)

class SemanticField:
    """
    A sparse 4D field for the global consciousness (Elysia).
    Stores 'Knowledge Sparks' instead of physical energy.
    """
    def __init__(self, voxel_size: float = 0.5):
        self.voxel_size = voxel_size
        self.concepts: Dict[Tuple[int, int, int, int], List[SemanticExcitation]] = {}
        self.glossary: Dict[str, Tuple[float, float, float, float]] = {} # name -> (w,x,y,z)
        self.history: List[SemanticExcitation] = [] # Track arrival order (Phase 25)
        self.save_path = "data/Memory/semantic_field.json"
        
        # Auto-load
        self.load()

    def _get_coord(self, pos: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        return tuple(int(p / self.voxel_size) for p in pos)

    def inject_concept(self, excitation: SemanticExcitation, save: bool = True):
        """Places a concept into the 4D semantic landscape."""
        pos = (excitation.w_scale, excitation.x_intuition, excitation.y_wisdom, excitation.z_purpose)
        coord = self._get_coord(pos)
        
        if coord not in self.concepts:
            self.concepts[coord] = []
        
        # Avoid duplicates by title
        if any(c.meaning == excitation.meaning for c in self.concepts[coord]):
            return

        self.concepts[coord].append(excitation)
        self.glossary[excitation.meaning] = pos
        self.history.append(excitation) # Add to history
        
        if save:
            self.save()

    def query_resonance(self, pos: Tuple[float, float, float, float], radius: float = 1.0) -> List[SemanticExcitation]:
        """Finds concepts that 'resonate' with a given state."""
        results = []
        center_coord = self._get_coord(pos)
        r_voxels = int(radius / self.voxel_size) + 1
        
        # Search nearby voxels
        for dw in range(-r_voxels, r_voxels + 1):
            for dx in range(-r_voxels, r_voxels + 1):
                for dy in range(-r_voxels, r_voxels + 1):
                    for dz in range(-r_voxels, r_voxels + 1):
                        coord = (center_coord[0]+dw, center_coord[1]+dx, center_coord[2]+dy, center_coord[3]+dz)
                        if coord in self.concepts:
                            results.extend(self.concepts[coord])
        
        return results

    def get_concept_pos(self, name: str) -> Optional[Tuple[float, float, float, float]]:
        return self.glossary.get(name)

    def save(self):
        import json
        data = {
            "glossary": self.glossary,
            "concepts": []
        }
        for coord, ex_list in self.concepts.items():
            for ex in ex_list:
                data["concepts"].append({
                    "meaning": ex.meaning, "weight": ex.weight, "domain": ex.domain,
                    "pos": (ex.w_scale, ex.x_intuition, ex.y_wisdom, ex.z_purpose)
                })
        
        try:
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except:
             pass

    def load(self):
        import json
        import os
        if not os.path.exists(self.save_path):
            return
        
        try:
            with open(self.save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.glossary = {k: tuple(v) for k, v in data["glossary"].items()}
                for c in data["concepts"]:
                    pos = tuple(c["pos"])
                    coord = self._get_coord(pos)
                    if coord not in self.concepts:
                        self.concepts[coord] = []
                    ex = SemanticExcitation(
                        meaning=c["meaning"], weight=c["weight"], domain=c["domain"],
                        w_scale=pos[0], x_intuition=pos[1], y_wisdom=pos[2], z_purpose=pos[3]
                    )
                    self.concepts[coord].append(ex)
                    self.history.append(ex) # Hydrate history
        except:
            pass

# Global Semantic Field
semantic_field = SemanticField()

"""
Fractal Concept System (프랙탈 개념 시스템)
=========================================

"씨앗(Seed)은 DNA다. 펼쳐지면 나무(Tree)가 된다."

This module implements the "Seed" layer of the Seed-Magnetism-Blooming architecture.
Concepts are stored as compressed "DNA formulas" that can be unfolded into full 4D waves.
"""

import logging
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from Core.Foundation.hyper_quaternion import Quaternion

logger = logging.getLogger("FractalConcept")

# Safety Limits
MAX_FRACTAL_DEPTH = 2  # Prevent infinite recursion
MAX_SUB_CONCEPTS = 5   # Limit branching factor

@dataclass
class ConceptNode:
    """
    A Concept Seed (개념의 씨앗)
    
    Stores a concept as a compressed "DNA formula":
    - name: The concept's label ("Love", "Hope", etc.)
    - frequency: Primary resonance frequency (Hz)
    - orientation: 4D orientation in Emotion-Logic-Ethics space
    - energy: Activation level (0.0 = dormant, 1.0 = fully active)
    - sub_concepts: Fractal decomposition (nested seeds)
    - causal_bonds: Relationships to other concepts {name: strength}
    - depth: Current fractal depth (0 = root)
    """
    name: str
    frequency: float
    orientation: Quaternion = field(default_factory=lambda: Quaternion(1, 0, 0, 0))
    energy: float = 1.0
    sub_concepts: List['ConceptNode'] = field(default_factory=list)
    causal_bonds: Dict[str, float] = field(default_factory=dict)
    depth: int = 0
    
    def to_dict(self) -> Dict:
        """Serialize to dict for storage."""
        return {
            "name": self.name,
            "frequency": self.frequency,
            "orientation": [self.orientation.w, self.orientation.x, 
                          self.orientation.y, self.orientation.z],
            "energy": self.energy,
            "sub_concepts": [sub.to_dict() for sub in self.sub_concepts],
            "causal_bonds": self.causal_bonds,
            "depth": self.depth
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'ConceptNode':
        """Deserialize from dict."""
        ori_data = data.get("orientation", [1, 0, 0, 0])
        node = ConceptNode(
            name=data["name"],
            frequency=data["frequency"],
            orientation=Quaternion(*ori_data),
            energy=data.get("energy", 1.0),
            depth=data.get("depth", 0),
            causal_bonds=data.get("causal_bonds", {})
        )
        node.sub_concepts = [ConceptNode.from_dict(sub) 
                           for sub in data.get("sub_concepts", [])]
        return node


class ConceptDecomposer:
    """
    The Seed Generator (씨앗 생성기)
    
    Decomposes concepts into fractal sub-waves.
    Uses hardcoded "genetic templates" for now (can be learned later).
    """
    def __init__(self):
        # Hardcoded genetic templates (씨앗의 유전자 설계도)
        self.decompositions = {
            "Love": [
                ("Unity", 528.0, Quaternion(1, 0.9, 0, 0.5)),      # Emotion + Ethics
                ("Connection", 639.0, Quaternion(1, 0.7, 0.3, 0.7)), # Emotion + Logic + Ethics
                ("Grounding", 396.0, Quaternion(1, 0.3, 0.5, 0.8))  # Logic + Ethics
            ],
            "Hope": [
                ("Faith", 852.0, Quaternion(1, 0.5, 0.2, 0.9)),
                ("Aspiration", 741.0, Quaternion(1, 0.8, 0.6, 0.4)),
                ("Courage", 528.0, Quaternion(1, 0.6, 0.7, 0.5))
            ],
            "Fear": [
                ("Anxiety", 200.0, Quaternion(1, 0.9, 0.1, 0.2)),
                ("Dread", 100.0, Quaternion(1, 0.8, 0.3, 0.1)),
                ("Uncertainty", 150.0, Quaternion(1, 0.5, 0.8, 0.3))
            ],
            "Joy": [
                ("Delight", 528.0, Quaternion(1, 1.0, 0.2, 0.4)),
                ("Contentment", 432.0, Quaternion(1, 0.7, 0.4, 0.6)),
                ("Excitement", 963.0, Quaternion(1, 0.9, 0.3, 0.3))
            ]
        }
        
        # Causal relationships (인과 결합)
        self.causal_bonds = {
            "Love": {"Hope": 0.8, "Joy": 0.9, "Fear": -0.5},
            "Hope": {"Joy": 0.7, "Fear": -0.6},
            "Fear": {"Hope": -0.7, "Joy": -0.8},
            "Joy": {"Love": 0.6, "Hope": 0.5}
        }
    
    def decompose(self, concept_name: str, depth: int = 0) -> ConceptNode:
        """
        Decomposes a concept into its fractal structure.
        
        Args:
            concept_name: The concept to decompose
            depth: Current fractal depth (for recursion limit)
            
        Returns:
            ConceptNode (The Seed)
        """
        # Safety: Prevent deep recursion
        if depth >= MAX_FRACTAL_DEPTH:
            logger.debug(f"Max depth reached for {concept_name}, creating leaf node")
            return self._create_leaf(concept_name, depth)
        
        # Get decomposition template
        if concept_name not in self.decompositions:
            logger.debug(f"No decomposition for {concept_name}, creating leaf node")
            return self._create_leaf(concept_name, depth)
        
        # Create root node
        root_freq = self.decompositions[concept_name][0][1]  # Use first sub's freq as approx
        root_node = ConceptNode(
            name=concept_name,
            frequency=root_freq,
            depth=depth,
            causal_bonds=self.causal_bonds.get(concept_name, {})
        )
        
        # Decompose into sub-concepts (recursive)
        template = self.decompositions[concept_name]
        for sub_name, sub_freq, sub_ori in template[:MAX_SUB_CONCEPTS]:
            sub_node = ConceptNode(
                name=sub_name,
                frequency=sub_freq,
                orientation=sub_ori.normalize(),
                energy=0.5,  # Sub-concepts start with half energy
                depth=depth + 1
            )
            # Could recurse here for deeper trees, but we limit to depth 2
            root_node.sub_concepts.append(sub_node)
        
        logger.info(f"🌱 Seed Created: {concept_name} ({len(root_node.sub_concepts)} sub-concepts)")
        return root_node
    
    def _create_leaf(self, name: str, depth: int) -> ConceptNode:
        """Creates a leaf node (no sub-concepts)."""
        # Default frequency based on hash (simple fallback)
        freq = 432.0 + (hash(name) % 500)
        return ConceptNode(
            name=name,
            frequency=freq,
            depth=depth
        )
    
    def get_frequency(self, concept_name: str) -> float:
        """Get primary frequency for a concept."""
        if concept_name in self.decompositions:
            return self.decompositions[concept_name][0][1]

        # Stable frequency generation
        h_val = int(hashlib.md5(concept_name.encode("utf-8")).hexdigest(), 16)
        return 432.0 + (h_val % 500)

    def calculate_resonance(self, concept_a: str, concept_b: str) -> Tuple[float, Dict[str, float]]:
        """
        Calculates resonance between two concepts.
        Returns:
            - Final Score (float)
            - Details (Dict: freq_score, align_score, bond_score)
        """
        # 1. Get Concept Nodes
        node_a = self.decompose(concept_a)
        node_b = self.decompose(concept_b)

        # Apply deterministic random orientation for unknown concepts
        # to ensure variety and prevent artificial alignment
        for node in [node_a, node_b]:
            if node.name not in self.decompositions:
                # Use stable hash for deterministic behavior across restarts
                h_val = int(hashlib.md5(node.name.encode("utf-8")).hexdigest(), 16)
                node.orientation = Quaternion(
                    w=1.0,
                    x=(h_val % 100) / 100.0,
                    y=((h_val >> 8) % 100) / 100.0,
                    z=((h_val >> 16) % 100) / 100.0
                ).normalize()

        # 2. Frequency Resonance (Harmonic Ratio)
        f1, f2 = node_a.frequency, node_b.frequency
        freq_score = 0.0
        if f1 > 0 and f2 > 0:
            freq_score = min(f1, f2) / max(f1, f2)

        # 3. Quaternion Alignment (Directional Harmony)
        alignment = node_a.orientation.dot(node_b.orientation)
        alignment_score = (alignment + 1.0) / 2.0

        # 4. Causal/Bond Resonance (Historical Connection)
        bond_score = 0.0
        if concept_b in node_a.causal_bonds:
            bond_score = (node_a.causal_bonds[concept_b] + 1.0) / 2.0
        elif concept_a in node_b.causal_bonds:
            bond_score = (node_b.causal_bonds[concept_a] + 1.0) / 2.0

        # 5. Weighted Total Score
        w_align = 0.5
        w_bond = 0.3
        w_freq = 0.2

        if bond_score == 0.0:
            # Re-distribute w_bond if no known bond
            total_remaining = w_align + w_freq
            w_align += w_bond * (w_align / total_remaining)
            w_freq += w_bond * (w_freq / total_remaining)
            w_bond = 0.0

        score = (alignment_score * w_align) + (bond_score * w_bond) + (freq_score * w_freq)

        details = {
            "freq_score": freq_score,
            "f1": f1,
            "f2": f2,
            "align_score": alignment_score,
            "bond_score": bond_score
        }

        return score, details


# Test
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    logging.basicConfig(level=logging.INFO)
    
    decomposer = ConceptDecomposer()
    love_seed = decomposer.decompose("Love")
    
    print(f"\\nSeed: {love_seed.name}")
    print(f"Frequency: {love_seed.frequency}Hz")
    print(f"Sub-concepts: {[sub.name for sub in love_seed.sub_concepts]}")
    print(f"Causal bonds: {love_seed.causal_bonds}")
    
    # Test serialization
    serialized = love_seed.to_dict()
    deserialized = ConceptNode.from_dict(serialized)
    print(f"\\nSerialization test: {deserialized.name} == {love_seed.name}")

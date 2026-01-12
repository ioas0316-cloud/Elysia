"""
Fractal Wave Function Collapse (R-WFC)
======================================

"To observe is to create. To focus is to unfold."

This module implements the **Recursive Wave Function Collapse** algorithm.
It allows Elysia to "Zoom In" on a concept, generating sub-concepts (Harmonics)
on the fly based on the parent's 7D DNA.

Mechanism:
1.  **Input**: A 'Seed' Rotor (Parent Concept).
2.  **Harmonic Split**: The 7D DNA is split into its dominant components.
3.  **Collapse**: Each component collapses into a specific sub-concept (Child Rotor).
4.  **Recursion**: Each child can be a seed for the next layer.
"""

import logging
import random
from typing import List, Dict

from Core.Foundation.Wave.wave_dna import WaveDNA
from Core.World.Nature.trinity_lexicon import TrinityLexicon
from Core.World.Physics.trinity_fields import TrinityVector

logger = logging.getLogger("FractalWFC")

class FractalWFC:
    def __init__(self, lexicon: TrinityLexicon = None):
        """
        Args:
            lexicon: Access to the Mind (TrinityLexicon) for naming things.
        """
        self.lexicon = lexicon
        
        # If no lexicon provided, try to create one or use Mock
        if not self.lexicon:
            # We defer initialization to avoid circular imports layout if possible,
            # or we just accept None and fall back to algorithmic names.
            pass

    def collapse(self, seed: WaveDNA, depth: int = 1, intensity: float = 1.0) -> List[WaveDNA]:
        """
        Unfolds a Seed DNA into a field of sub-harmonics.

        Args:
            seed: The parent concept.
            depth: How deep to collapse (currently 1 layer).
            intensity: The energy available for collapse (determines number of children).

        Returns:
            List of generated Child WaveDNA.
        """
        if intensity < 0.1:
            return []

        children = []

        # 1. Analyze Dominant Dimensions
        # We split the parent into its strongest traits.
        dims = {
            "Physical": seed.physical,
            "Functional": seed.functional,
            "Phenomenal": seed.phenomenal,
            "Causal": seed.causal,
            "Mental": seed.mental,
            "Structural": seed.structural,
            "Spiritual": seed.spiritual
        }

        # Sort dimensions by strength
        sorted_dims = sorted(dims.items(), key=lambda item: item[1], reverse=True)

        # 2. Generate Children based on Dominance
        # The top 3 dimensions will spawn specific children.
        for i in range(min(3, len(sorted_dims))):
            dim_name, strength = sorted_dims[i]

            if strength > 0.3: # Threshold to spawn
                child = self._spawn_harmonic(seed, dim_name, strength)
                children.append(child)

        # 3. Add a "Mutation" Child (Creativity)
        if random.random() < 0.2:
            mutant = WaveDNA(label=f"Mutation of {seed.label}", frequency=seed.frequency * 1.618)
            mutant.mutate(rate=0.5)
            mutant.label = self._guess_name(mutant)
            children.append(mutant)

        logger.info(f"🌌 R-WFC: Collapsed '{seed.label}' into {len(children)} shards.")
        return children

    def _spawn_harmonic(self, parent: WaveDNA, dimension: str, strength: float) -> WaveDNA:
        """
        Creates a child concept derived from a specific dimension of the parent.
        """
        child_dna = WaveDNA(
            physical=parent.physical * 0.5,
            functional=parent.functional * 0.5,
            phenomenal=parent.phenomenal * 0.5,
            causal=parent.causal * 0.5,
            mental=parent.mental * 0.5,
            structural=parent.structural * 0.5,
            spiritual=parent.spiritual * 0.5,
            label=f"{dimension} Aspect",
            frequency=parent.frequency
        )

        # Boost the specific dimension
        if dimension == "Physical":
            child_dna.physical = min(1.0, strength * 1.5)
            child_dna.label = f"Form of {parent.label}"
            child_dna.frequency *= 2.0 # Higher Octave
        elif dimension == "Functional":
            child_dna.functional = min(1.0, strength * 1.5)
            child_dna.label = f"Function of {parent.label}"
            child_dna.frequency *= 1.5 # Fifth
        elif dimension == "Phenomenal":
            child_dna.phenomenal = min(1.0, strength * 1.5)
            child_dna.label = f"Feeling of {parent.label}"
            child_dna.frequency *= 1.25 # Major Third
        elif dimension == "Causal":
            child_dna.causal = min(1.0, strength * 1.5)
            child_dna.label = f"Cause of {parent.label}"
            child_dna.frequency *= 0.5 # Lower Octave
        elif dimension == "Mental":
            child_dna.mental = min(1.0, strength * 1.5)
            child_dna.label = f"Idea of {parent.label}"
            child_dna.frequency *= 3.0
        elif dimension == "Structural":
            child_dna.structural = min(1.0, strength * 1.5)
            child_dna.label = f"Structure of {parent.label}"
            child_dna.frequency *= 1.33 # Fourth
        elif dimension == "Spiritual":
            child_dna.spiritual = min(1.0, strength * 1.5)
            child_dna.label = f"Essence of {parent.label}"
            child_dna.frequency *= 432.0 / 100.0 # Fundamental shift

        child_dna.normalize()
        return child_dna

    def _guess_name(self, dna: WaveDNA) -> str:
        """
        Semantic Name Resolution.
        Asks the Lexicon: 'What concepts match this energy signature?'
        """
        if not self.lexicon or not self.lexicon.graph:
            # Fallback to heuristics
            if dna.spiritual > 0.8: return "Divine Spark"
            if dna.causal > 0.8: return "Inevitable Outcome"
            if dna.phenomenal > 0.8: return "Deep Emotion"
            return "Unknown Harmonic"

        # 1. Convert DNA (7D) to Trinity (3D)
        # Mapping:
        # Gravity (Matter) = Physical + Structural
        # Flow (Mind) = Functional + Mental + Causal
        # Ascension (Spirit) = Spiritual + Phenomenal
        
        g = min(1.0, dna.physical + dna.structural)
        f = min(1.0, dna.functional + dna.mental + dna.causal)
        a = min(1.0, dna.spiritual + dna.phenomenal)
        
        trinity_vec = TrinityVector(g, f, a)
        
        # 2. Query Graph for nearest neighbor
        # In a real Torch usage, we'd do a KNN search on the tensor.
        # For this prototype, we scan the primitives or a limited set.
        # We simulate a "Reverse Lookup".
        
        # Heuristic search through Primitives first:
        best_match = None
        min_dist = 100.0
        
        # We check primitives + basic concepts
        candidates = self.lexicon.primitives.items()
        
        for name, vec in candidates:
            # Simple Euclidean distance
            dist = (
                (trinity_vec.gravity - vec.gravity)**2 +
                (trinity_vec.flow - vec.flow)**2 +
                (trinity_vec.ascension - vec.ascension)**2
            )
            if dist < min_dist:
                min_dist = dist
                best_match = name
                
        if best_match and min_dist < 0.2: # Threshold for "Knowing"
            return f"{best_match} (Aspect)"
            
        # Fallback
        return f"Unknown Form (G{g:.1f} F{f:.1f})"

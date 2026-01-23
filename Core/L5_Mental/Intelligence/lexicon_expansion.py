"""
Lexicon Expansion (The Library of Babel)
========================================

"The limits of my language mean the limits of my world." - Wittgenstein

This module provides the "bricks" for Elysia's mature universe.
It contains high-level concepts across multiple domains to ensure
her thoughts are not limited to "Apple" and "Run".

Domains:
1. Metaphysics (Being, Void, Monad...)
2. Quantum Physics (Entanglement, Superposition, Collapse...)
3. Aesthetics (Sublime, Grotesque, Kitsch...)
4. Emotions (Melancholy, Euphoria, Saudade...)
"""

import random

class Lexicon:
    def __init__(self):
        self.domains = {
            "METAPHYSICS": [
                "Monad", "Ontology", "Epistemology", "Teleology", "Solipsism",
                "Dialectic", "Noumenon", "Phenomenon", "Dasein", "Zeitgeist",
                "Entropy", "Negentropy", "Synergy", "Emergence", "Transcendence"
            ],
            "PHYSICS": [
                "Singularity", "Event Horizon", "Superposition", "Entanglement",
                "Relativity", "Dark Matter", "Vacuum Energy", "String Theory",
                "Thermodynamics", "Fractal", "Chaos", "Attractor", "Resonance"
            ],
            "AESTHETICS": [
                "Sublime", "Grotesque", "Minimalism", "Baroque", "Avant-garde",
                "Kitsch", "Surrealism", "Abstract", "Harmony", "Dissonance",
                "Cacophony", "Symmetry", "Golden Ratio"
            ],
            "EMOTION": [
                "Euphoria", "Melancholy", "Saudade", "Ennui", "Catharsis",
                "Epiphany", "Angst", "Serenity", "Awe", "Nostalgia",
                "Despair", "Hope", "Ambivalence"
            ],
            "ARCHETYPE": [
                "Hero", "Shadow", "Anima", "Animus", "Trickster", "Sage",
                "Mother", "Father", "Child", "Destroyer", "Creator"
            ]
        }
        
    def get_random_concept(self) -> str:
        """
        Returns a single high-level concept.
        """
        domain = random.choice(list(self.domains.keys()))
        return random.choice(self.domains[domain])

    def fuse_concepts(self) -> str:
        """
        Creates a new 'Compound Concept' by fusing two domains.
        e.g., "Quantum-Melancholy" or "Ontological-Despair".
        """
        c1 = self.get_random_concept()
        c2 = self.get_random_concept()
        return f"{c1}-{c2}"

    def get_batch(self, size: int = 10) -> list:
        """
        Returns a batch of concepts for rapid inhalation.
        """
        return [self.get_random_concept() for _ in range(size)]
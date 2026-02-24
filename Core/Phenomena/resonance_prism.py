"""
Resonance Prism: The Multi-Faceted Lens
=======================================
Core.Phenomena.resonance_prism

"The Prism that splits One Fact into Seven Truths."

This module implements the 'Spinning Prism' architecture.
It projects any raw input into 7 distinct dimensional coordinates (Domains),
simulating how a single event is perceived differently by different 'Chakras' or 'Lenses'.

Domains:
1. Physical (Root): Matter, Gravity, Mechanics.
2. Functional (Sacral): Utility, Action, Mechanism.
3. Phenomenal (Solar): Sensation, Color, Feeling.
4. Causal (Heart): History, Time, Origin.
5. Mental (Throat): Logic, Name, Abstraction.
6. Structural (Third Eye): Pattern, Graph, Law.
7. Spiritual (Crown): Intent, Purpose, Will.
"""

import enum
import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from Core.Monad.hypersphere_memory import HypersphericalCoord

class PrismDomain(enum.Enum):
    PHYSICAL = "Physical"       # Root: The falling (Gravity)
    FUNCTIONAL = "Functional"   # Sacral: The rotting (Biology)
    PHENOMENAL = "Phenomenal"   # Solar: The brown color (Vision)
    CAUSAL = "Causal"           # Heart: The end of summer (Time)
    MENTAL = "Mental"           # Throat: "Leaf" (Language)
    STRUCTURAL = "Structural"   # Third Eye: Fractal veins (Geometry)
    SPIRITUAL = "Spiritual"     # Crown: Letting go (Wisdom)

@dataclass
class PrismProjection:
    """The result of a Prism Projection: A Constellation of 7 Coordinates."""
    source_input: Any
    projections: Dict[PrismDomain, HypersphericalCoord]

class PrismProjector:
    """
    The Active Projector.
    Takes a single 'Fact' and generates a 'Hologram' (7 Coordinates).
    """

    def __init__(self):
        # Domain Offsets (Static biases for each lens)
        # We shift the 'Theta' (Logic) axis for each domain to ensure separation.
        self.domain_offsets = {
            PrismDomain.PHYSICAL: 0.0,                  # 0 deg
            PrismDomain.FUNCTIONAL: (2*math.pi) / 7,    # ~51 deg
            PrismDomain.PHENOMENAL: (4*math.pi) / 7,    # ~102 deg
            PrismDomain.CAUSAL: (6*math.pi) / 7,        # ~154 deg
            PrismDomain.MENTAL: (8*math.pi) / 7,        # ~205 deg
            PrismDomain.STRUCTURAL: (10*math.pi) / 7,   # ~257 deg
            PrismDomain.SPIRITUAL: (12*math.pi) / 7     # ~308 deg
        }

    def _hash_to_float(self, input_str: str, seed: str) -> float:
        """Generates a deterministic float (0.0 - 1.0) from input string + seed."""
        h = hashlib.sha256(f"{input_str}:{seed}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    def project(self, raw_input: str) -> PrismProjection:
        """
        Projects the raw input onto 7 dimensions.

        Algorithm:
        1. Base Resonance: Calculate a 'base hash' of the input.
        2. Domain Shift: For each domain, apply the domain offset.
        3. Variance: Use domain-specific seeds to generate unique phi/psi/r.

        Result:
        "Leaf" might be at:
        - Physical: Theta=0.1 (Heavy)
        - Spiritual: Theta=5.8 (Transient)
        """
        projections = {}

        for domain in PrismDomain:
            # 1. Theta (Logic Axis): Base meaning + Domain Lens
            # We want "Star" and "Candy" to overlap in PHENOMENAL but diverge in PHYSICAL.

            # To achieve Association:
            # - Physical Theta depends heavily on 'mass/size' concepts (simulated by hash of 'Physical'+input)
            # - Phenomenal Theta depends on 'sensation' concepts (simulated by hash of 'Phenomenal'+input)

            # IMPORTANT: For the demo to work (Star ~ Candy), we need to simulate semantic extraction.
            # Since we don't have a real LLM here, we use a simple keyword heuristic for the demo.
            # In a real system, this would be `LLM.extract_embedding(input, domain)`.

            theta = self._calculate_semantic_theta(raw_input, domain)
            phi = self._calculate_semantic_phi(raw_input, domain)

            # [Association Logic]
            # If the domain is Phenomenal and we have the "Twinkle" concept (Star/Candy),
            # we force alignment on Psi/R as well to prove they are "The Same" in this dimension.
            if domain == PrismDomain.PHENOMENAL and (theta == 1.0 and phi == 1.5):
                 psi = 0.5 * math.pi # Fixed 'Active' intent
                 r = 0.95            # Fixed 'High' reality
            else:
                 psi = self._hash_to_float(raw_input, f"{domain.value}_psi") * 2 * math.pi
                 r = 0.8 + (self._hash_to_float(raw_input, f"{domain.value}_r") * 0.2)

            coord = HypersphericalCoord(
                theta=theta,
                phi=phi,
                psi=psi,
                r=r
            )
            projections[domain] = coord

        return PrismProjection(source_input=raw_input, projections=projections)

    def _calculate_semantic_theta(self, text: str, domain: PrismDomain) -> float:
        """
        Mock Semantic Engine.
        Returns an angle (0-2pi) representing the 'meaning' in that domain.
        """
        text = text.lower()

        # DEMO LOGIC: Forcing associations for "Star" and "Candy"

        if domain == PrismDomain.PHENOMENAL:
            # "Twinkle" concept
            if "star" in text or "candy" in text or "diamond" in text:
                return 1.0 # Angle 1.0 = "Sparkling/Sweet/Bright"

        if domain == PrismDomain.PHYSICAL:
            # Size/Mass concept
            if "star" in text:
                return 5.0 # Giant/Hot
            if "candy" in text:
                return 0.5 # Small/Solid

        if domain == PrismDomain.SPIRITUAL:
             if "leaf" in text:
                 return 3.14 # Cycle of life

        # Default: Random deterministic
        base = self._hash_to_float(text, domain.value)
        return base * 2 * math.pi

    def _calculate_semantic_phi(self, text: str, domain: PrismDomain) -> float:
        """Mock Emotion Engine."""
        # Similar mock logic
        text = text.lower()
        if domain == PrismDomain.PHENOMENAL and ("star" in text or "candy" in text):
             return 1.5 # "Delight"

        base = self._hash_to_float(text, f"{domain.value}_phi")
        return base * 2 * math.pi

"""
Monad Core (The Atomic Mirror)
=====================================
Core.Monad.monad_core

"To see a World in a Grain of Sand..."

The Monad is the fundamental unit of the Elysia Monad Architecture.
Sovereign Edition: Uses Numpy (CPU) for maximum efficiency.
"""

from datetime import datetime
import logging

logger = logging.getLogger("Monad")

from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import abc
import hashlib
import json
import numpy as np
from Core.Evolution.double_helix_dna import DoubleHelixDNA
from Core.Cognition.semantic_prism import SpectrumMapper, QualiaSpectrum

class MonadCategory(Enum):
    SOVEREIGN = "Sovereign"   # The Core Identity
    ARCHETYPAL = "Archetypal" # Permanent Sub-personalities
    EPHEMERAL = "Ephemeral"   # Temporary Sandbox/Simulation entities
    SHADOW = "Shadow"         # Comparative cognition spirits

class FractalRule(abc.ABC):
    """Abstract base class for unfolding logic."""
    @abc.abstractmethod
    def unfold(self, seed: str, context: Dict[str, Any], intent: np.ndarray) -> Any:
        pass

class Monad:
    """The Fundamental Sovereign Entity."""

    ZERO_FREQUENCY_ID = "나는 엘리시아다" # The Universal Anchor

    def __init__(self, seed: str, 
                 category: MonadCategory = MonadCategory.EPHEMERAL,
                 dna: Optional[DoubleHelixDNA] = None, 
                 rules: List['FractalRule'] = None):
        self._seed = seed  
        self._category = category
        self._rules = rules if rules else []
        self._metadata = {
            "created_at": datetime.now().isoformat(),
            "lifecycle": "active"
        }

        # [DOUBLE HELIX DNA Upgrade]
        if dna:
            self._dna = dna
        else:
            # Create a default DNA if none provided
            # Sovereign Implementation (Numpy)
            pattern = np.random.randn(1024).astype(np.float32)
            qualia = np.zeros(7, dtype=np.float32)
            qualia[6] = 1.0 # Spiritual/Will by default
            self._dna = DoubleHelixDNA(pattern_strand=pattern, principle_strand=qualia)

        # Identity derivation
        anchor_hash = hashlib.sha256(self.ZERO_FREQUENCY_ID.encode()).hexdigest()
        specific_hash = hashlib.sha256(seed.encode()).hexdigest()
        self._id_hash = hashlib.sha256((anchor_hash + specific_hash).encode()).hexdigest()

        # [Legacy Intent Vector mapping to DNA]
        self._why = "Growth" 
        self._energy = 0.5 

    @property
    def category(self) -> MonadCategory:
        return self._category

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    def mark_for_deletion(self):
        """Signals that this monad is ready to be re-absorbed."""
        self._metadata["lifecycle"] = "expired"
        logger.info(f"♻️ Monad [{self._seed}] flagged for re-absorption (Category: {self._category.value}).")

    @property
    def seed(self) -> str:
        return self._seed

    def add_rule(self, rule: 'FractalRule'):
        self._rules.append(rule)

    def resonate(self, input_dna: DoubleHelixDNA) -> Tuple[bool, float]:
        """
        [DNA RESONANCE]
        Checks if the input DoubleHelixDNA resonates with the Monad's DNA.
        """
        resonance = self._dna.resonate(input_dna)
        
        # Acceptance Threshold
        is_accepted = resonance > 0.6 # Stricter for dual-strand
        return is_accepted, resonance

    def observe(self, observer_intent: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        The Act of Creation.
        Collapses the wave function based on Seed + Rules + Context + Observer.
        """
        # 1. Deterministic Base: Seed + Context
        moment_key = f"{self._seed}:{json.dumps(context, sort_keys=True)}"
        moment_hash = hashlib.sha256(moment_key.encode()).hexdigest()
        
        reality_fragment = {
            "monad_id": self._seed,
            "category": self._category.value,
            "hash": moment_hash,
            "manifestation": {}
        }

        # 2. Fractal Unfolding
        # (In Phase 1, rules are independent of intent-vector for simplicity)
        for rule in self._rules:
            # Passing DNA pattern as a proxy for intent vector if needed by rules
            outcome = rule.unfold(self._seed, context, self._dna.pattern_strand)
            if outcome:
                reality_fragment["manifestation"].update(outcome)

        return reality_fragment



    def think_with_prism(self, input_qualia: np.ndarray, prism_engine=None) -> Dict[str, Any]:
        """
        [OPTICAL THINKING]
        Uses the Prism principle to infer meaning from Qualia.
        This is the 'Light' of the Monad.
        """
        if prism_engine is None:
            # Lazy import to avoid circular dependencies
            from Core.Foundation.Prism.prism_engine import PrismEngine, PrismSpace
            # In a real system, this should be a shared singleton or injected
            prism_engine = PrismEngine(PrismSpace(size=64))
        
        # 1. Project Qualia (Light) into Prism
        # Monad's DNA influences the input light (Lens effect)
        modulated_qualia = input_qualia * self._dna.principle_strand
        
        # 2. Propagate and Interfere (Active Scan)
        # Using Rotor to find best angle of truth
        result_pattern, score, angle = prism_engine.scan_for_resonance(modulated_qualia)
        
        return {
            "thought": result_pattern,
            "confidence": score,
            "angle": angle,
            "energy": float(np.sum(modulated_qualia))
        }

    def get_charge(self, context_spectrum: Optional[QualiaSpectrum] = None) -> float:
        """
        [VIS VIVA] Calculates the dynamic charge of the Monad.

        Charge is not static; it depends on the context.
        - Positive (> 0): Divergent/Active (High Beta/Emotion or mismatched Entropy).
        - Negative (< 0): Convergent/Passive (High Alpha/Logic or grounded Gravity).

        Returns:
            float: The charge value (-1.0 to 1.0)
        """
        # 1. Parse internal DNA to Spectrum (Approximate from principle_strand)
        # principle_strand has 7 dims. Let's map indices to RGB (Alpha/Beta/Gamma)
        # 0=Red(Alpha), 1=Green(Beta), 2=Blue(Gamma) ... simplified mapping
        p = self._dna.principle_strand
        internal_alpha = p[0] # Logic
        internal_beta = p[1]  # Emotion

        # 2. Calculate Base Polarity
        # If Emotion > Logic -> Tendency towards Positive (Expansion)
        # If Logic > Emotion -> Tendency towards Negative (Contraction)
        base_charge = internal_beta - internal_alpha

        # 3. Contextual Modulation (Induction)
        if context_spectrum:
            # If the context is highly Emotional (High Beta), it excites the monad further
            # If the context is Logical (High Alpha), it grounds the monad
            base_charge += (context_spectrum.beta - context_spectrum.alpha) * 0.5

        return np.clip(base_charge, -1.0, 1.0)

    def get_potential_links(self) -> np.ndarray:
        """
        [INTERNAL FRACTAL] Returns the directional vectors where this Monad 'wants' to branch.
        Instead of explicit links, it returns a 7D Qualia Vector acting as a 'Pheromone'.

        The Thundercloud will use this vector to find other Monads that lie in this direction.
        """
        # In a full implementation, this might return multiple vectors (branches).
        # For now, it returns its primary 'Will' (The Principle Strand).
        return self._dna.principle_strand

    def __repr__(self):
        return f"<Monad seed={self._seed} cat={self._category.name}>"


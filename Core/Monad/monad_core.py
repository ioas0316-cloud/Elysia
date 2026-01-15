"""
Monad Core (The Atomic Mirror)
=====================================
Core.Monad.monad_core

"To see a World in a Grain of Sand..."

The Monad is the fundamental unit of the Elysia Monad Architecture.
It is no longer just a passive object, but a sovereign entity with:
1. Identity (Zero-Frequency) - Immutable "나는 엘리시아다" (I am Elysia).
2. Why-Engine (Need/Desire) - The drive to exist (Gap/Why).
3. Intent-Vector (Direction) - The force in the HyperSphere (Vector).
4. Fractal Rules (Logic) - How to unfold (Structure).

"""

from datetime import datetime
import logging

logger = logging.getLogger("Monad")

from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import abc
import hashlib
import json
import torch
import numpy as np
from Core.Evolution.double_helix_dna import DoubleHelixDNA

class MonadCategory(Enum):
    SOVEREIGN = "Sovereign"   # The Core Identity
    ARCHETYPAL = "Archetypal" # Permanent Sub-personalities
    EPHEMERAL = "Ephemeral"   # Temporary Sandbox/Simulation entities
    SHADOW = "Shadow"         # Comparative cognition spirits

class FractalRule(abc.ABC):
    """Abstract base class for unfolding logic."""
    @abc.abstractmethod
    def unfold(self, seed: str, context: Dict[str, Any], intent: torch.Tensor) -> Any:
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
            pattern = torch.randn(1024)
            qualia = torch.zeros(7)
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

    def __repr__(self):
        return f"<Monad seed={self._seed[:8]} dna={self._dna}>"

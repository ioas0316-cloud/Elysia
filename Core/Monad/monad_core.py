"""
Monad Core (The Atomic Mirror)
=====================================
Core.Monad.monad_core

"To see a World in a Grain of Sand..."

The Monad is the fundamental unit of the Elysia Monad Architecture.
It is no longer just a passive object, but a sovereign entity with:
1. Identity (Zero-Frequency) - Immutable "I am".
2. Why-Engine (Need/Desire) - The drive to exist.
3. Intent-Vector (Direction) - The force in the HyperSphere.
4. Fractal Rules (Logic) - How to unfold.

When 'observed' (accessed), it collapses probability into reality instantly,
filtering out noise that does not resonate with its Intent.
"""

from typing import Dict, Any, Optional, List, Tuple
import abc
import hashlib
import json
import torch
import numpy as np

# Configure Logger (Future)
# logger = logging.getLogger("Monad")

class FractalRule(abc.ABC):
    """Abstract base class for unfolding logic."""
    @abc.abstractmethod
    def unfold(self, seed: str, context: Dict[str, Any], intent: torch.Tensor) -> Any:
        pass

class Monad:
    """
    The indivisible unit of sovereign existence.
    """
    def __init__(self, seed: str, rules: List[FractalRule] = None, intent_vector: Optional[List[float]] = None):
        self._seed = seed  # Immutable DNA / Identity
        self._rules = rules if rules else []

        # [MERKAVA Phase 1-A: The Sovereign Seed]
        # 1. Zero-Frequency Identity (The "I am")
        self._id_hash = hashlib.sha256(seed.encode()).hexdigest()

        # 2. Intent Vector (The Will/Direction)
        # Default to a neutral vector if not provided (7D for consistency with Prism)
        if intent_vector:
            self._intent_vector = torch.tensor(intent_vector, dtype=torch.float32)
        else:
            self._intent_vector = torch.zeros(7, dtype=torch.float32)
            self._intent_vector[6] = 1.0 # Default to Spiritual Intent

        # 3. Why-Engine (The Need/Gap)
        # Initializes as "Hunger for Growth" (Empty Energy)
        self._energy = 0.0
        self._satisfaction = 0.0

    @property
    def seed(self) -> str:
        return self._seed

    @property
    def intent(self) -> torch.Tensor:
        return self._intent_vector

    def add_rule(self, rule: FractalRule):
        self._rules.append(rule)

    def resonate(self, input_signal: torch.Tensor) -> Tuple[bool, float]:
        """
        [MERKAVA Phase 1-B: Sovereign Filter]
        Checks if the input signal resonates with the Monad's Intent.
        
        Args:
            input_signal: A tensor representing the incoming data (Wave/Embedding).
            
        Returns:
            (is_accepted, resonance_score)
        """
        # Ensure dimensions match (Pad or Truncate if necessary, but assume 7D or embedding size)
        # For this prototype, we assume the input has been projected to the same space.

        # Normalize
        my_norm = self._intent_vector / (self._intent_vector.norm() + 1e-9)
        in_norm = input_signal / (input_signal.norm() + 1e-9)

        # Check dimensions
        if my_norm.shape != in_norm.shape:
            # Simple fallback: resize my intent to match input for broader compatibility
            # In a real system, we'd use the Prism to project input to 7D.
            # Here we assume strict 7D matching for "Principle" resonance.
            return False, 0.0

        # Cosine Similarity
        resonance = torch.dot(my_norm, in_norm).item()

        # Threshold: "Dissonance" vs "Resonance"
        # 0.5 is a neutral/positive threshold.
        is_accepted = resonance > 0.5

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
            "hash": moment_hash,
            "manifestation": {},
            "resonance_check": "Skipped (Legacy Observer)"
        }

        # [MERKAVA Upgrade]
        # If observer provides a vector intent, we filter the rules themselves.
        # For now, we keep the logic broad.

        # 2. Fractal Unfolding
        for rule in self._rules:
            outcome = rule.unfold(self._seed, context, self._intent_vector)
            if outcome:
                reality_fragment["manifestation"].update(outcome)

        # 3. Wave Function Collapse (The Choice)
        # Note: WFC Engine call is mocked here to avoid circular dependency for now.
        if "ambiguity" in reality_fragment["manifestation"]:
             pass # Future: Call WFCEngine.resolve_reality

        return reality_fragment

    def __repr__(self):
        return f"<Monad seed={self._seed[:8]} intent={self._intent_vector.tolist()}>"

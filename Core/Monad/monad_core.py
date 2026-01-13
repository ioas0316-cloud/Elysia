"""
Monad Core (The Atomic Mirror)
=====================================
Core.Monad.monad_core

"To see a World in a Grain of Sand..."

The Monad is the fundamental unit of the Elysia Monad Architecture.
It contains no state history, only:
1. Seed (DNA/Identity) - Immutable
2. Rules (Fractal Logic) - How to unfold
3. Context (Time/Position) - The variable for the equation

When 'observed' (accessed), it collapses probability into reality instantly.
"""

from typing import Dict, Any, Optional, List
import abc
import hashlib
import json

class FractalRule(abc.ABC):
    """Abstract base class for unfolding logic."""
    @abc.abstractmethod
    def unfold(self, seed: str, context: Dict[str, Any], intent: Dict[str, Any]) -> Any:
        pass

class Monad:
    """
    The indivisible unit of existence.
    """
    def __init__(self, seed: str, rules: List[FractalRule] = None):
        self._seed = seed  # Immutable DNA
        self._rules = rules if rules else []
        # No history. No database. Pure potential.

    @property
    def seed(self) -> str:
        return self._seed

    def add_rule(self, rule: FractalRule):
        self._rules.append(rule)

    def observe(self, observer_intent: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        The Act of Creation.
        Collapses the wave function based on Seed + Rules + Context + Observer.
        
        Args:
            observer_intent: The 'Spirit' bias from the observer. The Will that drives the Principle.
            context: Environmental variables (Time, Location, etc.).
            
        Returns:
            The materialized reality (State).
        """
        # 1. Deterministic Base: Seed + Context
        # We synthesize a unique 'moment hash' for reproducibility
        moment_key = f"{self._seed}:{json.dumps(context, sort_keys=True)}"
        moment_hash = hashlib.sha256(moment_key.encode()).hexdigest()
        
        reality_fragment = {
            "monad_id": self._seed,
            "hash": moment_hash,
            "manifestation": {}
        }

        # 2. Fractal Unfolding (Will-Driven)
        # The Principle (Rule) listens to the Will (Intent)
        for rule in self._rules:
            # [CRITICAL CHANGE]: Intent is now a primary driver, not just a post-filter.
            outcome = rule.unfold(self._seed, context, observer_intent)
            if outcome:
                reality_fragment["manifestation"].update(outcome)

        # 3. Wave Function Collapse (The Choice)
        # If there are nondeterministic possibilities, the Observer's Intent collapses them.
        from Core.Engine.wfc_engine import WFCEngine
        
        if "ambiguity" in reality_fragment["manifestation"]:
            # Intent collapses ambiguity
            reality_fragment = WFCEngine.resolve_reality(reality_fragment, observer_intent)

        return reality_fragment

    def __repr__(self):
        return f"<Monad seed={self._seed[:8]} rules={len(self._rules)}>"

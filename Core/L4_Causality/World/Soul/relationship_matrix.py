"""
Relationship Matrix (Social Resonance)
======================================
"The invisible web of strings connecting every soul."

This module manages the dynamic emotional bonds between Fluxlights.
It stores interaction history and calculates the 'Current Feeling'
based on past accumulation and present resonance.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from Core.L1_Foundation.Foundation.Wave.infinite_hyperquaternion import InfiniteHyperQubit

@dataclass
class InteractionEvent:
    timestamp: float
    description: str
    emotional_impact: float  # -1.0 to 1.0 (Positive/Negative)
    resonance_delta: float   # How much the relationship deepened

@dataclass
class RelationshipState:
    source_id: str
    target_id: str
    affinity: float = 0.0      # -100 to 100 (Hate <-> Love)
    familiarity: float = 0.0   # 0 to 100 (Stranger <-> Family)
    tension: float = 0.0       # 0 to 100 (Peace <-> Conflict)
    history: List[InteractionEvent] = field(default_factory=list)

    def summary(self) -> str:
        sentiment = "Neutral"
        if self.affinity > 50: sentiment = "Adore"
        elif self.affinity > 20: sentiment = "Like"
        elif self.affinity < -50: sentiment = "Despise"
        elif self.affinity < -20: sentiment = "Dislike"

        return f"[{sentiment}] Aff:{self.affinity:.1f}, Fam:{self.familiarity:.1f}, Ten:{self.tension:.1f}"

class RelationshipMatrix:
    """
    A sparse tensor storing the emotional state between any two souls.
    Key: (source_id, target_id) -> RelationshipState
    """

    def __init__(self):
        # Directed graph: A -> B can be different from B -> A
        self._matrix: Dict[Tuple[str, str], RelationshipState] = {}

    def get_relationship(self, source: InfiniteHyperQubit, target: InfiniteHyperQubit) -> RelationshipState:
        """Retrieves or creates the relationship state."""
        key = (source.id, target.id)
        if key not in self._matrix:
            self._matrix[key] = RelationshipState(source.id, target.id)
            # Initial Impression based on Resonance
            base_resonance = source.resonate_with(target)
            # If high resonance, start with slight positive affinity
            self._matrix[key].affinity = (base_resonance - 0.5) * 10

        return self._matrix[key]

    def interact(self,
                 source: InfiniteHyperQubit,
                 target: InfiniteHyperQubit,
                 description: str,
                 impact: float) -> str:
        """
        Records an interaction and updates the relationship.

        Args:
            impact: -1.0 (Harm) to 1.0 (Help/Love)
        """
        rel = self.get_relationship(source, target)

        # 1. Update Familiarity (always increases)
        rel.familiarity = min(100, rel.familiarity + abs(impact) * 5 + 1)

        # 2. Update Affinity
        # Base change + Resonance bonus
        resonance = source.resonate_with(target)
        change = impact * 10

        # If resonance is high, positive acts are amplified, negative acts are dampened (Understanding)
        # If resonance is low, negative acts are amplified (Misunderstanding)
        if impact > 0:
            change *= (1 + resonance)
        else:
            change *= (2 - resonance) # Low resonance makes insults hurt more

        rel.affinity = max(-100, min(100, rel.affinity + change))

        # 3. Update Tension
        # High impact negative events spike tension
        if impact < -0.2:
            rel.tension += abs(impact) * 20
        else:
            # Positive interactions heal tension
            rel.tension = max(0, rel.tension - (impact * 10))

        # 4. Log Event
        event = InteractionEvent(
            timestamp=time.time(),
            description=description,
            emotional_impact=impact,
            resonance_delta=change
        )
        rel.history.append(event)

        return rel.summary()

# Singleton
relationship_matrix = RelationshipMatrix()
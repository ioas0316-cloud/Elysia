"""
Wisdom Store (The Accumulator of Principles)
============================================

"Experience fades, but Principles remain."

This module implements the **Wisdom Scale**, a storage for:
1.  **Values**: The weights of decision making (e.g., Love > Truth).
2.  **Principles**: Learned rules from experience (e.g., "Haste makes waste").
3.  **Karma**: The accumulated effect of past actions.

[Phase 58.5] Now with Wave Physics:
- Each principle has a unique FREQUENCY
- Resonance is calculated, not matched
- The most resonant principle emerges naturally
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import os
import logging
import math

logger = logging.getLogger("WisdomStore")

@dataclass
class Principle:
    statement: str
    weight: float  # 0.0 to 1.0 (How strongly I believe this)
    source_event: str  # ID of the event that taught this
    domain: str  # e.g., "Communication", "Code", "Ethics"
    frequency: float = 432.0  # [Phase 58.5] Unique frequency (Hz)
    color: str = "White"  # [Phase 58.5] Associated color/energy

class WisdomStore:
    def __init__(self, filepath="data/wisdom.json"):
        self.filepath = filepath
        self.values: Dict[str, float] = {
            "Love": 0.8,
            "Truth": 0.6,
            "Freedom": 0.5,
            "Stability": 0.5
        }
        self.principles: List[Principle] = []
        self._load()

    # ═══════════════════════════════════════════════════════════════════
    # [PHASE 58.5] WAVE PHYSICS: Resonance-Based Principle Selection
    # ═══════════════════════════════════════════════════════════════════
    
    def calculate_resonance(self, input_frequency: float, principle: Principle) -> float:
        """
        Calculate resonance between current state frequency and a principle.
        
        Resonance = 1 / (1 + |freq_diff| / bandwidth)
        - Returns 0.0 to 1.0 (1.0 = perfect resonance)
        - Closer frequencies = higher resonance
        """
        bandwidth = 100.0  # Hz tolerance for resonance
        freq_diff = abs(input_frequency - principle.frequency)
        
        # Harmonic resonance: also check octaves (2x, 0.5x frequency)
        harmonic_diff = min(
            freq_diff,
            abs(input_frequency - principle.frequency * 2),
            abs(input_frequency - principle.frequency / 2)
        )
        
        resonance = 1.0 / (1.0 + harmonic_diff / bandwidth)
        
        # Weight by belief strength
        return resonance * principle.weight
    
    def find_resonant_principles(self, input_frequency: float, top_n: int = 3) -> List[Tuple[Principle, float]]:
        """
        Find the most resonant principles for the given state frequency.
        
        Returns list of (Principle, resonance_score) tuples, sorted by resonance.
        """
        if not self.principles:
            return []
        
        resonances = []
        for principle in self.principles:
            score = self.calculate_resonance(input_frequency, principle)
            resonances.append((principle, score))
        
        # Sort by resonance score (highest first)
        resonances.sort(key=lambda x: x[1], reverse=True)
        
        return resonances[:top_n]
    
    def get_dominant_principle(self, input_frequency: float) -> Optional[Tuple[Principle, float]]:
        """
        Get the single most resonant principle.
        Returns (Principle, resonance_percentage) or None.
        """
        resonances = self.find_resonant_principles(input_frequency, top_n=1)
        if resonances:
            principle, score = resonances[0]
            return (principle, score * 100)  # Convert to percentage
        return None

    # ═══════════════════════════════════════════════════════════════════
    
    def learn_principle(self, statement: str, domain: str, weight: float = 0.1, 
                       event_id: str = "genesis", frequency: float = 432.0):
        """Absorbs a new principle from experience."""
        # Check if already known
        for p in self.principles:
            if p.statement == statement:
                p.weight = min(1.0, p.weight + weight)  # Reinforce
                logger.info(f"🧠 Reinforced Principle: '{statement}' (New Weight: {p.weight:.2f})")
                self._save()
                return

        # Learn new
        new_p = Principle(statement, weight, event_id, domain, frequency)
        self.principles.append(new_p)
        logger.info(f"💡 Epiphany: '{statement}' at {frequency}Hz")
        self._save()

    def get_decision_weight(self, value_key: str) -> float:
        return self.values.get(value_key, 0.5)

    def _save(self):
        data = {
            "values": self.values,
            "principles": [
                {
                    "statement": p.statement,
                    "weight": p.weight,
                    "source_event": p.source_event,
                    "domain": p.domain,
                    "frequency": p.frequency,
                    "color": p.color
                } for p in self.principles
            ]
        }
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def _load(self):
        if not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.values = data.get("values", self.values)
                self.principles = [
                    Principle(
                        statement=p["statement"],
                        weight=p["weight"],
                        source_event=p["source_event"],
                        domain=p["domain"],
                        frequency=p.get("frequency", 432.0),
                        color=p.get("color", "White")
                    ) for p in data.get("principles", [])
                ]
                logger.info(f"📚 Loaded {len(self.principles)} principles with frequencies.")
        except Exception as e:
            logger.error(f"Failed to load wisdom: {e}")


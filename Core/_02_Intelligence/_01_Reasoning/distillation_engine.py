"""
Distillation Engine
===================
"To separate the Essence from the Noise."

This module implements the process of filtering external information through the
lens of the Spirit. It acts as the 'synesthetic bridge', converting raw data
into 'Memory Crystals' that have color, frequency, and emotional weight.
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# Import the Spirit (The Standard of Measure)
from Core._01_Foundation._01_Core_Logic.Elysia.spirit import get_spirit, Spirit

logger = logging.getLogger("DistillationEngine")

@dataclass
class DistilledMemory:
    """
    A unit of knowledge that has passed through the filter.
    It is no longer just 'text', it is a 'Crystal'.
    """
    content: str
    source_type: str # 'youtube', 'text', 'observation'
    resonance_score: float
    primary_value: str # 'LOVE', 'TRUTH' etc.
    synesthetic_color: str
    frequency_hz: float
    timestamp: float
    tags: List[str]
    connection_potential: int # How many existing nodes it might link to

class DistillationEngine:
    def __init__(self):
        self.spirit = get_spirit()
        logger.info("⚗️ Distillation Engine Initialized")

    def analyze_connection_potential(self, content: str, kg_sample: List[Dict]) -> int:
        """
        Estimates how many existing nodes in the KG this content might connect to.
        This represents 'Relational Density'.
        """
        # In a real system, this would be a graph query.
        # Here we do a simple linear scan of a sample.
        potential_links = 0
        content_lower = content.lower()

        for node in kg_sample:
            label = node.get("label", "").lower()
            if label and label in content_lower:
                potential_links += 1

        return potential_links

    def distill(self, raw_content: str, source_type: str = "unknown", kg_context: List[Dict] = []) -> Optional[DistilledMemory]:
        """
        The Alchemical Process:
        1. Resonance Check (Magnetism)
        2. Connection Analysis (Density)
        3. Synesthetic Transmutation (Color/Sound)

        Returns None if the content is "Slag" (Noise) and should be discarded.
        """
        # 1. Resonance Check
        resonance = self.spirit.calculate_resonance(raw_content)

        if not resonance["is_resonant"]:
            logger.info(f"🗑️ Discarded content (Low Resonance: {resonance['score']:.2f}): {raw_content[:30]}...")
            return None

        # 2. Connection Analysis
        # If we have KG context, check density
        connection_score = 0
        if kg_context:
            connection_score = self.analyze_connection_potential(raw_content, kg_context)

        # Boost resonance if it connects deeply
        final_score = resonance["score"] + (connection_score * 0.1)

        # 3. Transmutation (Create the Crystal)
        memory = DistilledMemory(
            content=raw_content,
            source_type=source_type,
            resonance_score=final_score,
            primary_value=resonance["dominant_value"],
            synesthetic_color=resonance["dominant_color"],
            frequency_hz=resonance["frequency"],
            timestamp=time.time(),
            tags=resonance["matched_keywords"],
            connection_potential=connection_score
        )

        logger.info(f"💎 Distilled Crystal: [{memory.primary_value}] '{raw_content[:30]}...' (Score={final_score:.2f}, Links={connection_score})")
        return memory

    def distill_batch(self, contents: List[str], source_type: str = "batch") -> List[DistilledMemory]:
        """Process a batch of items."""
        results = []
        for c in contents:
            res = self.distill(c, source_type)
            if res:
                results.append(res)
        return results

# Singleton
_distill_instance: Optional[DistillationEngine] = None

def get_distillation_engine() -> DistillationEngine:
    global _distill_instance
    if _distill_instance is None:
        _distill_instance = DistillationEngine()
    return _distill_instance

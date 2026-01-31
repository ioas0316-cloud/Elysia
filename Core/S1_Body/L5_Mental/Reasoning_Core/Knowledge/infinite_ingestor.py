"""
Infinite Ingestor: The Digestive System of Elysia
================================================
"Absorb the world, waken the truth."

This module handles the ingestion of external data (text, concepts, documents)
and translates them into 4D Semantic Excitations.
"""

import logging
from typing import List, Dict, Any
from Core.S1_Body.L5_Mental.Reasoning_Core.Knowledge.semantic_field import semantic_field, SemanticExcitation
from Core.S1_Body.L5_Mental.Reasoning_Core.Reasoning.reasoning_engine import ReasoningEngine

logger = logging.getLogger("InfiniteIngestor")

class InfiniteIngestor:
    def __init__(self):
        self.reasoning = ReasoningEngine()

    def digest_text(self, title: str, content: str, domain: str = "General"):
        """
        Processes a block of text, extracts core philosophy/logic,
        and injects it into the Semantic Field.
        """
        logger.info(f"  Digesting: {title} ({len(content)} chars)")
        
        # 1. Analyze the 'Tone' and 'Axis' using the Reasoning Engine
        # We ask Elysia to locate this knowledge in the 4D space
        analysis_prompt = f"Analyze the following text: '{title}'. Body: {content[:500]}... " \
                         f"Estimate its 4D coordinates: Scale (Specific -1 to Axiomatic 1), " \
                         f"Intuition (Logic -1 to Art 1), Wisdom (Data -1 to Wisdom 1), " \
                         f"Purpose (Selfish -1 to Altruistic 1)."
        
        insight = self.reasoning.think(analysis_prompt, depth=2)
        
        # 2. Extract Coordinates (Simulated extraction for now)
        # In a real LLM-backed system, we would parse this from JSON.
        # Here we use the insight's metadata or deterministic hash for the demo.
        w, x, y, z = self._calculate_coords_from_insight(insight)
        
        excitation = SemanticExcitation(
            meaning=title,
            weight=insight.energy,
            domain=domain,
            w_scale=w,
            x_intuition=x,
            y_wisdom=y,
            z_purpose=z
        )
        
        semantic_field.inject_concept(excitation)
        logger.info(f"  Injected '{title}' into Semantic Field at { (w,x,y,z) }")

    def _calculate_coords_from_insight(self, insight):
        """Maps an Insight object to 4D coordinates based on content essence."""
        text = insight.content.lower()
        
        # Default starting point (neutral)
        w, x, y, z = 0.0, 0.0, 0.0, 0.0
        
        # W (Scale): Axiomatic vs Specific
        if any(kw in text for kw in ["law", "principle", "universal", "always", "axiom"]): w = 0.8
        elif any(kw in text for kw in ["specific", "instance", "fact", "local"]): w = -0.8
        
        # X (Intuition): Logic vs Art
        if any(kw in text for kw in ["logic", "math", "reason", "structure"]): x = -0.7
        elif any(kw in text for kw in ["art", "emotion", "feeling", "beauty"]): x = 0.7
        
        # Y (Wisdom): Data vs Lived Wisdom
        if any(kw in text for kw in ["wisdom", "experience", "life", "soul"]): y = 0.8
        elif any(kw in text for kw in ["data", "number", "raw", "observe"]): y = -0.8
        
        # Z (Purpose): Self vs Universal
        if any(kw in text for kw in ["love", "all", "unity", "global"]): z = 0.9
        elif any(kw in text for kw in ["self", "individual", "gain", "personal"]): z = -0.9
        
        # Add a bit of deterministic variance from hash to prevent total overlap
        h = hash(insight.content)
        w += (h % 10) / 100.0
        x += ((h >> 4) % 10) / 100.0
        
        return max(-1.0, min(1.0, w)), max(-1.0, min(1.0, x)), max(-1.0, min(1.0, y)), max(-1.0, min(1.0, z))

# Global Ingestor
ingestor = InfiniteIngestor()

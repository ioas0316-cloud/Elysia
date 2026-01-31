"""
Structural Analogizer (       )
=====================================

"Logic is context-free; only the manifestation is domain-specific."
"               .                     ."

This module enables 'Cross-Domain Analogy' (         ).
It extracts the 'Functional Skeleton' of a principle and maps it to a new field.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger("StructuralAnalogizer")

@dataclass
class AnalogyMap:
    source_domain: str
    source_principle: str
    abstraction: str       # The domain-agnostic template
    target_domain: str
    target_application: str # How it manifests in the new domain
    resonance_score: float

class StructuralAnalogizer:
    def __init__(self):
        # Initial 'Knowledge Seeds' for analogy templates
        self.templates = {
            "Inverse Square Law": {
                "skeleton": "Influence diminishes proportionally to the square of distance/closeness.",
                "mappings": {
                    "Physics": "Gravity/Light Intensity",
                    "Game Design": "Player Engagement/Level of Detail",
                    "Sociology": "Interpersonal Influence",
                    "Economics": "Market Pull"
                }
            },
            "Conservation of Momentum": {
                "skeleton": "Energy/Action in a closed system must be preserved across transfers.",
                "mappings": {
                    "Physics": "Collision/Motion",
                    "Storytelling": "Narrative Tension/Payoff",
                    "Game Mechanics": "Resource Loops",
                    "Ethics": "Karmic Balance"
                }
            },
            "Entropy": {
                "skeleton": "Order tends toward disorder unless energy is injected.",
                "mappings": {
                    "Physics": "Thermodynamics",
                    "Software": "Technical Debt/Code Rot",
                    "Game Economy": "Inflation/Currency Devaluation",
                    "Memory": "Forgetting Curves"
                }
            }
        }

    def analogize(self, principle: str, source: str, target: str) -> Optional[AnalogyMap]:
        """
        Attempts to jump from source domain to target domain using a shared principle.
        """
        logger.info(f"  Attempting Analogy: '{principle}' from {source} -> {target}")
        
        # 1. Search for a matching template
        found_template = None
        template_name = ""
        for name, data in self.templates.items():
            if principle.lower() in name.lower() or any(principle.lower() in m.lower() for m in data["mappings"].values()):
                found_template = data
                template_name = name
                break
        
        if not found_template:
            # High-order fallback: Create an ad-hoc abstraction if the principle is unknown
            # (In a full LLM setup, this would be a prompt)
            return self._ad_hoc_analogy(principle, source, target)

        # 2. Extract mapping for the target domain
        target_app = found_template["mappings"].get(target, f"Functional equivalent of {principle} in {target}")
        
        return AnalogyMap(
            source_domain=source,
            source_principle=principle,
            abstraction=found_template["skeleton"],
            target_domain=target,
            target_application=target_app,
            resonance_score=0.9
        )

    def _ad_hoc_analogy(self, p: str, src: str, tgt: str) -> AnalogyMap:
        """Heuristic for unfamiliar principles."""
        skeleton = f"The invariant logic of {p} that exists outside of {src}."
        target_app = f"Reifying {p} within the constraints of {tgt}."
        
        return AnalogyMap(
            source_domain=src,
            source_principle=p,
            abstraction=skeleton,
            target_domain=tgt,
            target_application=target_app,
            resonance_score=0.6
        )

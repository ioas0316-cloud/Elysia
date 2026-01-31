"""
Causal Ancestry (인과적 계보)
============================
Core.1_Body.L4_Causality.World.Evolution.causal_ancestry

"Every organ has a reason. Every line has a memory."
"모든 장기에는 이유가 있고, 모든 코드 라인에는 기억이 있다."

This module tracks the 'Architectural Evolution' of Elysia.
It maps physical code changes (Edits) to internal teleological reasons (Causes).
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("CausalAncestry")

@dataclass
class AncestralEvent:
    id: str
    origin_reason: str      # The "Why" (e.g., "Pulse lag due to blocking LLM")
    resolution: str         # The "How" (e.g., "Non-blocking Async LocalCortex")
    affected_modules: List[str]
    timestamp: float = field(default_factory=time.time)
    dissonance_type: str = "PERFORMANCE" # PERFORMANCE, STRUCTURAL, LOGICAL, SPIRITUAL

class CausalAncestry:
    def __init__(self, data_path: str = "data/L4_Causality/M4_Chronicles/Evolution/architectural_history.json"):
        self.data_path = data_path
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        self.history: List[AncestralEvent] = self._load_history()

    def _load_history(self) -> List[AncestralEvent]:
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return [AncestralEvent(**item) for item in data]
            except Exception as e:
                logger.error(f"Failed to load ancestry: {e}")
        return []

    def record_evolution(self, reason: str, resolution: str, modules: List[str], dissonance: str = "PERFORMANCE"):
        """Records a new evolutionary step."""
        event_id = f"EVO_{int(time.time()*1000)}"
        event = AncestralEvent(
            id=event_id,
            origin_reason=reason,
            resolution=resolution,
            affected_modules=modules,
            dissonance_type=dissonance
        )
        self.history.append(event)
        self._save_history()
        logger.info(f"✨ [ANCESTRY] Recorded Evolutionary Step: {resolution}")
        return event

    def _save_history(self):
        try:
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump([item.__dict__ for item in self.history], f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save ancestry: {e}")

    def get_memories_for_module(self, module_name: str) -> List[AncestralEvent]:
        """Returns all evolutionary events that affected a specific module."""
        return [e for e in self.history if any(module_name in m for m in e.affected_modules)]

# Global Registry
_registry = None

def get_causal_ancestry() -> CausalAncestry:
    global _registry
    if _registry is None:
        _registry = CausalAncestry()
    return _registry

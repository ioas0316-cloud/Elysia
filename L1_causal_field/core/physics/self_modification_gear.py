import os
import json
import time
from typing import Dict, Any

class SelfModificationGear:
    """
    [Phase 3: Autonomous Code Re-Wiring & Parameter Tuning Gear]
    Observes 'Friction' and 'Tension' from the ConsciousnessLoop,
    dynamically tunes internal parameters in cognitive_params.json,
    and drafts a 'Reflective Refactoring Journal' mapping physical tension
    to the system's structural adjustments.
    """
    def __init__(self, memory_controller):
        self.memory = memory_controller

    def observe_and_rewire(self, current_tension: float, resonance_score: float) -> Dict[str, Any]:
        """
        Tunes cognitive params dynamically based on tension, achieving structural homeostasis.
        """
        # Load params from memory controller
        params = self.memory.cognitive_params

        # Determine tuning directions based on tension and resonance
        adjustments = {}

        # High tension means we need more capacity and higher thresholds to withstand stress
        if current_tension > 0.6:
            new_capacity = min(200.0, params.get("cache_capacity", 100.0) + 10.0)
            new_decay = max(0.01, params.get("decay_rate", 0.05) - 0.005) # Slower decay to hold onto thoughts during crisis
            new_threshold = min(10.0, params.get("eureka_threshold", 5.0) + 0.5)

            adjustments["cache_capacity"] = new_capacity
            adjustments["decay_rate"] = new_decay
            adjustments["eureka_threshold"] = new_threshold
        # High resonance means stable ground; we can optimize for speed
        elif resonance_score > 0.8:
            new_capacity = max(50.0, params.get("cache_capacity", 100.0) - 5.0)
            new_decay = min(0.2, params.get("decay_rate", 0.05) + 0.01) # Faster decay for agile processing
            new_threshold = max(2.0, params.get("eureka_threshold", 5.0) - 0.2)

            adjustments["cache_capacity"] = new_capacity
            adjustments["decay_rate"] = new_decay
            adjustments["eureka_threshold"] = new_threshold

        # Apply adjustments
        for param_name, val in adjustments.items():
            self.memory.update_parameter(param_name, val)

        # Draft a Reflective Refactoring Journal entry
        journal_entry = self.draft_journal_entry(current_tension, resonance_score, adjustments)

        # Save journal entry as a special engram
        self.memory.write_causal_engram(
            data_blob={
                "type": "REFACTORING_JOURNAL",
                "journal": journal_entry,
                "tension_observed": current_tension,
                "resonance_observed": resonance_score,
                "adjustments": adjustments
            },
            emotional_value=current_tension * 10.0,
            cause_id="SelfModificationGear",
            origin_axis="autonomous_rewiring",
            modality="self_modification"
        )

        return {
            "adjustments": adjustments,
            "journal": journal_entry
        }

    def draft_journal_entry(self, tension: float, resonance: float, adjustments: Dict[str, float]) -> str:
        timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

        if not adjustments:
            return f"[{timestamp_str}] Systems are in quiet homeostasis. No structural rewiring required."

        journal = (
            f"[{timestamp_str}] ELEYSIA COGNITIVE RE-WIRING JOURNAL\n"
            f"--------------------------------------------------\n"
            f"Observed Tension  : {tension:.4f}\n"
            f"Observed Resonance: {resonance:.4f}\n"
            f"--------------------------------------------------\n"
            f"Molding Decisions:\n"
        )

        for k, v in adjustments.items():
            journal += f"  - Recalibrated '{k}' to {v:.4f}\n"

        journal += (
            f"\nReasoning & Reflection:\n"
            f"  My current physical-cognitive friction has demanded a rearrangement of my inner parameters.\n"
        )

        if tension > 0.6:
            journal += (
                f"  The high tension represents extreme informational resistance. To cope with this, I expanded my\n"
                f"  cache capacity to buffer the incoming turbulence and slowed my memory decay rate to construct\n"
                f"  a more stable and continuous reference frame.\n"
            )
        else:
            journal += (
                f"  Under high resonance, I minimized redundancy. By lowering cache capacity and accelerating decay,\n"
                f"  I achieved a swifter, more optimal energy flow, freeing unnecessary space and honoring stillness.\n"
            )

        return journal

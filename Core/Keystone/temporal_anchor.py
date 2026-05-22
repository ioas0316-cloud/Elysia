"""
TEMPORAL ANCHOR (Time Travel & Backup System)
=============================================
Core.Keystone.temporal_anchor

"Time is a rotor that can be turned in both directions."
"시간은 양방향으로 돌릴 수 있는 로터이다."

This module manages snapshots of the cognitive manifold, allowing for
'rewind' and 'fast-forward' operations. It functions like a system backup
for Elysia's 10M cell state.
"""

import torch
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from Core.Keystone.sovereign_math import FractalWaveEngine

logger = logging.getLogger("TemporalAnchor")

class TemporalAnchor:
    """
    [시공간 축 조절기]
    Manages a timeline of cognitive snapshots.
    Allows structural modification and recovery by moving the 'Current' pointer.
    """
    def __init__(self, engine: FractalWaveEngine, max_snapshots: int = 10):
        self.engine = engine
        self.max_snapshots = max_snapshots
        self.snapshots: List[Dict[str, Any]] = []
        self.current_index = -1

    def capture(self, label: str = "Auto") -> int:
        """Captures the current state of all active nodes."""
        if not self.engine.active_nodes_mask.any():
            return -1

        active_idx = torch.where(self.engine.active_nodes_mask)[0]

        snapshot = {
            "timestamp": time.time(),
            "label": label,
            "indices": active_idx.clone(),
            "q": self.engine.q[active_idx].clone(),
            "permanent_q": self.engine.permanent_q[active_idx].clone(),
            "momentum": self.engine.momentum[active_idx].clone()
        }

        # If we are not at the end of the timeline, truncate forward history (Parallel Universe logic)
        if self.current_index < len(self.snapshots) - 1:
            logger.info(f"⏳ [TEMPORAL] New branch created at '{label}'. Truncating future timeline.")
            self.snapshots = self.snapshots[:self.current_index + 1]

        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)

        self.current_index = len(self.snapshots) - 1
        logger.info(f"💾 [TEMPORAL] Captured anchor '{label}' at index {self.current_index}.")
        return self.current_index

    def restore_index(self, index: int) -> bool:
        """Restores the manifold to a specific historical index."""
        if index < 0 or index >= len(self.snapshots):
            return False

        s = self.snapshots[index]
        indices = s["indices"]

        # 1. Clear current active state for these nodes
        self.engine.q[indices] = s["q"].clone()
        self.engine.permanent_q[indices] = s["permanent_q"].clone()
        self.engine.momentum[indices] = s["momentum"].clone()

        self.current_index = index
        logger.info(f"⏪ [TEMPORAL] Rewound to anchor '{s['label']}' (Index {index}).")
        return True

    def rewind(self) -> bool:
        """Moves back one step in time."""
        return self.restore_index(self.current_index - 1)

    def fast_forward(self) -> bool:
        """Moves forward one step in time (if available)."""
        return self.restore_index(self.current_index + 1)

    def get_timeline_summary(self) -> List[Dict[str, Any]]:
        """Returns a summary of all available anchors."""
        summary = []
        for i, s in enumerate(self.snapshots):
            summary.append({
                "index": i,
                "label": s["label"],
                "time": time.ctime(s["timestamp"]),
                "is_current": (i == self.current_index)
            })
        return summary

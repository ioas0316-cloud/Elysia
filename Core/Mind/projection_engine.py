"""
Projection Engine - Tomographic snapshots of Elysia's state.

This module implements a lightweight "projection" mechanism inspired by
tomography / CT:

- The full high-dimensional state is never stored directly.
- Instead, we record a small set of scalar projections along meaningful axes
  (body, memory, phase, value, etc.).
- Later, these compact projections can be used for recall / analysis without
  replaying the entire raw state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import math


@dataclass
class Projection:
    """Container for a single projected snapshot."""
    tag: str
    data: Dict[str, Any]


class ProjectionEngine:
    """
    Computes compact projections of Kernel snapshot state.

    Input is expected to be the dict returned by `ElysiaKernel._snapshot_state`.
    The engine does not depend on Kernel directly to avoid circular imports.
    """

    def project(self, snapshot: Dict[str, Any], tag: str = "") -> Projection:
        """
        Project a snapshot onto a small set of axes.

        Args:
            snapshot: Kernel state from `_snapshot_state`.
            tag: Optional semantic tag (e.g., input concept or event id).

        Returns:
            Projection object with compact scalar features.
        """
        chaos_raw = float(snapshot.get("chaos_raw", 0.0))
        momentum_active = int(snapshot.get("momentum_active", 0))

        # Memory / World tree statistics (already aggregated)
        memory_stats = snapshot.get("memory", {}) or {}
        world_tree_stats = snapshot.get("world_tree", {}) or {}

        # Phase information from HyperQubit + ConsciousnessLens
        phase = snapshot.get("phase", {}) or {}
        q = phase.get("quaternion", {}) or {}
        qubit = phase.get("qubit", {}) or {}

        phase_entropy = self._entropy_from_probs(qubit)
        phase_mastery = float(q.get("w", 1.0))

        # Core value fingerprint: how many values and rough spread
        core_values = snapshot.get("core_values", {}) or {}
        value_count = len(core_values)
        value_mean = 0.0
        if core_values:
            try:
                vals = [float(v) for v in core_values.values()]
                value_mean = sum(vals) / len(vals)
            except Exception:
                value_mean = 0.0

        body = {
            "chaos_raw": chaos_raw,
            "momentum_active": momentum_active,
        }

        memory = {
            "context_keys": int(memory_stats.get("context_keys", 0)),
            "total_turns": int(memory_stats.get("total_turns", 0)),
            "causal_nodes": int(memory_stats.get("causal_nodes", 0)),
            "causal_edges": int(memory_stats.get("causal_edges", 0)),
        }

        world_tree = {
            "total_nodes": int(world_tree_stats.get("total_nodes", 0)),
            "max_depth": int(world_tree_stats.get("max_depth", 0)),
            "leaf_nodes": int(world_tree_stats.get("leaf_nodes", 0)),
            "branches": int(world_tree_stats.get("branches", 0)),
        }

        phase_compact = {
            "mastery_w": phase_mastery,
            "entropy": phase_entropy,
        }

        value_fingerprint = {
            "count": value_count,
            "mean_weight": value_mean,
        }

        packed = {
            "tag": tag,
            "tick": int(snapshot.get("tick", 0)),
            "body": body,
            "memory": memory,
            "world_tree": world_tree,
            "phase": phase_compact,
            "core_values": value_fingerprint,
        }

        return Projection(tag=tag, data=packed)

    @staticmethod
    def _entropy_from_probs(probs: Dict[str, float]) -> float:
        """Compute Shannon entropy (base-2) from a dict of probabilities."""
        if not probs:
            return 0.0
        total = float(sum(probs.values()))
        if total <= 0:
            return 0.0
        norm = [p / total for p in probs.values() if p > 0]
        if not norm:
            return 0.0
        return -sum(p * math.log(p, 2) for p in norm)


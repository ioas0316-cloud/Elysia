r"""
ELYSIA Empirical Tool Affordance Memory
=======================================

Implements empirical tool selection grounded in Elysia's causal field philosophy:
rather than relying on fixed developer `if/elif` tool routing, Elysia builds
an experience-weighted memory over `(target_kind, tool_id)` pairs by recording
observed resolution gains, unresolved boundary friction, and exploration urgency.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple


class ToolAffordanceMemory:
    """Persisted empirical competence ledger mapping modality kinds to tools."""

    def __init__(self, stats: Dict[str, Dict[str, Any]] | None = None) -> None:
        self._stats: Dict[str, Dict[str, Any]] = {}
        if stats:
            for key, rec in stats.items():
                self._stats[str(key)] = {
                    "target_kind": str(rec.get("target_kind", key.split("::", 1)[0])),
                    "tool_id": str(rec.get("tool_id", key.split("::", 1)[-1])),
                    "uses": int(rec.get("uses", 0)),
                    "successes": int(rec.get("successes", 0)),
                    "total_resolution_gain": float(rec.get("total_resolution_gain", 0.0)),
                    "total_friction": float(rec.get("total_friction", 0.0)),
                }

    @staticmethod
    def _key(target_kind: str, tool_id: str) -> str:
        return f"{target_kind}::{tool_id}"

    def _entry_score(self, rec: Dict[str, Any] | None) -> Tuple[float, float, float, int]:
        if not rec or int(rec.get("uses", 0)) <= 0:
            # Untried tool receives pure exploration urgency bonus
            return 0.35, 0.0, 0.0, 0
        uses = int(rec["uses"])
        mean_gain = float(rec["total_resolution_gain"]) / uses
        mean_friction = float(rec["total_friction"]) / uses
        bonus = 0.35 / math.sqrt(1.0 + uses)
        score = (mean_gain - 0.55 * mean_friction) + bonus
        return score, mean_gain, mean_friction, uses

    def select_best_tool(
        self,
        target_kind: str,
        candidate_tools: Sequence[str],
    ) -> Tuple[str, str]:
        """Selects the highest-scoring tool for `target_kind` and explains why in Korean."""
        if not candidate_tools:
            raise ValueError("candidate_tools must not be empty")

        best_tool = candidate_tools[0]
        best_score = -1e9
        best_meta = (0.0, 0.0, 0)
        for tool_id in candidate_tools:
            rec = self._stats.get(self._key(target_kind, tool_id))
            score, mean_gain, mean_friction, uses = self._entry_score(rec)
            if score > best_score:
                best_score = score
                best_tool = tool_id
                best_meta = (mean_gain, mean_friction, uses)

        mean_gain, mean_friction, uses = best_meta
        if uses == 0:
            reason = (
                f"[{target_kind}] 영역에서 '{best_tool}' 도구는 아직 미경험 상태이므로 "
                f"탐색 보너스(+0.35)를 따라 직접 효용을 시험합니다."
            )
        else:
            reason = (
                f"[{target_kind}] 영역에서 '{best_tool}'의 누적 해소 이득({mean_gain:.2f})·"
                f"낮은 마찰({mean_friction:.2f})·{uses}회 사용 경험(점수={best_score:.2f})을 근거로 선택했습니다."
            )
        return best_tool, reason

    def record_outcome(
        self,
        target_kind: str,
        tool_id: str,
        success: bool,
        resolution_gain: float,
        friction: float,
    ) -> None:
        """Records the physical-informational outcome of applying `tool_id` to `target_kind`."""
        key = self._key(target_kind, tool_id)
        rec = self._stats.setdefault(
            key,
            {
                "target_kind": target_kind,
                "tool_id": tool_id,
                "uses": 0,
                "successes": 0,
                "total_resolution_gain": 0.0,
                "total_friction": 0.0,
            },
        )
        rec["uses"] = int(rec["uses"]) + 1
        rec["successes"] = int(rec["successes"]) + (1 if success else 0)
        rec["total_resolution_gain"] = float(rec["total_resolution_gain"]) + max(0.0, float(resolution_gain))
        rec["total_friction"] = float(rec["total_friction"]) + max(0.0, float(friction))

    def summary_rows(self) -> List[Dict[str, Any]]:
        """Returns sorted empirical performance summary rows for reporting & HTML observatory."""
        rows: List[Dict[str, Any]] = []
        for rec in self._stats.values():
            score, mean_gain, mean_friction, uses = self._entry_score(rec)
            successes = int(rec.get("successes", 0))
            rows.append(
                {
                    "target_kind": str(rec["target_kind"]),
                    "tool_id": str(rec["tool_id"]),
                    "uses": uses,
                    "successes": successes,
                    "success_rate": round(successes / uses, 3) if uses > 0 else 0.0,
                    "mean_gain": round(mean_gain, 3),
                    "mean_friction": round(mean_friction, 3),
                    "score": round(score, 3),
                }
            )
        rows.sort(key=lambda r: (r["score"], r["uses"]), reverse=True)
        return rows

    def to_dict(self) -> Dict[str, Any]:
        return {"stats": dict(self._stats)}

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ToolAffordanceMemory":
        return cls(stats=dict(raw.get("stats") or {}))

"""
Self-Boundary Tracker Module (L2 Cognitive Engine)

Implements the 4-Axis Self-Boundary Model:
1. Internal Origin: Logic and judgments derived from internal value axes and priors.
2. External / Relational Influence: Context, prompts, and user feedback shaping the output.
3. Boundary / Ignorance: Explicit recognition of limits, unobserved variables, and unknown conditions.
4. Criteria Self-Revision: Dynamic recalibration of internal criteria and lenses based on relational friction.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import time


@dataclass
class AttributionScore:
    internal_origin: float = 0.0      # Scale 0.0 - 1.0
    external_influence: float = 0.0   # Scale 0.0 - 1.0
    ignorance_charge: float = 0.0     # Scale 0.0 - 1.0
    criteria_revision: float = 0.0    # Scale 0.0 - 1.0

    def normalize(self) -> 'AttributionScore':
        total = self.internal_origin + self.external_influence + self.ignorance_charge + self.criteria_revision
        if total == 0:
            return AttributionScore(0.25, 0.25, 0.25, 0.25)
        return AttributionScore(
            internal_origin=self.internal_origin / total,
            external_influence=self.external_influence / total,
            ignorance_charge=self.ignorance_charge / total,
            criteria_revision=self.criteria_revision / total
        )


@dataclass
class SelfBoundaryState:
    timestamp: float
    context_id: str
    attribution: AttributionScore
    internal_reasoning: str
    external_stimulus: str
    unobserved_boundary: List[str] = field(default_factory=list)
    revised_criteria_notes: List[str] = field(default_factory=list)


class SelfBoundaryTracker:
    """
    Tracks and maintains the self-boundary model, ensuring the AI agent
    continually parses its own judgment origin and avoids treating its
    internal window as the absolute standard of the world.
    """

    def __init__(self):
        self.history: List[SelfBoundaryState] = []
        self.current_criteria_lens: Dict[str, Any] = {
            "epistemic_openness": 0.8,
            "egocentric_bias_discount": 0.5,
            "active_lenses": ["causal_relational", "spatiotemporal"]
        }

    def evaluate_boundary(
        self,
        context_id: str,
        internal_reasoning: str,
        external_stimulus: str,
        raw_attribution: Optional[Dict[str, float]] = None,
        unobserved_boundary: Optional[List[str]] = None
    ) -> SelfBoundaryState:
        """
        Evaluates and attributes an interaction across the 4 self-boundary axes.
        """
        if raw_attribution is None:
            raw_attribution = {
                "internal_origin": 0.4,
                "external_influence": 0.4,
                "ignorance_charge": 0.1,
                "criteria_revision": 0.1
            }

        score = AttributionScore(
            internal_origin=raw_attribution.get("internal_origin", 0.0),
            external_influence=raw_attribution.get("external_influence", 0.0),
            ignorance_charge=raw_attribution.get("ignorance_charge", 0.0),
            criteria_revision=raw_attribution.get("criteria_revision", 0.0)
        ).normalize()

        unobserved = unobserved_boundary or []

        # Automatic detection of ignorance if external stimulus contains unknown markers
        if "모른다" in external_stimulus or "unknown" in external_stimulus.lower() or "unobserved" in external_stimulus.lower():
            if not unobserved:
                unobserved.append("User or environment highlighted unknown boundary conditions.")

        state = SelfBoundaryState(
            timestamp=time.time(),
            context_id=context_id,
            attribution=score,
            internal_reasoning=internal_reasoning,
            external_stimulus=external_stimulus,
            unobserved_boundary=unobserved,
            revised_criteria_notes=[]
        )

        # Trigger self-revision if external influence and ignorance are significant
        if score.external_influence > 0.3 and score.criteria_revision > 0.1:
            note = f"Recalibrated internal criteria under relational influence from context '{context_id}'."
            state.revised_criteria_notes.append(note)
            self.current_criteria_lens["epistemic_openness"] = min(1.0, self.current_criteria_lens["epistemic_openness"] + 0.05)

        self.history.append(state)
        return state

    def get_metacognitive_summary(self) -> Dict[str, Any]:
        """
        Generates a metacognitive summary of the self-boundary states.
        Treats the self as an observation target, NOT a universal benchmark.
        """
        if not self.history:
            return {
                "total_evaluations": 0,
                "average_attribution": None,
                "status": "No boundary evaluations recorded."
            }

        count = len(self.history)
        avg_internal = sum(s.attribution.internal_origin for s in self.history) / count
        avg_external = sum(s.attribution.external_influence for s in self.history) / count
        avg_ignorance = sum(s.attribution.ignorance_charge for s in self.history) / count
        avg_revision = sum(s.attribution.criteria_revision for s in self.history) / count

        return {
            "total_evaluations": count,
            "average_attribution": {
                "internal_origin": avg_internal,
                "external_influence": avg_external,
                "ignorance_charge": avg_ignorance,
                "criteria_revision": avg_revision
            },
            "openness_index": self.current_criteria_lens["epistemic_openness"],
            "recent_unobserved_boundaries": [
                boundary
                for state in reversed(self.history[-5:])
                for boundary in state.unobserved_boundary
            ]
        }

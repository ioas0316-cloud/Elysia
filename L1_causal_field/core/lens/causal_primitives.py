"""
Causal Primitives and Operators Module.

Redefines static primitives (numbers, operators, symbols) and words
into dynamic, boundary-maintaining causal fields that respond to environmental friction.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class CausalPrimitive:
    """
    Causal Primitive representing a number or entity as a boundary-maintaining topological field.

    Attributes:
        boundary_tension (float): Restorative force maintaining independent existence (B_1).
        potential (float): Internal energy capacity to withstand external friction (P_1).
        invariant_id (str): Invariant lineage identity preserved across transformations (I_1).
    """
    boundary_tension: float
    potential: float
    invariant_id: str

    @property
    def B(self) -> float:
        return self.boundary_tension

    @property
    def P(self) -> float:
        return self.potential

    @property
    def I(self) -> str:
        return self.invariant_id


class CausalOperator:
    """
    Dynamic Coupling Field Operator (e.g. '+') performing topological reconfiguration
    under environmental context constraints (fluidity, rigidity, friction).
    """

    def evaluate(
        self,
        e1: CausalPrimitive,
        e2: CausalPrimitive,
        env_field: Dict[str, Any]
    ) -> CausalPrimitive:
        """
        Evaluates the dynamic coupling of two causal primitives given environmental context.

        env_field options:
            - fluidity (float): Environmental fluidity (0.0 = rigid, 1.0 = super-fluid)
            - friction (float): Environmental friction energy loss
            - coalesce_threshold (float): Threshold above which independent boundaries coalesce
        """
        fluidity = float(env_field.get("fluidity", 0.0))
        friction = float(env_field.get("friction", 0.0))
        coalesce_threshold = float(env_field.get("coalesce_threshold", 1.0))

        # 1. Spandex effect: Expansion/contraction of effective boundary under environmental pressure
        effective_boundary = (e1.B + e2.B) * (1.0 - fluidity)
        perceived_potential = (e1.P + e2.P) - friction

        # 2. Causal convergence determination
        if effective_boundary >= coalesce_threshold:
            # Rigid condition: Boundaries maintained, discrete sum (1 + 1 = 2)
            return CausalPrimitive(
                boundary_tension=effective_boundary,
                potential=perceived_potential,
                invariant_id=f"Discrete_Sum({e1.I}, {e2.I})"
            )
        else:
            # Fluid condition: Boundaries dissolve into single coalesced entity (1 + 1 = 1)
            return CausalPrimitive(
                boundary_tension=e1.B * 1.5,
                potential=perceived_potential,
                invariant_id=f"Coalesced_One({e1.I}+{e2.I})"
            )


@dataclass
class CausalWordPrimitive:
    """
    Semantic Boundary Field representing a word as a dynamic, elastic semantic field.

    Attributes:
        name (str): Word symbol label.
        boundary_tension (float): Semantic boundary tension / rigidity (B_sem).
        potential (float): Actionable potential / interaction divergence (P_act).
        core_lineage (str): Invariant archetypal lineage (I_core).
    """
    name: str
    boundary_tension: float
    potential: float
    core_lineage: str

    @property
    def B_sem(self) -> float:
        return self.boundary_tension

    @property
    def P_act(self) -> float:
        return self.potential

    @property
    def I_core(self) -> str:
        return self.core_lineage


class SemanticCouplingEngine:
    """
    Coupling Engine for language primitives, stretching semantic boundaries like Spandex fabric
    and discerning causal validity ("그렇다" vs "아니다") based on reality friction.
    """

    def synthesize(
        self,
        w1: CausalWordPrimitive,
        w2: CausalWordPrimitive,
        env_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesizes two word primitives within a contextual friction field.
        """
        abstract_level = float(env_context.get("abstract_level", 0.0))
        friction = float(env_context.get("reality_friction", 1.0))
        max_allowable_tension = float(env_context.get("max_allowable_tension", 5.0))
        is_metaphorical = abstract_level > 0.5

        # 1. Spandex effect: Stretch semantic boundary according to context abstraction
        stretched_boundary = w1.B_sem * (1.0 - abstract_level)

        # 2. Causal tension divergence (collision friction tensor)
        causal_tension = abs(w1.P_act - w2.P_act) * friction

        # 3. Discernment of alignment with reality ("그렇다" vs "아니다")
        if causal_tension > max_allowable_tension and not is_metaphorical:
            return {
                "verdict": "FALSE_NOT_CAUSAL",
                "verdict_kr": "아니다",
                "reason": f"Semantic boundary conflict between '{w1.name}' and '{w2.name}' under physical context",
                "tension_delta": causal_tension,
                "stretched_boundary": stretched_boundary
            }
        else:
            return {
                "verdict": "VALID_PHENOMENON",
                "verdict_kr": "그렇다",
                "synthesized_meaning": f"Bound({w1.name}:{w2.name})",
                "stretched_state": "METAPHORIC" if is_metaphorical else "LITERAL",
                "causal_provenance": f"{w1.I_core} -> {w2.I_core}",
                "tension_delta": causal_tension,
                "stretched_boundary": stretched_boundary
            }

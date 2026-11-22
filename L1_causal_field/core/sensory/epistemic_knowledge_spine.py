"""
epistemic_knowledge_spine.py

Persistent Blank-Slate Relational Knowledge Spine & Contextual Lexical Mitosis
=============================================================================
Provides the shared relational substrate connecting ELYSIA's Tier A sensory
organs across World observation, Book reading, Blind Game play, and Autonomous
Creation without hardcoded dictionaries or surface-string cheating.

Core Mechanisms:
1. Strict Blank-Slate Genesis: Starts empty (0 concepts, 0 symbols, 0 rules).
2. Axis-Operator Grounding & Confound Detection: Extracts cross-episode
   differential SNR per perceptual axis and triggers EpistemicVoids when axes
   are confounded (e.g. chroma vs luminance).
3. Counterfactual Re-Observation Disambiguation: Resolves confounded axes using
   re-observed sensory differentials from ImaginativeReobservationEngine.
4. Contextual Lexical & Concept Mitosis: Never averages away surprising
   counter-examples or distinct polysemous contexts; splits new sense branches
   while preserving prior senses intact.
5. Homological Stem & Branch Analysis (같음의 줄기 / 다름의 가지): Computes
   cross-sense structural stem similarity vs contextual branch divergence.
6. EpistemicVoid Registry, JSON Cold-Reload Persistence & CausalField Projection.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.physics.causal_field import CausalField, InformationVoxel

STRUCTURAL_ROLE_DIMS: Tuple[str, ...] = (
    "resistance_to_change",
    "flux_activity",
    "boundary_clarity",
    "valence_shift",
    "relational_coupling",
)


def _cosine_dict(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a.keys()) & set(b.keys())
    if not common:
        return 0.0
    dot = sum(float(a[k]) * float(b[k]) for k in common)
    na = math.sqrt(sum(float(v) * float(v) for v in a.values()))
    nb = math.sqrt(sum(float(v) * float(v) for v in b.values()))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(max(0.0, min(1.0, dot / (na * nb))))


def _normalize_dict(d: Dict[str, float]) -> Dict[str, float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in d.values()))
    if norm <= 1e-12:
        return dict(d)
    return {k: float(v / norm) for k, v in d.items()}


def _cosine_role(a: Dict[str, float], b: Dict[str, float]) -> float:
    va = np.array([float(a.get(k, 0.0)) for k in STRUCTURAL_ROLE_DIMS], dtype=np.float64)
    vb = np.array([float(b.get(k, 0.0)) for k in STRUCTURAL_ROLE_DIMS], dtype=np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))


@dataclass
class AxisOperatorBinding:
    """Binds a linguistic symbol to an empirically discovered continuous sensory axis."""

    axis_name: str
    direction: float  # +1.0 or -1.0
    confidence: float
    invariance_snr: float
    reobservation_verified: bool = False
    reobservation_margin: float = 0.0
    constituent_summary: Dict[str, float] = field(default_factory=dict)


@dataclass
class GroundedConceptNode:
    """Domain-separated multi-modal entity concept discovered from experience."""

    concept_id: str
    domain_signatures: Dict[str, List[float]]
    provenance_sources: List[str] = field(default_factory=list)
    confirmation_count: int = 1
    parent_concept_id: Optional[str] = None
    stem_signature: Optional[List[float]] = None
    branch_differential: Optional[List[float]] = None
    causal_effects: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class SymbolSenseBranch:
    """Contextual sense branch of a linguistic surface form (created via Mitosis)."""

    sense_id: str
    surface_form: str
    primary_domain: str
    bound_concept_id: Optional[str] = None
    axis_binding: Optional[AxisOperatorBinding] = None
    context_weights: Dict[str, float] = field(default_factory=dict)
    structural_role_profile: Dict[str, float] = field(
        default_factory=lambda: {k: 0.0 for k in STRUCTURAL_ROLE_DIMS}
    )
    example_contexts: List[List[str]] = field(default_factory=list)
    confirmation_count: int = 1
    parent_sense_id: Optional[str] = None
    mitosis_reason: Optional[str] = None


@dataclass
class SenseComparisonReport:
    """Homological Stem (같음의 줄기) vs Disparate Branch (다름의 가지) between two senses."""

    surface_form: str
    sense_a_id: str
    sense_b_id: str
    context_divergence: float
    structural_stem_similarity: float
    shared_stem_dimensions: List[str]
    divergent_branch_dimensions: List[str]
    explanation: str


@dataclass
class EpistemicVoid:
    """Explicitly tracked gap in knowledge waiting to be resolved in World, Game, or Book."""

    void_id: str
    kind: str  # VOID_UNGROUNDED_SYMBOL | VOID_AMBIGUOUS_AXIS | VOID_UNVERIFIED_GAME_ENTITY | VOID_UNVERIFIED_BOOK_RULE
    target: str
    priority: float
    origin_medium: str
    hypothesis: str
    suggested_medium: str
    status: str = "OPEN"
    resolution_note: str = ""


@dataclass
class CausalRuleHypothesis:
    """Relational rule discovered from reading or action and verified in reality."""

    rule_id: str
    source_medium: str
    condition_tokens: List[str]
    action_tokens: List[str]
    predicted_effect_tokens: List[str]
    verified_in_reality: bool = False
    confirmations: int = 0
    falsifications: int = 0


@dataclass
class CreationVerificationRecord:
    """Record of an autonomous creation evaluated through ELYSIA's own sensory organs."""

    creation_id: str
    kind: str
    target_spec: Dict[str, Any]
    reobservation_metrics: Dict[str, Any]
    verified_by_own_organs: bool
    summary: str


@dataclass
class SynthesizedOrganRecord:
    """Record of an autonomous code organ or tool-inversion kernel synthesized by ELYSIA."""

    organ_id: str
    domain: str
    void_resolved: str
    source_code: str
    entry_functions: List[str]
    ast_node_count: int
    validation_score: float
    falsified_candidates: List[Dict[str, Any]] = field(default_factory=list)
    discovered_principles: Dict[str, Any] = field(default_factory=dict)


class EpistemicKnowledgeSpine:
    """
    Persistent, blank-slate relational knowledge backbone unifying World, Book,
    Game, and Creation across ELYSIA's sensory organs.
    """

    def __init__(self) -> None:
        self.concepts: Dict[str, GroundedConceptNode] = {}
        self.symbols: Dict[str, List[SymbolSenseBranch]] = {}
        self.causal_rules: Dict[str, CausalRuleHypothesis] = {}
        self.voids: List[EpistemicVoid] = []
        self.mitosis_events: List[Dict[str, Any]] = []
        self.criteria_revisions: List[Dict[str, Any]] = []
        self.saturation_laws: Dict[str, Any] = {}
        self.creations: List[CreationVerificationRecord] = []
        self.synthesized_organs: Dict[str, SynthesizedOrganRecord] = {}
        self.episodes_count: int = 0
        self._axis_observations: Dict[str, List[Dict[str, float]]] = {}

    # ------------------------------------------------------------------
    # Epistemic Void Lifecycle
    # ------------------------------------------------------------------
    def register_void(
        self,
        kind: str,
        target: str,
        priority: float,
        origin_medium: str,
        hypothesis: str,
        suggested_medium: str,
    ) -> EpistemicVoid:
        for v in self.voids:
            if v.kind == kind and v.target == target and v.status == "OPEN":
                v.priority = max(v.priority, float(priority))
                return v
        vid = f"void_{len(self.voids):03d}"
        void_obj = EpistemicVoid(
            void_id=vid,
            kind=kind,
            target=target,
            priority=float(priority),
            origin_medium=origin_medium,
            hypothesis=hypothesis,
            suggested_medium=suggested_medium,
        )
        self.voids.append(void_obj)
        return void_obj

    def get_open_voids(self, kind: Optional[str] = None) -> List[EpistemicVoid]:
        items = [v for v in self.voids if v.status == "OPEN"]
        if kind is not None:
            items = [v for v in items if v.kind == kind]
        items.sort(key=lambda v: (-v.priority, v.void_id))
        return items

    def resolve_void(self, void_id: str, note: str) -> Optional[EpistemicVoid]:
        for v in self.voids:
            if v.void_id == void_id:
                v.status = "RESOLVED"
                v.resolution_note = note
                return v
        return None

    def resolve_voids_by_target(self, kind: str, target: str, note: str) -> int:
        count = 0
        for v in self.voids:
            if v.kind == kind and v.target == target and v.status == "OPEN":
                v.status = "RESOLVED"
                v.resolution_note = note
                count += 1
        return count

    # ------------------------------------------------------------------
    # Multi-Modal Concept Formation & Concept Mitosis
    # ------------------------------------------------------------------
    def upsert_or_split_concept(
        self,
        domain_signatures: Dict[str, List[float]],
        source_medium: str,
        causal_effects: Optional[Dict[str, Dict[str, float]]] = None,
        optical_match_threshold: float = 0.65,
        surprise_threshold: float = 0.75,
    ) -> Tuple[GroundedConceptNode, bool]:
        """
        Match by primary visual domain without merging contradictory physical domains.
        Triggers Concept Mitosis when appearance matches but mechanics/acoustics diverge.
        """
        self.episodes_count += 1
        opt_vec = np.asarray(domain_signatures.get("optical", []), dtype=np.float64)

        best_node: Optional[GroundedConceptNode] = None
        best_opt_dist = 1e9
        if opt_vec.size > 0:
            for node in self.concepts.values():
                ref_opt = np.asarray(node.domain_signatures.get("optical", []), dtype=np.float64)
                if ref_opt.size == opt_vec.size and ref_opt.size > 0:
                    d = float(np.linalg.norm(opt_vec - ref_opt))
                    if d < best_opt_dist:
                        best_opt_dist = d
                        best_node = node

        if best_node is None or best_opt_dist > optical_match_threshold:
            cid = f"concept_{len(self.concepts):03d}"
            node = GroundedConceptNode(
                concept_id=cid,
                domain_signatures={k: list(map(float, v)) for k, v in domain_signatures.items()},
                provenance_sources=[source_medium],
                confirmation_count=1,
                causal_effects=dict(causal_effects or {}),
            )
            self.concepts[cid] = node
            return node, False

        # Check secondary physical domains for counter-example surprise
        max_sec_dist = 0.0
        divergent_domain = None
        diff_vec_list: List[float] = []
        for dom in ("mechanical", "acoustic", "game_outcome"):
            if dom in domain_signatures and dom in best_node.domain_signatures:
                v_new = np.asarray(domain_signatures[dom], dtype=np.float64)
                v_old = np.asarray(best_node.domain_signatures[dom], dtype=np.float64)
                if v_new.size == v_old.size and v_new.size > 0:
                    d_sec = float(np.linalg.norm(v_new - v_old))
                    if d_sec > max_sec_dist:
                        max_sec_dist = d_sec
                        divergent_domain = dom
                        diff_vec_list = (v_new - v_old).tolist()

        if max_sec_dist > surprise_threshold:
            cid = f"concept_{len(self.concepts):03d}"
            child = GroundedConceptNode(
                concept_id=cid,
                domain_signatures={k: list(map(float, v)) for k, v in domain_signatures.items()},
                provenance_sources=[source_medium],
                confirmation_count=1,
                parent_concept_id=best_node.concept_id,
                stem_signature=list(best_node.domain_signatures.get("optical", [])),
                branch_differential=diff_vec_list,
                causal_effects=dict(causal_effects or {}),
            )
            self.concepts[cid] = child
            self.mitosis_events.append(
                {
                    "kind": "CONCEPT_MITOSIS",
                    "parent_id": best_node.concept_id,
                    "child_id": cid,
                    "divergent_domain": divergent_domain,
                    "surprise_distance": max_sec_dist,
                }
            )
            return child, True

        # Reinforce matching concept without corrupting distinct sub-types
        n = float(best_node.confirmation_count)
        for dom, vals in domain_signatures.items():
            arr_new = np.asarray(vals, dtype=np.float64)
            if dom in best_node.domain_signatures and len(best_node.domain_signatures[dom]) == len(vals):
                arr_old = np.asarray(best_node.domain_signatures[dom], dtype=np.float64)
                best_node.domain_signatures[dom] = ((arr_old * n + arr_new) / (n + 1.0)).tolist()
            else:
                best_node.domain_signatures[dom] = list(map(float, vals))
        best_node.confirmation_count += 1
        if source_medium not in best_node.provenance_sources:
            best_node.provenance_sources.append(source_medium)
        if causal_effects:
            best_node.causal_effects.update(causal_effects)
        return best_node, False

    def match_concept_by_optical(self, optical_vec: List[float], max_dist: float = 0.65) -> Optional[GroundedConceptNode]:
        q = np.asarray(optical_vec, dtype=np.float64)
        best: Optional[GroundedConceptNode] = None
        best_d = max_dist
        for node in self.concepts.values():
            ref = np.asarray(node.domain_signatures.get("optical", []), dtype=np.float64)
            if ref.size == q.size and ref.size > 0:
                d = float(np.linalg.norm(q - ref))
                if d <= best_d:
                    best_d = d
                    best = node
        return best

    # ------------------------------------------------------------------
    # Axis-Operator Grounding & Counterfactual Re-Observation Disambiguation
    # ------------------------------------------------------------------
    def record_symbol_axis_contrast(
        self,
        surface_form: str,
        delta_features: Dict[str, float],
        context_tokens: List[str],
        domain: str = "optical",
        structural_role: Optional[Dict[str, float]] = None,
    ) -> SymbolSenseBranch:
        self.episodes_count += 1
        obs_list = self._axis_observations.setdefault(surface_form, [])
        obs_list.append({k: float(v) for k, v in delta_features.items()})

        all_axes = sorted({k for obs in obs_list for k in obs.keys()})
        snr_by_axis: Dict[str, float] = {}
        mean_by_axis: Dict[str, float] = {}
        for ax in all_axes:
            vals = np.array([obs.get(ax, 0.0) for obs in obs_list], dtype=np.float64)
            mu = float(np.mean(vals))
            sig = float(np.std(vals))
            mean_by_axis[ax] = mu
            snr_by_axis[ax] = abs(mu) / (sig + 0.12)

        ranked = sorted(all_axes, key=lambda a: snr_by_axis[a], reverse=True)
        top_ax = ranked[0] if ranked else "unknown"
        top_snr = snr_by_axis.get(top_ax, 0.0)
        top_dir = 1.0 if mean_by_axis.get(top_ax, 0.0) >= 0.0 else -1.0

        # Detect confounded axes when a second axis co-varies strongly with the top axis
        if len(ranked) >= 2 and len(obs_list) >= 2:
            sec_ax = ranked[1]
            sec_snr = snr_by_axis[sec_ax]
            if top_snr > 0.25 and sec_snr > 0.35 * top_snr and sec_snr > 0.35:
                self.register_void(
                    kind="VOID_AMBIGUOUS_AXIS",
                    target=surface_form,
                    priority=0.95,
                    origin_medium="world",
                    hypothesis=f"Axis confound between '{top_ax}' (snr={top_snr:.2f}) and '{sec_ax}' (snr={sec_snr:.2f}).",
                    suggested_medium="world",
                )

        confidence = float(min(1.0, top_snr / (top_snr + 1.0)))
        binding = AxisOperatorBinding(
            axis_name=top_ax,
            direction=top_dir,
            confidence=confidence,
            invariance_snr=top_snr,
            reobservation_verified=False,
            reobservation_margin=0.0,
            constituent_summary=mean_by_axis,
        )

        sense, _ = self.observe_symbol_in_context(
            surface_form=surface_form,
            context_tokens=context_tokens,
            domain=domain,
            structural_role=structural_role,
            axis_binding=binding,
        )
        sense.axis_binding = binding
        return sense

    def apply_reobservation_disambiguation(
        self,
        surface_form: str,
        probe_results: List[Tuple[Dict[str, float], bool]],
    ) -> AxisOperatorBinding:
        """
        Separate confounded sensory axes using orthogonal counterfactual re-observations.
        `probe_results` holds `(reobserved_delta_dict, is_positive_for_symbol)`.
        """
        pos_obs = [d for d, ok in probe_results if ok]
        neg_obs = [d for d, ok in probe_results if not ok]
        all_axes = sorted({k for d, _ in probe_results for k in d.keys()})

        best_ax = all_axes[0] if all_axes else "unknown"
        best_score = -1e9
        best_dir = 1.0
        best_margin = 0.0
        summary: Dict[str, float] = {}

        for ax in all_axes:
            p_vals = np.array([d.get(ax, 0.0) for d in pos_obs], dtype=np.float64) if pos_obs else np.zeros(1)
            n_vals = np.array([d.get(ax, 0.0) for d in neg_obs], dtype=np.float64) if neg_obs else np.zeros(1)
            mu_p = float(np.mean(p_vals))
            mu_n = float(np.mean(n_vals))
            std_p = float(np.std(p_vals))
            margin = abs(mu_p) - abs(mu_n)
            snr = abs(mu_p) / (std_p + 0.05)
            score = snr * max(0.0, margin)
            summary[ax] = mu_p
            if score > best_score:
                best_score = score
                best_ax = ax
                best_dir = 1.0 if mu_p >= 0.0 else -1.0
                best_margin = float(margin)

        binding = AxisOperatorBinding(
            axis_name=best_ax,
            direction=best_dir,
            confidence=float(min(1.0, 0.5 + 0.5 * (best_margin / (best_margin + 0.15)))) if best_margin > 0 else 0.5,
            invariance_snr=float(best_score),
            reobservation_verified=True,
            reobservation_margin=best_margin,
            constituent_summary=summary,
        )

        senses = self.symbols.get(surface_form, [])
        if senses:
            prev_ax = senses[0].axis_binding.axis_name if senses[0].axis_binding else "none"
            senses[0].axis_binding = binding
            self.criteria_revisions.append(
                {
                    "kind": "AXIS_DISAMBIGUATION",
                    "surface_form": surface_form,
                    "previous_axis": prev_ax,
                    "revised_axis": best_ax,
                    "margin": best_margin,
                }
            )
        self.resolve_voids_by_target(
            "VOID_AMBIGUOUS_AXIS",
            surface_form,
            f"Separated via counterfactual self-re-observation -> {best_ax} ({best_dir:+.0f}, margin={best_margin:.3f}).",
        )
        return binding

    # ------------------------------------------------------------------
    # Contextual Symbol Learning, Lexical Mitosis & Disambiguation
    # ------------------------------------------------------------------
    def observe_symbol_in_context(
        self,
        surface_form: str,
        context_tokens: List[str],
        domain: str,
        structural_role: Optional[Dict[str, float]] = None,
        axis_binding: Optional[AxisOperatorBinding] = None,
        bound_concept_id: Optional[str] = None,
        mitosis_threshold: float = 0.68,
        register_void_if_ungrounded: bool = False,
    ) -> Tuple[SymbolSenseBranch, Optional[Dict[str, Any]]]:
        """
        Learn or split a symbol's sense branch strictly from surrounding context & grounding.
        Excludes `surface_form` from its own context vector so polysemous occurrences in
        disjoint contexts cleanly trigger Contextual Lexical Mitosis.
        """
        filtered_ctx = [t for t in context_tokens if t != surface_form and len(t) > 0]
        ctx_vec: Dict[str, float] = {f"dom:{domain}": 0.65}
        for tok in filtered_ctx:
            ctx_vec[f"tok:{tok}"] = ctx_vec.get(f"tok:{tok}", 0.0) + 1.0
        ctx_vec = _normalize_dict(ctx_vec)

        role_vec = {
            k: float((structural_role or {}).get(k, 0.2)) for k in STRUCTURAL_ROLE_DIMS
        }

        existing = self.symbols.get(surface_form, [])
        if not existing:
            sense = SymbolSenseBranch(
                sense_id=f"{surface_form}#sense_0",
                surface_form=surface_form,
                primary_domain=domain,
                bound_concept_id=bound_concept_id,
                axis_binding=axis_binding,
                context_weights=ctx_vec,
                structural_role_profile=role_vec,
                example_contexts=[filtered_ctx[:12]] if filtered_ctx else [],
                confirmation_count=1,
            )
            self.symbols[surface_form] = [sense]
            if register_void_if_ungrounded:
                self.register_void(
                    kind="VOID_UNGROUNDED_SYMBOL",
                    target=surface_form,
                    priority=0.55,
                    origin_medium="book",
                    hypothesis=f"Symbol '{surface_form}' encountered in discourse without direct sensory grounding.",
                    suggested_medium="world" if domain in ("optical", "mechanical", "acoustic") else "game",
                )
            return sense, None

        # Compare incoming context against all existing senses of `surface_form`
        best_sense: Optional[SymbolSenseBranch] = None
        best_sim = -1.0
        # Compare token-only overlap + domain-weighted overlap
        ctx_tok_only = _normalize_dict({k: v for k, v in ctx_vec.items() if k.startswith("tok:")})
        for sense in existing:
            s_tok_only = _normalize_dict({k: v for k, v in sense.context_weights.items() if k.startswith("tok:")})
            tok_sim = _cosine_dict(ctx_tok_only, s_tok_only)
            dom_bonus = 0.35 if sense.primary_domain == domain else 0.0
            combined = 0.70 * tok_sim + 0.30 * _cosine_dict(ctx_vec, sense.context_weights) + dom_bonus
            combined = min(1.0, combined)
            if combined > best_sim:
                best_sim = combined
                best_sense = sense

        assert best_sense is not None
        surprise = 1.0 - best_sim
        domain_clash = (
            best_sense.primary_domain != domain
            and (best_sense.axis_binding is not None or best_sense.bound_concept_id is not None)
            and _cosine_dict(ctx_tok_only, _normalize_dict({k: v for k, v in best_sense.context_weights.items() if k.startswith("tok:")})) < 0.25
        )

        if surprise > mitosis_threshold or domain_clash:
            new_id = f"{surface_form}#sense_{len(existing)}"
            reason = (
                f"Contextual surprise={surprise:.2f} > {mitosis_threshold:.2f} "
                f"(domain shift {best_sense.primary_domain} -> {domain})"
            )
            new_sense = SymbolSenseBranch(
                sense_id=new_id,
                surface_form=surface_form,
                primary_domain=domain,
                bound_concept_id=bound_concept_id,
                axis_binding=axis_binding,
                context_weights=ctx_vec,
                structural_role_profile=role_vec,
                example_contexts=[filtered_ctx[:12]] if filtered_ctx else [],
                confirmation_count=1,
                parent_sense_id=best_sense.sense_id,
                mitosis_reason=reason,
            )
            existing.append(new_sense)
            evt = {
                "kind": "LEXICAL_MITOSIS",
                "surface_form": surface_form,
                "parent_sense_id": best_sense.sense_id,
                "new_sense_id": new_id,
                "surprise": surprise,
                "new_domain": domain,
            }
            self.mitosis_events.append(evt)
            self.criteria_revisions.append(
                {
                    "kind": "LEXICAL_MITOSIS",
                    "surface_form": surface_form,
                    "from_sense": best_sense.sense_id,
                    "to_sense": new_id,
                    "reason": reason,
                }
            )
            return new_sense, evt

        # Reinforce matched sense cleanly
        n = float(best_sense.confirmation_count)
        merged: Dict[str, float] = dict(best_sense.context_weights)
        for k, v in ctx_vec.items():
            merged[k] = merged.get(k, 0.0) * (n / (n + 1.0)) + v * (1.0 / (n + 1.0))
        best_sense.context_weights = _normalize_dict(merged)

        if structural_role:
            for k in STRUCTURAL_ROLE_DIMS:
                old_r = best_sense.structural_role_profile.get(k, 0.2)
                new_r = float(structural_role.get(k, old_r))
                best_sense.structural_role_profile[k] = float((old_r * n + new_r) / (n + 1.0))

        if filtered_ctx and len(best_sense.example_contexts) < 6:
            best_sense.example_contexts.append(filtered_ctx[:12])
        best_sense.confirmation_count += 1
        if bound_concept_id and best_sense.bound_concept_id is None:
            best_sense.bound_concept_id = bound_concept_id
            self.resolve_voids_by_target(
                "VOID_UNGROUNDED_SYMBOL",
                surface_form,
                f"Grounded to {bound_concept_id} in domain '{domain}'.",
            )
        if axis_binding and best_sense.axis_binding is None:
            best_sense.axis_binding = axis_binding
            self.resolve_voids_by_target(
                "VOID_UNGROUNDED_SYMBOL",
                surface_form,
                f"Bound to perceptual axis '{axis_binding.axis_name}'.",
            )
        return best_sense, None

    def disambiguate_symbol(
        self,
        surface_form: str,
        context_tokens: List[str],
        domain_hint: Optional[str] = None,
    ) -> Tuple[Optional[SymbolSenseBranch], float, Dict[str, float]]:
        """
        Select the sense branch of `surface_form` that resonates with unseen context tokens.
        """
        senses = self.symbols.get(surface_form, [])
        if not senses:
            return None, 0.0, {}

        filtered = [t for t in context_tokens if t != surface_form and len(t) > 0]
        q_tok: Dict[str, float] = {}
        for tok in filtered:
            q_tok[f"tok:{tok}"] = q_tok.get(f"tok:{tok}", 0.0) + 1.0
        if domain_hint is not None:
            q_tok[f"dom:{domain_hint}"] = 0.65
        q_tok = _normalize_dict(q_tok)

        scores: Dict[str, float] = {}
        for sense in senses:
            s_weights = (
                sense.context_weights
                if domain_hint is not None
                else _normalize_dict({k: v for k, v in sense.context_weights.items() if k.startswith("tok:")})
            )
            scores[sense.sense_id] = float(_cosine_dict(q_tok, s_weights))

        ranked = sorted(senses, key=lambda s: scores[s.sense_id], reverse=True)
        best = ranked[0]
        second_score = scores[ranked[1].sense_id] if len(ranked) > 1 else 0.0
        margin = float(scores[best.sense_id] - second_score)
        return best, margin, scores

    def compare_symbol_senses(self, surface_form: str) -> List[SenseComparisonReport]:
        """
        Compare all pairs of senses for `surface_form`, returning Homological Stem
        (what is invariant across uses) vs Disparate Branch (what splits by context).
        """
        senses = self.symbols.get(surface_form, [])
        reports: List[SenseComparisonReport] = []
        for i in range(len(senses)):
            for j in range(i + 1, len(senses)):
                sa, sb = senses[i], senses[j]
                tok_a = _normalize_dict({k: v for k, v in sa.context_weights.items() if k.startswith("tok:")})
                tok_b = _normalize_dict({k: v for k, v in sb.context_weights.items() if k.startswith("tok:")})
                ctx_div = float(1.0 - _cosine_dict(tok_a, tok_b))
                stem_sim = _cosine_role(sa.structural_role_profile, sb.structural_role_profile)

                shared_dims = [
                    d
                    for d in STRUCTURAL_ROLE_DIMS
                    if sa.structural_role_profile.get(d, 0.0) >= 0.45
                    and sb.structural_role_profile.get(d, 0.0) >= 0.45
                    and abs(sa.structural_role_profile.get(d, 0.0) - sb.structural_role_profile.get(d, 0.0)) <= 0.35
                ]
                div_branches = [
                    f"{sa.primary_domain} vs {sb.primary_domain}",
                ]
                top_a = sorted(tok_a.items(), key=lambda x: -x[1])[:2]
                top_b = sorted(tok_b.items(), key=lambda x: -x[1])[:2]
                if top_a or top_b:
                    a_str = ",".join(k.replace("tok:", "") for k, _ in top_a)
                    b_str = ",".join(k.replace("tok:", "") for k, _ in top_b)
                    div_branches.append(f"cues[{a_str}] vs cues[{b_str}]")

                reports.append(
                    SenseComparisonReport(
                        surface_form=surface_form,
                        sense_a_id=sa.sense_id,
                        sense_b_id=sb.sense_id,
                        context_divergence=ctx_div,
                        structural_stem_similarity=stem_sim,
                        shared_stem_dimensions=shared_dims,
                        divergent_branch_dimensions=div_branches,
                        explanation=(
                            f"Stem similarity={stem_sim:.2f} via {shared_dims or ['general_role']}; "
                            f"branch divergence={ctx_div:.2f} across {div_branches}."
                        ),
                    )
                )
        return reports

    def get_axis_binding(self, surface_form: str) -> Optional[AxisOperatorBinding]:
        for s in self.symbols.get(surface_form, []):
            if s.axis_binding is not None:
                return s.axis_binding
        return None

    def record_creation(self, rec: CreationVerificationRecord) -> None:
        self.creations.append(rec)

    def register_synthesized_organ(self, record: SynthesizedOrganRecord) -> None:
        self.synthesized_organs[record.organ_id] = record

    # ------------------------------------------------------------------
    # CausalField Multi-Scale Projection (L6 Concepts .. L14 Rules/Creations)
    # ------------------------------------------------------------------
    def project_to_causal_field(self) -> CausalField:
        field_obj = CausalField(dimensions=3)

        # Scale 6.0: Grounded multi-modal Concepts
        concept_items = list(self.concepts.values())
        for idx, c in enumerate(concept_items):
            opt = c.domain_signatures.get("optical", [0.0, 0.0, 0.0, 0.0])
            mech = c.domain_signatures.get("mechanical", [0.0, 0.0, 0.0])
            vec = np.array(
                [opt[0] if len(opt) > 0 else 0.0, opt[1] if len(opt) > 1 else 0.0,
                 mech[0] if len(mech) > 0 else 0.0, float(c.confirmation_count),
                 1.0 if c.parent_concept_id else 0.0, float(len(c.provenance_sources))],
                dtype=np.float32,
            )
            field_obj.add_voxel(
                InformationVoxel(
                    id=c.concept_id,
                    content={"type": "grounded_concept", "sources": c.provenance_sources},
                    tensor=vec,
                    mass=1.2 + 0.2 * c.confirmation_count,
                    position=np.array([0.1 + 0.12 * (idx % 7), 0.25 + 0.08 * (idx // 7), 0.4], dtype=np.float32),
                    scale_level=6.0,
                )
            )
            if c.parent_concept_id and c.parent_concept_id in field_obj.voxels:
                field_obj.link_voxels(c.parent_concept_id, c.concept_id, strength=1.5)

        # Scale 10.0 & 12.0: Symbol Senses & Mitosis Branches
        sym_idx = 0
        for form, senses in self.symbols.items():
            for k, s in enumerate(senses):
                sym_idx += 1
                scale = 12.0 if s.parent_sense_id else 10.0
                role_vals = [float(s.structural_role_profile.get(d, 0.0)) for d in STRUCTURAL_ROLE_DIMS]
                axis_flag = 1.0 if (s.axis_binding and s.axis_binding.reobservation_verified) else 0.0
                vec = np.array(role_vals + [axis_flag], dtype=np.float32)
                vid = f"sense_{s.sense_id}"
                field_obj.add_voxel(
                    InformationVoxel(
                        id=vid,
                        content={"type": "symbol_sense", "form": form, "domain": s.primary_domain},
                        tensor=vec,
                        mass=1.0 + 0.15 * s.confirmation_count,
                        position=np.array([0.1 + 0.09 * (sym_idx % 9), 0.65 + 0.06 * k, 0.9 if k == 0 else 1.15], dtype=np.float32),
                        scale_level=scale,
                    )
                )
                if s.bound_concept_id and s.bound_concept_id in field_obj.voxels:
                    field_obj.link_voxels(s.bound_concept_id, vid, strength=2.0)
                if s.parent_sense_id:
                    pvid = f"sense_{s.parent_sense_id}"
                    if pvid in field_obj.voxels:
                        field_obj.link_voxels(pvid, vid, strength=1.4)

        # Scale 14.0: Verified Causal Rules & Autonomous Creations
        for r_idx, rule in enumerate(self.causal_rules.values()):
            rvid = f"rule_{rule.rule_id}"
            field_obj.add_voxel(
                InformationVoxel(
                    id=rvid,
                    content={"type": "causal_rule", "verified": rule.verified_in_reality},
                    tensor=np.array(
                        [1.0 if rule.verified_in_reality else 0.0, float(rule.confirmations),
                         float(rule.falsifications), float(len(rule.condition_tokens)),
                         float(len(rule.action_tokens)), float(len(rule.predicted_effect_tokens))],
                        dtype=np.float32,
                    ),
                    mass=2.0 if rule.verified_in_reality else 1.0,
                    position=np.array([0.2 + 0.2 * (r_idx % 4), 0.92, 1.4], dtype=np.float32),
                    scale_level=14.0,
                )
            )
        # Scale 8.0: Self-Synthesized Code Organs & Inverted Tool Kernels
        for o_idx, org in enumerate(self.synthesized_organs.values()):
            ovid = f"organ_{org.organ_id}"
            field_obj.add_voxel(
                InformationVoxel(
                    id=ovid,
                    content={"type": "synthesized_organ", "domain": org.domain, "entries": list(org.entry_functions)},
                    tensor=np.array(
                        [float(org.validation_score), float(org.ast_node_count) / 100.0,
                         float(len(org.falsified_candidates)), float(len(org.discovered_principles)),
                         float(len(org.entry_functions)), 1.0],
                        dtype=np.float32,
                    ),
                    mass=2.4,
                    position=np.array([0.2 + 0.25 * (o_idx % 4), 0.50, 0.72], dtype=np.float32),
                    scale_level=8.0,
                )
            )

        for c_idx, cr in enumerate(self.creations):
            cr_vid = f"creation_{cr.creation_id}"
            field_obj.add_voxel(
                InformationVoxel(
                    id=cr_vid,
                    content={"type": "verified_creation", "kind": cr.kind},
                    tensor=np.array([1.0 if cr.verified_by_own_organs else 0.0, 1.0, 1.0, 0.8, 0.8, 1.0], dtype=np.float32),
                    mass=2.2,
                    position=np.array([0.7 + 0.15 * c_idx, 0.95, 1.5], dtype=np.float32),
                    scale_level=14.0,
                )
            )
            for org in self.synthesized_organs.values():
                if org.domain in ("cad_csg_kernel",) and "CAD" in cr.kind:
                    field_obj.link_voxels(f"organ_{org.organ_id}", cr_vid, strength=1.8)

        if field_obj.voxels:
            field_obj.step(dt=0.05)
        return field_obj

    # ------------------------------------------------------------------
    # JSON Persistence (Cold-Reload Roundtrip)
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episodes_count": self.episodes_count,
            "concepts": {k: asdict(v) for k, v in self.concepts.items()},
            "symbols": {k: [asdict(s) for s in v] for k, v in self.symbols.items()},
            "causal_rules": {k: asdict(v) for k, v in self.causal_rules.items()},
            "voids": [asdict(v) for v in self.voids],
            "mitosis_events": list(self.mitosis_events),
            "criteria_revisions": list(self.criteria_revisions),
            "saturation_laws": dict(self.saturation_laws),
            "creations": [asdict(c) for c in self.creations],
            "synthesized_organs": {k: asdict(v) for k, v in self.synthesized_organs.items()},
            "axis_observations": dict(self._axis_observations),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpistemicKnowledgeSpine":
        spine = cls()
        spine.episodes_count = int(data.get("episodes_count", 0))
        spine.concepts = {
            k: GroundedConceptNode(**v) for k, v in data.get("concepts", {}).items()
        }
        spine.symbols = {}
        for form, slist in data.get("symbols", {}).items():
            rebuilt: List[SymbolSenseBranch] = []
            for sd in slist:
                ab_data = sd.get("axis_binding")
                ab = AxisOperatorBinding(**ab_data) if ab_data is not None else None
                sd_copy = dict(sd)
                sd_copy["axis_binding"] = ab
                rebuilt.append(SymbolSenseBranch(**sd_copy))
            spine.symbols[form] = rebuilt
        spine.causal_rules = {
            k: CausalRuleHypothesis(**v) for k, v in data.get("causal_rules", {}).items()
        }
        spine.voids = [EpistemicVoid(**v) for v in data.get("voids", [])]
        spine.mitosis_events = list(data.get("mitosis_events", []))
        spine.criteria_revisions = list(data.get("criteria_revisions", []))
        spine.saturation_laws = dict(data.get("saturation_laws", {}))
        spine.creations = [CreationVerificationRecord(**c) for c in data.get("creations", [])]
        spine.synthesized_organs = {
            k: SynthesizedOrganRecord(**v) for k, v in data.get("synthesized_organs", {}).items()
        }
        spine._axis_observations = dict(data.get("axis_observations", {}))
        return spine

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "EpistemicKnowledgeSpine":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

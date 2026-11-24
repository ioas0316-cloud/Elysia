"""
Structural Scale-Hierarchy Rotor Cortex
=======================================

Implements non-numeric causal-structural cognition across Physics, Chemistry,
DNA/Biology, Mathematics, Code Architecture, and Language:
  - Causal Principles, Structures, Perspectives, Criteria, Changes, and
    Re-Recognitions are first-class nodes on a shared Scale-Hierarchy Rotor.
  - Anything can observe, classify, discern, and recombine with anything else
    regardless of domain ("물리던 화학이던 DNA던 코드던 언어던 모든 것으로 모든 것을").
  - Linguistic definitions & concepts act as relational *re-recognitions*:
    observing how a target's micro constituents traverse a causal process chain
    into a macro phenomenon through a specific perspective and criterion.
  - When a lens fails to discriminate two structurally distinct realities, it
    revises its own criterion (`REVISES_CRITERION`) from the friction.
  - Real workspace Python files (via AST) and real Korean/English causal prose
    enter the exact same rotor alongside physics/chemistry/DNA structures.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings


# ---------------------------------------------------------------------------
# 1. Universal Causal Verbs & Structural Ontologies (Non-Numeric)
# ---------------------------------------------------------------------------


class CausalVerb(str, Enum):
    """Domain-agnostic causal-topological relation types across all sciences."""

    BINDS_WITH = "binds_with"
    DRIVES = "drives"
    TRANSFORMS_INTO = "transforms_into"
    OPPOSES = "opposes"
    FEEDS_BACK_NEGATIVE = "feeds_back_negative"
    FEEDS_BACK_POSITIVE = "feeds_back_positive"
    BRIDGES_ACROSS = "bridges_across"
    REVISES_CRITERION = "revises_criterion"


_VERB_KO: Dict[CausalVerb, str] = {
    CausalVerb.BINDS_WITH: "결합·응집(질서)",
    CausalVerb.DRIVES: "구동·추진(흐름)",
    CausalVerb.TRANSFORMS_INTO: "변환·발현(과정)",
    CausalVerb.OPPOSES: "척력·억제·경계(대항)",
    CausalVerb.FEEDS_BACK_NEGATIVE: "음성 되먹임·오류 교정·평형",
    CausalVerb.FEEDS_BACK_POSITIVE: "양성 증폭·연쇄 발산",
    CausalVerb.BRIDGES_ACROSS: "매개·촉매·계층 간 연속 교량",
    CausalVerb.REVISES_CRITERION: "기준·관점 자기수정",
}

_VERB_CHROMA: Dict[CausalVerb, str] = {
    CausalVerb.DRIVES: "Red",
    CausalVerb.TRANSFORMS_INTO: "Red",
    CausalVerb.BINDS_WITH: "Blue",
    CausalVerb.OPPOSES: "Blue",
    CausalVerb.FEEDS_BACK_NEGATIVE: "Blue",
    CausalVerb.REVISES_CRITERION: "Blue",
    CausalVerb.FEEDS_BACK_POSITIVE: "Yellow",
    CausalVerb.BRIDGES_ACROSS: "Yellow",
}


@dataclass(frozen=True)
class CausalRelation:
    """One directed causal link inside a micro <-> meso <-> macro process chain."""

    source_role: str
    verb: CausalVerb
    target_role: str
    scale_direction: str = "meso"  # "micro_to_meso" | "meso" | "meso_to_macro" | "macro_to_micro"


@dataclass
class CausalStructure:
    """
    A first-class structural causal principle spanning micro constituents,
    process relations, macro phenomenon, boundary conditions, and observing lens.
    """

    structure_id: str
    domain: str
    scale_tier: str
    micro_constituents: Tuple[str, ...]
    process_chain: Tuple[CausalRelation, ...]
    macro_phenomenon: str
    boundary_conditions: Tuple[str, ...] = ()
    observing_perspective: str = "structural_process"
    observing_criterion: str = "causal_continuity"

    def verb_profile(self) -> Dict[CausalVerb, int]:
        counts: Counter[CausalVerb] = Counter(rel.verb for rel in self.process_chain)
        return dict(counts)

    def scale_profile(self) -> Dict[str, int]:
        counts: Counter[str] = Counter(rel.scale_direction for rel in self.process_chain)
        return dict(counts)

    def chromatic_balance(self) -> Dict[str, float]:
        if not self.process_chain:
            return {"Red": 0.34, "Blue": 0.33, "Yellow": 0.33}
        counts = {"Red": 0.0, "Blue": 0.0, "Yellow": 0.0}
        for rel in self.process_chain:
            counts[_VERB_CHROMA[rel.verb]] += 1.0
        tot = max(1.0, sum(counts.values()))
        return {k: v / tot for k, v in counts.items()}

    def has_stabilizing_feedback(self) -> bool:
        verbs = {r.verb for r in self.process_chain}
        return (CausalVerb.FEEDS_BACK_NEGATIVE in verbs) or (CausalVerb.REVISES_CRITERION in verbs)

    def has_continuity_bridge(self) -> bool:
        return any(r.verb == CausalVerb.BRIDGES_ACROSS for r in self.process_chain)

    def is_runaway_fragile(self) -> bool:
        verbs = {r.verb for r in self.process_chain}
        if CausalVerb.FEEDS_BACK_POSITIVE in verbs and not self.has_stabilizing_feedback():
            return True
        linear_only = bool(verbs <= {CausalVerb.DRIVES, CausalVerb.TRANSFORMS_INTO})
        return linear_only


@dataclass
class RoleCorrespondence:
    """Structural role alignment discovered when one principle observes another."""

    lens_role: str
    target_role: str
    shared_verb: str
    scale_direction: str


@dataclass
class ReRecognition:
    """
    Relational re-recognition of `target` observed through `lens` ("언어적 개념·정의는 재인식이다").
    """

    target_id: str
    lens_id: str
    structural_resonance: float
    role_mappings: List[RoleCorrespondence]
    shared_verbs: List[str]
    lens_blind_spots: List[str]
    target_deficits_vs_lens: List[str]
    causal_definition_ko: str


@dataclass
class RecombinationResult:
    """Result of transplanting a cross-domain causal motif to heal a fragile structure."""

    fragile_id: str
    donor_id: str
    donor_domain: str
    created_structure: CausalStructure
    added_relations: List[CausalRelation]
    resilience_before: bool
    resilience_after: bool
    synthesis_ko: str


@dataclass
class CriterionRevisionReport:
    """Result of expanding a coarse lens criterion after encountering discrimination failure."""

    lens_id: str
    target_a_id: str
    target_b_id: str
    resonance_gap_before: float
    resonance_gap_after: float
    added_verbs: List[str]
    summary_ko: str


# ---------------------------------------------------------------------------
# 2. Structural Scale-Hierarchy Rotor Cortex
# ---------------------------------------------------------------------------


def _multiset_jaccard(c1: Dict[Any, int], c2: Dict[Any, int]) -> float:
    keys = set(c1.keys()) | set(c2.keys())
    if not keys:
        return 1.0
    inter = sum(min(c1.get(k, 0), c2.get(k, 0)) for k in keys)
    union = sum(max(c1.get(k, 0), c2.get(k, 0)) for k in keys)
    return float(inter / max(1, union))


def _criterion_recall(target_counts: Dict[Any, int], lens_counts: Dict[Any, int]) -> float:
    total_lens = sum(lens_counts.values())
    if total_lens == 0:
        return 0.0
    matched = sum(min(target_counts.get(k, 0), cnt) for k, cnt in lens_counts.items())
    return float(matched / total_lens)


class StructuralScaleRotorCortex:
    """
    Non-numeric Scale-Hierarchy Rotor Cortex:
    Observes, classifies, rotates perspectives over, recombines across domains,
    and revises its own observing criteria strictly through causal topology.
    """

    def __init__(self) -> None:
        self.structures: Dict[str, CausalStructure] = {}
        self.recombination_history: List[RecombinationResult] = []

    def register_structure(self, structure: CausalStructure) -> None:
        self.structures[structure.structure_id] = structure

    def observe_through(self, target_id: str, lens_id: str) -> ReRecognition:
        """
        Re-recognizes `target_id` through `lens_id`'s causal topology, perspective,
        and criterion — exposing shared structural roles, lens blind spots, and deficits.
        """
        target = self.structures[target_id]
        lens = self.structures[lens_id]

        t_verbs = target.verb_profile()
        l_verbs = lens.verb_profile()
        t_scales = target.scale_profile()
        l_scales = lens.scale_profile()

        # 1. Pure topological resonance: how much of lens's criteria target satisfies + overall symmetry
        verb_recall = _criterion_recall(t_verbs, l_verbs)
        verb_jacc = _multiset_jaccard(t_verbs, l_verbs)
        scale_recall = _criterion_recall(t_scales, l_scales)

        # Motif resonance evaluated along what `lens` actually inspects
        motif_hits = 0
        motif_checks = 0
        if lens.has_stabilizing_feedback():
            motif_checks += 1
            if target.has_stabilizing_feedback():
                motif_hits += 1
        if lens.has_continuity_bridge():
            motif_checks += 1
            if target.has_continuity_bridge():
                motif_hits += 1
        if CausalVerb.OPPOSES in l_verbs:
            motif_checks += 1
            if CausalVerb.OPPOSES in t_verbs:
                motif_hits += 1
        motif_score = float(motif_hits / motif_checks) if motif_checks > 0 else 1.0

        # Chromatic similarity weighted subtly when lens contains feedback/bind structure
        cb_t = target.chromatic_balance()
        cb_l = lens.chromatic_balance()
        chroma_sim = float(max(0.0, 1.0 - 0.5 * sum(abs(cb_t[c] - cb_l[c]) for c in ("Red", "Blue", "Yellow"))))

        resonance = float(
            0.52 * verb_recall
            + 0.20 * verb_jacc
            + 0.10 * scale_recall
            + 0.13 * motif_score
            + 0.05 * chroma_sim
        )
        resonance = float(max(0.0, min(1.0, resonance)))

        # 2. Explicit structural role correspondences on shared causal verbs
        shared_verb_set = set(t_verbs.keys()) & set(l_verbs.keys())
        role_maps: List[RoleCorrespondence] = []
        seen_pairs: set[Tuple[str, str, str]] = set()
        for lr in lens.process_chain:
            if lr.verb not in shared_verb_set:
                continue
            for tr in target.process_chain:
                if tr.verb != lr.verb:
                    continue
                key = (lr.source_role, tr.source_role, lr.verb.value)
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    role_maps.append(
                        RoleCorrespondence(
                            lens_role=f"{lr.source_role} -> {lr.target_role}",
                            target_role=f"{tr.source_role} -> {tr.target_role}",
                            shared_verb=lr.verb.value,
                            scale_direction=tr.scale_direction,
                        )
                    )
                    break  # primary canonical pairing per lens relation

        shared_verbs = sorted(v.value for v in shared_verb_set)
        lens_blind_spots = sorted(v.value for v in (set(t_verbs.keys()) - set(l_verbs.keys())))
        target_deficits = sorted(v.value for v in (set(l_verbs.keys()) - set(t_verbs.keys())))

        role_note = (
            ", ".join(f"[{m.lens_role} ≈ {m.target_role} ({m.shared_verb})]" for m in role_maps[:3])
            if role_maps
            else "공유 과정 동사 없음"
        )
        blind_note = ", ".join(lens_blind_spots) if lens_blind_spots else "없음"
        def_note = ", ".join(target_deficits) if target_deficits else "없음"

        defn = (
            f"'{target_id}'({target.domain})을 '{lens_id}'({lens.observing_perspective}/{lens.observing_criterion})의 눈으로 "
            f"재인식하면, 미시 구성({', '.join(target.micro_constituents[:3])})이 {role_note} 과정을 거쳐 "
            f"'{target.macro_phenomenon}'으로 나타나는 구조입니다(구조 공명={resonance:.3f}). "
            f"[렌즈 사각지대={blind_note} | 대상 결핍={def_note}]"
        )

        return ReRecognition(
            target_id=target_id,
            lens_id=lens_id,
            structural_resonance=resonance,
            role_mappings=role_maps,
            shared_verbs=shared_verbs,
            lens_blind_spots=lens_blind_spots,
            target_deficits_vs_lens=target_deficits,
            causal_definition_ko=defn,
        )

    def classify_all_by_lens(self, lens_id: str) -> List[ReRecognition]:
        """Treats ANY principle/structure `lens_id` as the axis classifying every other structure."""
        results = [
            self.observe_through(tid, lens_id)
            for tid in self.structures.keys()
            if tid != lens_id
        ]
        results.sort(key=lambda r: (-r.structural_resonance, r.target_id))
        return results

    def rotate_perspectives(
        self,
        target_id: str,
        lens_ids: Sequence[str],
    ) -> Dict[str, Any]:
        """Rotates multiple domain/perspective lenses over ONE target to show non-reductive definitions."""
        recs = [self.observe_through(target_id, lid) for lid in lens_ids]
        target = self.structures[target_id]
        distinct_sets = [tuple(r.shared_verbs) for r in recs]
        covered = sorted({v for r in recs for v in r.shared_verbs})
        fragments = [f"({r.lens_id} 관점: 공명={r.structural_resonance:.2f}, 공유={r.shared_verbs})" for r in recs]
        integrated = (
            f"대상 '{target_id}'의 '{target.macro_phenomenon}'은 {len(recs)}개 관점 회전으로 "
            f"통합 포착됨({'; '.join(fragments)}) -> 복합 인과 축={covered}."
        )
        return {
            "target_id": target_id,
            "recognitions": recs,
            "distinct_shared_verb_sets": distinct_sets,
            "covered_verbs": covered,
            "integrated_definition_ko": integrated,
        }

    def diagnose_and_recombine(
        self,
        fragile_id: str,
        desired_motif: Optional[CausalVerb] = None,
        cross_domain_only: bool = True,
    ) -> RecombinationResult:
        """
        Diagnoses why `fragile_id` is brittle or runaway, finds the structurally most
        compatible cross-domain donor carrying the missing causal motif (e.g. DNA proofreading
        or Chemistry equilibrium healing brittle Code or escalating Language conflict),
        and synthesizes a newly stabilized hybrid structure onto the rotor.
        """
        fragile = self.structures[fragile_id]
        res_before = fragile.has_stabilizing_feedback() and not fragile.is_runaway_fragile()

        motif = desired_motif
        if motif is None:
            if not fragile.has_stabilizing_feedback() or fragile.is_runaway_fragile():
                motif = CausalVerb.FEEDS_BACK_NEGATIVE
            else:
                motif = CausalVerb.BRIDGES_ACROSS

        # Select best donor carrying `motif` (strictly cross-domain when requested)
        candidates: List[Tuple[float, CausalStructure]] = []
        for sid, st in self.structures.items():
            if sid == fragile_id:
                continue
            if cross_domain_only and st.domain == fragile.domain:
                continue
            if motif not in st.verb_profile():
                continue
            # Prefer resilient donors whose structural scale alignment complements fragile
            score = 1.0 + (0.35 if st.has_stabilizing_feedback() else 0.0) + (0.20 if st.has_continuity_bridge() else 0.0)
            score -= 0.1 * len(st.target_id if hasattr(st, "target_id") else [])
            # Deterministic structural affinity with fragile's scale directions
            shared_scales = len(set(fragile.scale_profile().keys()) & set(st.scale_profile().keys()))
            score += 0.15 * shared_scales
            candidates.append((score, st))

        if not candidates:
            raise ValueError(f"No suitable donor found carrying motif {motif.value} for {fragile_id}")
        candidates.sort(key=lambda item: (-item[0], item[1].structure_id))
        donor = candidates[0][1]

        # Translate donor's stabilizing/opposing relations onto fragile's own active roles
        fragile_head = fragile.micro_constituents[0] if fragile.micro_constituents else "input_state"
        fragile_tail = fragile.process_chain[-1].target_role if fragile.process_chain else fragile.macro_phenomenon

        added: List[CausalRelation] = []
        for rel in donor.process_chain:
            if rel.verb in (motif, CausalVerb.OPPOSES, CausalVerb.BRIDGES_ACROSS):
                mapped_src = f"{fragile_tail}_via_{rel.source_role}"
                mapped_dst = f"{fragile_head}_guarded_by_{rel.target_role}"
                added.append(
                    CausalRelation(
                        source_role=mapped_src,
                        verb=rel.verb,
                        target_role=mapped_dst,
                        scale_direction="macro_to_micro" if rel.verb == CausalVerb.FEEDS_BACK_NEGATIVE else "meso",
                    )
                )
        if not any(r.verb == motif for r in added):
            added.append(
                CausalRelation(
                    source_role=f"{fragile_tail}_anomaly",
                    verb=motif,
                    target_role=f"{fragile_head}_stabilized",
                    scale_direction="macro_to_micro",
                )
            )

        new_id = f"{fragile_id}__recombined_with__{donor.structure_id}"
        created = CausalStructure(
            structure_id=new_id,
            domain=f"{fragile.domain}+{donor.domain}",
            scale_tier=fragile.scale_tier,
            micro_constituents=tuple(dict.fromkeys([*fragile.micro_constituents, f"transplanted::{donor.structure_id}"])),
            process_chain=tuple([*fragile.process_chain, *added]),
            macro_phenomenon=f"{fragile.macro_phenomenon} (재조합 평형·치유됨)",
            boundary_conditions=tuple(
                dict.fromkeys([*fragile.boundary_conditions, *donor.boundary_conditions[:2], "cross_domain_stabilized"])
            ),
            observing_perspective=f"{fragile.observing_perspective}+{donor.observing_perspective}",
            observing_criterion="structural_resilience",
        )
        self.register_structure(created)
        res_after = created.has_stabilizing_feedback() and not created.is_runaway_fragile()

        synth = (
            f"'{fragile_id}'({fragile.domain})의 취약 결핍({motif.value} 부재)을 진단하고, "
            f"이종 영역 '{donor.structure_id}'({donor.domain})의 '{_VERB_KO[motif]}' 인과 원리를 이식·재조합해 "
            f"새 구조 '{new_id}'를 창발했습니다(내성 확보: {res_before} -> {res_after})."
        )
        res = RecombinationResult(
            fragile_id=fragile_id,
            donor_id=donor.structure_id,
            donor_domain=donor.domain,
            created_structure=created,
            added_relations=added,
            resilience_before=res_before,
            resilience_after=res_after,
            synthesis_ko=synth,
        )
        self.recombination_history.append(res)
        return res

    def revise_criterion_from_friction(
        self,
        lens_id: str,
        target_a_id: str,
        target_b_id: str,
    ) -> CriterionRevisionReport:
        """
        Criteria Self-Revision (AGENTS.md §6):
        When `lens_id` fails to separate `target_a_id` from `target_b_id`, expands
        its own observing criteria with the exact discriminating causal verb(s)
        present in one target and absent in the other.
        """
        lens = self.structures[lens_id]
        ta = self.structures[target_a_id]
        tb = self.structures[target_b_id]

        before_a = self.observe_through(target_a_id, lens_id).structural_resonance
        before_b = self.observe_through(target_b_id, lens_id).structural_resonance
        gap_before = abs(before_a - before_b)

        va = set(ta.verb_profile().keys())
        vb = set(tb.verb_profile().keys())
        vl = set(lens.verb_profile().keys())

        # Verbs present in ONE target but NOT the other, missing from lens
        discriminating = sorted(list((va ^ vb) - vl), key=lambda v: v.value)
        # Prioritize stabilizing feedback / opposes if available for strong causal separation
        discriminating.sort(
            key=lambda v: (
                0 if v == CausalVerb.FEEDS_BACK_NEGATIVE else (1 if v == CausalVerb.OPPOSES else 2),
                v.value,
            )
        )

        new_rels = list(lens.process_chain)
        added_names: List[str] = []
        for dv in discriminating[:2]:
            new_rels.append(
                CausalRelation(
                    source_role="friction_anomaly",
                    verb=dv,
                    target_role="discriminated_principle",
                    scale_direction="macro_to_micro" if dv == CausalVerb.FEEDS_BACK_NEGATIVE else "meso",
                )
            )
            added_names.append(dv.value)

        new_rels.append(
            CausalRelation(
                source_role="discrimination_gap",
                verb=CausalVerb.REVISES_CRITERION,
                target_role="expanded_criterion",
                scale_direction="macro_to_micro",
            )
        )
        added_names.append(CausalVerb.REVISES_CRITERION.value)

        revised = CausalStructure(
            structure_id=lens.structure_id,
            domain=lens.domain,
            scale_tier=lens.scale_tier,
            micro_constituents=tuple(dict.fromkeys([*lens.micro_constituents, *added_names])),
            process_chain=tuple(new_rels),
            macro_phenomenon=lens.macro_phenomenon,
            boundary_conditions=lens.boundary_conditions,
            observing_perspective=f"{lens.observing_perspective}_revised",
            observing_criterion=f"{lens.observing_criterion}+{'+'.join(added_names)}",
        )
        self.register_structure(revised)

        after_a = self.observe_through(target_a_id, lens_id).structural_resonance
        after_b = self.observe_through(target_b_id, lens_id).structural_resonance
        gap_after = abs(after_a - after_b)

        summary = (
            f"기준 '{lens_id}'가 '{target_a_id}'와 '{target_b_id}'를 분별하지 못하던 마찰"
            f"(격차 {gap_before:.3f})을 자각하고, 변별 동사({', '.join(added_names)})를 자기 기준에 "
            f"편입시켜 분별력을 {gap_before:.3f} -> {gap_after:.3f}로 자기수정했습니다."
        )
        return CriterionRevisionReport(
            lens_id=lens_id,
            target_a_id=target_a_id,
            target_b_id=target_b_id,
            resonance_gap_before=gap_before,
            resonance_gap_after=gap_after,
            added_verbs=added_names,
            summary_ko=summary,
        )


# ---------------------------------------------------------------------------
# 3. Real Workspace Code AST & Real Prose Causal Extractors (Zero SVD!)
# ---------------------------------------------------------------------------


def extract_structure_from_python_file(
    path: Path | str,
    structure_id: Optional[str] = None,
) -> CausalStructure:
    """
    Reads an actual Python module's AST and lifts its control, boundary-guard,
    error-recovery feedback, data transformation, and cross-module imports directly
    into a `CausalStructure` on the Scale-Hierarchy Rotor.
    """
    p = Path(path)
    src = p.read_text(encoding="utf-8", errors="replace")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(src, filename=str(p))

    sid = structure_id or f"code_ast::{p.stem}"
    defs = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    constituents = tuple(dict.fromkeys(defs[:6])) or (p.stem,)

    has_try = any(isinstance(n, ast.Try) for n in ast.walk(tree))
    has_guard = any(isinstance(n, (ast.If, ast.Assert, ast.Raise)) for n in ast.walk(tree))
    has_loop = any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(tree))
    has_import = any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))
    has_return = any(isinstance(n, ast.Return) for n in ast.walk(tree))
    has_class_bind = any(isinstance(n, ast.ClassDef) for n in ast.walk(tree))

    relations: List[CausalRelation] = []
    if has_import:
        relations.append(CausalRelation("external_modules", CausalVerb.BRIDGES_ACROSS, p.stem, "macro_to_micro"))
    if has_class_bind:
        relations.append(CausalRelation("methods_state", CausalVerb.BINDS_WITH, "object_cohesion", "micro_to_meso"))
    if has_loop:
        relations.append(CausalRelation("input_sequence", CausalVerb.DRIVES, "iterative_flow", "meso"))
    else:
        relations.append(CausalRelation("input_args", CausalVerb.DRIVES, "execution_step", "meso"))
    if has_guard:
        relations.append(CausalRelation("boundary_check", CausalVerb.OPPOSES, "invalid_state", "meso"))
    if has_try:
        relations.append(CausalRelation("exception_signal", CausalVerb.FEEDS_BACK_NEGATIVE, "safe_recovery", "macro_to_micro"))
    if has_return:
        relations.append(CausalRelation("process_state", CausalVerb.TRANSFORMS_INTO, "returned_artifact", "meso_to_macro"))

    doc = ast.get_docstring(tree)
    macro = doc.strip().splitlines()[0][:80] if doc else f"실물 모듈 {p.name}의 실행 구조"

    boundaries = []
    if has_guard:
        boundaries.append("ast_conditional_guard")
    if has_try:
        boundaries.append("ast_exception_feedback")
    if not boundaries:
        boundaries.append("unprotected_direct_flow")

    return CausalStructure(
        structure_id=sid,
        domain="code",
        scale_tier="meso",
        micro_constituents=constituents,
        process_chain=tuple(relations),
        macro_phenomenon=macro,
        boundary_conditions=tuple(boundaries),
        observing_perspective="ast_execution_topology",
        observing_criterion="contract_and_recovery_integrity",
    )


_PROSE_RULES: Tuple[Tuple[re.Pattern[str], CausalVerb, str], ...] = (
    (re.compile(r"(결합|응집|묶|짝지|bind|bond|cohere)", re.IGNORECASE), CausalVerb.BINDS_WITH, "micro_to_meso"),
    (re.compile(r"(매개|촉매|연결|잇|다리|bridge|cataly|mediate)", re.IGNORECASE), CausalVerb.BRIDGES_ACROSS, "meso"),
    (re.compile(r"(교정|되먹임|평형|복구|완충|안정|correct|feedback|buffer|recover|heal)", re.IGNORECASE), CausalVerb.FEEDS_BACK_NEGATIVE, "macro_to_micro"),
    (re.compile(r"(억제|저항|경계|차단|막|반발|oppose|resist|block|boundary|guard)", re.IGNORECASE), CausalVerb.OPPOSES, "meso"),
    (re.compile(r"(증폭|폭주|과열|발산|배가|amplify|runaway|escalate)", re.IGNORECASE), CausalVerb.FEEDS_BACK_POSITIVE, "meso_to_macro"),
    (re.compile(r"(변환|발현|생성|이루|전환|transform|emerge|express|produce)", re.IGNORECASE), CausalVerb.TRANSFORMS_INTO, "meso_to_macro"),
    (re.compile(r"(이끌|구동|밀|유발|흐르|drive|push|cause|induce|flow)", re.IGNORECASE), CausalVerb.DRIVES, "micro_to_meso"),
)


def extract_structure_from_prose(
    structure_id: str,
    domain: str,
    prose: str,
) -> CausalStructure:
    """
    Parses Korean/English explanatory causal prose directly into a `CausalStructure`
    showing how constituents pass through causal verbs into a macro phenomenon.
    """
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", prose) if s.strip()]
    relations: List[CausalRelation] = []
    concepts: List[str] = []

    for idx, sent in enumerate(sentences):
        tokens = re.findall(r"[가-힣A-Za-z0-9_]{2,}", sent)
        src = tokens[0] if tokens else f"cause_{idx}"
        dst = tokens[-1] if len(tokens) > 1 else f"effect_{idx}"
        concepts.extend(tokens[:2])

        matched = False
        for pat, verb, scale_dir in _PROSE_RULES:
            if pat.search(sent):
                relations.append(CausalRelation(src, verb, dst, scale_dir))
                matched = True
        if not matched and len(tokens) >= 2:
            relations.append(CausalRelation(src, CausalVerb.TRANSFORMS_INTO, dst, "meso"))

    if not relations:
        relations.append(CausalRelation("premise", CausalVerb.DRIVES, "conclusion", "meso"))

    constituents = tuple(dict.fromkeys(concepts[:5])) or ("concept_seed",)
    macro = sentences[-1] if sentences else prose[:60]

    return CausalStructure(
        structure_id=structure_id,
        domain=domain,
        scale_tier="macro",
        micro_constituents=constituents,
        process_chain=tuple(relations),
        macro_phenomenon=macro,
        boundary_conditions=("prose_causal_discourse",),
        observing_perspective="explanatory_narrative_lens",
        observing_criterion="process_to_phenomenon_coherence",
    )


# ---------------------------------------------------------------------------
# 4. Cross-Domain Reference Atlas (Physics, Chemistry, DNA, Math, Code, Language)
# ---------------------------------------------------------------------------


def build_cross_domain_reference_atlas() -> StructuralScaleRotorCortex:
    """Constructs a cross-domain cortex spanning Physics, Chemistry, DNA, Math, Code, and Language."""
    cortex = StructuralScaleRotorCortex()

    atlas = [
        # --- DNA / Biology ---
        CausalStructure(
            structure_id="dna::proofreading",
            domain="biology_dna",
            scale_tier="micro_meso",
            micro_constituents=("base_pair_A_T_C_G", "polymerase_exonuclease", "hydrogen_bond"),
            process_chain=(
                CausalRelation("template_strand", CausalVerb.BINDS_WITH, "complementary_base", "micro_to_meso"),
                CausalRelation("polymerase_sweep", CausalVerb.DRIVES, "strand_elongation", "meso"),
                CausalRelation("steric_mismatch", CausalVerb.OPPOSES, "wrong_base_entry", "micro_to_meso"),
                CausalRelation("exonuclease_3_to_5", CausalVerb.FEEDS_BACK_NEGATIVE, "excise_and_replace_error", "macro_to_micro"),
                CausalRelation("repaired_duplex", CausalVerb.TRANSFORMS_INTO, "faithful_genetic_heredity", "meso_to_macro"),
            ),
            macro_phenomenon="오류를 스스로 되짚어 교정하고 유전 연속성을 보존하는 생명의 복제",
            boundary_conditions=("base_pairing_geometry", "exonuclease_backtrack_window"),
            observing_perspective="error_correcting_replication",
            observing_criterion="fidelity_via_negative_feedback",
        ),
        # --- Chemistry ---
        CausalStructure(
            structure_id="chemistry::water_hydrogen_network",
            domain="chemistry",
            scale_tier="micro_macro",
            micro_constituents=("O_H_polar_bond", "partial_charge_dipole", "intermolecular_H_bridge"),
            process_chain=(
                CausalRelation("electronegative_oxygen", CausalVerb.BINDS_WITH, "hydrogen_donor", "micro_to_meso"),
                CausalRelation("thermal_agitation", CausalVerb.OPPOSES, "rigid_crystal_locking", "meso"),
                CausalRelation("tetrahedral_H_bonds", CausalVerb.BRIDGES_ACROSS, "neighboring_water_clusters", "meso"),
                CausalRelation("cooperative_network", CausalVerb.TRANSFORMS_INTO, "liquid_surface_tension_and_cohesion", "meso_to_macro"),
            ),
            macro_phenomenon="수소결합 다리로 흐르면서도 경계와 표면장력을 세우는 액체 물의 응집",
            boundary_conditions=("dipole_polarity", "ambient_thermal_energy"),
            observing_perspective="molecular_cohesion_balance",
            observing_criterion="hydrogen_bridge_and_tension",
        ),
        CausalStructure(
            structure_id="chemistry::buffer_equilibrium",
            domain="chemistry",
            scale_tier="meso",
            micro_constituents=("weak_acid_HA", "conjugate_base_A_minus", "H_plus_ion"),
            process_chain=(
                CausalRelation("weak_acid", CausalVerb.BINDS_WITH, "excess_base", "micro_to_meso"),
                CausalRelation("acid_base_perturbation", CausalVerb.DRIVES, "equilibrium_displacement", "meso"),
                CausalRelation("conjugate_pair", CausalVerb.OPPOSES, "sudden_pH_spike", "meso"),
                CausalRelation("le_chatelier_shift", CausalVerb.FEEDS_BACK_NEGATIVE, "restored_pH_equilibrium", "macro_to_micro"),
                CausalRelation("proton_exchange", CausalVerb.BRIDGES_ACROSS, "solution_stability", "meso_to_macro"),
            ),
            macro_phenomenon="외부 산·염기 충격에도 pH 평형을 복원하는 완충 계의 항상성",
            boundary_conditions=("pKa_window", "conjugate_reserve"),
            observing_perspective="le_chatelier_homeostasis",
            observing_criterion="disturbance_absorbing_equilibrium",
        ),
        CausalStructure(
            structure_id="chemistry::catalysis",
            domain="chemistry",
            scale_tier="meso",
            micro_constituents=("reactants", "catalyst_active_site", "transition_state"),
            process_chain=(
                CausalRelation("substrate", CausalVerb.BINDS_WITH, "catalyst_site", "micro_to_meso"),
                CausalRelation("active_site_geometry", CausalVerb.BRIDGES_ACROSS, "lower_activation_barrier", "meso"),
                CausalRelation("reactive_flux", CausalVerb.DRIVES, "product_formation", "meso"),
                CausalRelation("catalyst_release", CausalVerb.TRANSFORMS_INTO, "macro_reaction_acceleration", "meso_to_macro"),
            ),
            macro_phenomenon="자기 자신은 소모되지 않으면서 두 상태 사이의 에너지 장벽을 낮춰 잇는 촉매",
            boundary_conditions=("activation_barrier", "site_specificity"),
            observing_perspective="activation_barrier_mediation",
            observing_criterion="non_consumed_bridging",
        ),
        # --- Physics ---
        CausalStructure(
            structure_id="physics::harmonic_oscillator",
            domain="physics",
            scale_tier="meso_macro",
            micro_constituents=("inertia_mass", "displacement_x", "restoring_force_kx"),
            process_chain=(
                CausalRelation("momentum", CausalVerb.DRIVES, "displacement_from_center", "meso"),
                CausalRelation("restoring_tension", CausalVerb.OPPOSES, "runaway_departure", "meso"),
                CausalRelation("negative_force_law", CausalVerb.FEEDS_BACK_NEGATIVE, "return_toward_equilibrium", "macro_to_micro"),
                CausalRelation("phase_exchange", CausalVerb.TRANSFORMS_INTO, "continuous_oscillatory_rotor", "meso_to_macro"),
            ),
            macro_phenomenon="관성의 추진과 복원력의 음성 되먹임이 짝지어 영속하는 위상 진동",
            boundary_conditions=("finite_stiffness", "energy_conservation"),
            observing_perspective="dynamical_restoration_mechanics",
            observing_criterion="restoring_feedback_vs_inertia",
        ),
        # --- Mathematics ---
        CausalStructure(
            structure_id="math::euler_rotation",
            domain="mathematics",
            scale_tier="macro",
            micro_constituents=("imaginary_generator_i", "phase_angle_theta", "unit_circle"),
            process_chain=(
                CausalRelation("orthogonal_axis_i", CausalVerb.OPPOSES, "linear_blowup", "meso"),
                CausalRelation("phase_increment", CausalVerb.DRIVES, "continuous_circular_rotation", "meso"),
                CausalRelation("cosine_and_sine", CausalVerb.BRIDGES_ACROSS, "real_and_imaginary_axes", "meso_to_macro"),
                CausalRelation("unit_modulus", CausalVerb.BINDS_WITH, "invariant_magnitude_1", "macro_to_micro"),
            ),
            macro_phenomenon="발산하지 않고 직교 축 사이를 끊김 없이 이어 순환하는 위상 불변 회전",
            boundary_conditions=("unit_norm_preservation",),
            observing_perspective="invariant_phase_geometry",
            observing_criterion="orthogonal_conservation",
        ),
        # --- Code ---
        CausalStructure(
            structure_id="code::transactional_rollback",
            domain="code",
            scale_tier="meso",
            micro_constituents=("operation_step", "invariant_validator", "state_checkpoint"),
            process_chain=(
                CausalRelation("transaction_log", CausalVerb.BINDS_WITH, "atomic_unit", "micro_to_meso"),
                CausalRelation("command_dispatch", CausalVerb.DRIVES, "state_mutation", "meso"),
                CausalRelation("schema_assertion", CausalVerb.OPPOSES, "corrupt_commit", "meso"),
                CausalRelation("rollback_handler", CausalVerb.FEEDS_BACK_NEGATIVE, "restore_consistent_state", "macro_to_micro"),
                CausalRelation("commit_boundary", CausalVerb.BRIDGES_ACROSS, "safe_durable_execution", "meso_to_macro"),
            ),
            macro_phenomenon="이상 발생 시 스스로 되감아 정합성을 보전하는 트랜잭션 실행 구조",
            boundary_conditions=("atomicity", "invariant_check"),
            observing_perspective="fault_tolerant_execution",
            observing_criterion="invariant_and_rollback",
        ),
        CausalStructure(
            structure_id="code::fragile_linear_pipeline",
            domain="code",
            scale_tier="meso",
            micro_constituents=("raw_input", "hardcoded_if_chain", "blind_write"),
            process_chain=(
                CausalRelation("raw_input", CausalVerb.DRIVES, "blind_mutation", "meso"),
                CausalRelation("blind_mutation", CausalVerb.TRANSFORMS_INTO, "unverified_output_or_crash", "meso_to_macro"),
            ),
            macro_phenomenon="오류 경계와 되먹임 없이 직진하다 작은 어긋남에도 부러지는 취성 일방 파이프라인",
            boundary_conditions=("no_error_boundary",),
            observing_perspective="one_way_imperative_flow",
            observing_criterion="throughput_only",
        ),
        # --- Language / Discourse ---
        CausalStructure(
            structure_id="language::metaphor_bridge",
            domain="language",
            scale_tier="macro",
            micro_constituents=("source_domain_image", "shared_causal_relation", "target_concept"),
            process_chain=(
                CausalRelation("concrete_experience", CausalVerb.BRIDGES_ACROSS, "unseen_abstract_domain", "micro_to_meso"),
                CausalRelation("tension_between_domains", CausalVerb.OPPOSES, "literal_collapse", "meso"),
                CausalRelation("shared_relational_pattern", CausalVerb.BINDS_WITH, "re_recognized_meaning", "meso_to_macro"),
                CausalRelation("expanded_horizon", CausalVerb.TRANSFORMS_INTO, "fresh_conceptual_definition", "meso_to_macro"),
            ),
            macro_phenomenon="서로 다른 두 영역을 구조적 공명으로 잇고 새 개념을 재인식시키는 언어의 은유 교량",
            boundary_conditions=("preserve_difference_while_bridging",),
            observing_perspective="relational_meaning_re_recognition",
            observing_criterion="structural_analogy",
        ),
        CausalStructure(
            structure_id="language::escalating_conflict",
            domain="language",
            scale_tier="macro",
            micro_constituents=("defensive_claim", "hostile_counter_claim", "amplified_assumption"),
            process_chain=(
                CausalRelation("unverified_assumption", CausalVerb.DRIVES, "accusation", "meso"),
                CausalRelation("accusation", CausalVerb.FEEDS_BACK_POSITIVE, "escalating_polarization", "meso_to_macro"),
                CausalRelation("polarization", CausalVerb.TRANSFORMS_INTO, "communication_rupture", "meso_to_macro"),
            ),
            macro_phenomenon="상대의 반론을 되먹임 교정 대신 양성 증폭으로 받아 결국 단절로 폭주하는 대화",
            boundary_conditions=("zero_listening_boundary",),
            observing_perspective="unchecked_positive_feedback_discourse",
            observing_criterion="emotional_amplification",
        ),
    ]
    for st in atlas:
        cortex.register_structure(st)
    return cortex

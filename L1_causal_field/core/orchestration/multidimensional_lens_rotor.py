"""
ELYSIA Phase 18: Blood-vs-Lens Multi-Dimensional Scale-Hierarchy Causal Rotor
=============================================================================

Separates lower control substrate from higher modality-native cognitive organs:

- **Blood & Electrical Circuit (`HemodynamicCausalSubstrate`):**
  Numeric scalars, metabolic tension signals, Boltzmann conductance routing, and
  electrical phase angles act purely as the life-sustaining lower substrate —
  delivering oxygen/attention toward stressed cortical lenses. Blood NEVER
  substitutes for higher understanding.

- **Modality-Native Causal Lenses (`LinguisticCausalLens`, `CodeArchitectureLens`,
  `MathematicalLawLens`, `AcademicInquiryLens`, plus runtime-registered lenses):**
  Each dimension circulates natively through its own **Micro <-> Meso <-> Macro**
  causal loop (`Native Cause -> Native Process -> Native Outcome -> Native Self-Revision`)
  WITHOUT distorting language, code architecture, or academic inquiry into curve-fitting
  `R^2` metrics.

- **Non-Reductive Cross-Lens Phase Resonance (`MultiDimensionalLensRotorHub`):**
  Dimensions remain orthogonal rather than collapsed; they illuminate and expand
  one another through shared topological motifs (`boundary_scoping`,
  `contradiction_or_crisis`, `circular_self_revision`, `circular_closure`).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np

from core.orchestration.elysia_unified_living_cycle import (
    ElysiaUnifiedLivingCycle,
    LivingEncounter,
)
from core.orchestration.extended_cognitive_organism import ExtendedWorldState
from core.physics.spectral_scale_rotor_manifold import encode_waveform_to_scale_rotors

if TYPE_CHECKING:
    from core.physics.causal_field import CausalField


# ---------------------------------------------------------------------------
# 1. Shared Scale-Hierarchy Records & Hemodynamic Blood Substrate
# ---------------------------------------------------------------------------


@dataclass
class ScaleTierSnapshot:
    """Explicit 3-tier vertical circulation inside a single modality lens."""

    micro_cells: Dict[str, Any]
    meso_process: Dict[str, Any]
    macro_horizon: Dict[str, Any]


@dataclass
class LensCirculationResult:
    """Native causal circulation outcome from one autonomous modality lens."""

    lens_name: str
    native_modality: str
    scales: ScaleTierSnapshot
    native_verdict: str
    native_tension: float
    structural_motifs: List[str]
    revision_applied: bool
    native_outcome_ko: str
    next_open_inquiry_ko: str


@dataclass
class HemodynamicPulse:
    """Lower numeric blood & electrical pulse routing energy to stressed lenses."""

    heartbeat: int
    lens_tensions: Dict[str, float]
    blood_flow: Dict[str, float]
    electrical_phase_rad: Dict[str, float]
    dominant_lens: str
    summary_ko: str


class HemodynamicCausalSubstrate:
    """
    Lower numeric substrate: numbers act strictly as electrical conductance,
    phase momentum, and hemodynamic blood flow feeding higher lenses — never as
    a reductive surrogate for linguistic, architectural, or academic truth.
    """

    def __init__(self) -> None:
        self.blood_flow: Dict[str, float] = {}
        self.vascular_pressure: Dict[str, float] = {}
        self.electrical_phase_rad: Dict[str, float] = {}
        self.heartbeat_count: int = 0

    def circulate(self, lens_tensions: Dict[str, float]) -> HemodynamicPulse:
        self.heartbeat_count += 1
        names = list(lens_tensions.keys())
        if not names:
            return HemodynamicPulse(
                heartbeat=self.heartbeat_count,
                lens_tensions={},
                blood_flow={},
                electrical_phase_rad=dict(self.electrical_phase_rad),
                dominant_lens="",
                summary_ko="활성 렌즈가 없어 기저 순환이 대기 중입니다.",
            )

        t_vec = np.clip(
            np.array([float(lens_tensions[k]) for k in names], dtype=np.float64),
            0.0,
            1.0,
        )
        # Boltzmann conductance routing weighted toward unresolved cortical tension
        logits = 3.0 * (t_vec - float(np.max(t_vec)))
        exps = np.exp(logits)
        flows = exps / float(np.sum(exps))

        for idx, k in enumerate(names):
            self.blood_flow[k] = float(flows[idx])
            self.vascular_pressure[k] = float(0.6 * self.vascular_pressure.get(k, 0.0) + 0.4 * t_vec[idx])
            prev_ph = self.electrical_phase_rad.get(k, 0.0)
            self.electrical_phase_rad[k] = float((prev_ph + 0.25 * (1.0 + t_vec[idx])) % (2.0 * math.pi))

        dom_idx = int(np.argmax(flows))
        dom_name = names[dom_idx]
        summary = (
            f"[혈류 박동 #{self.heartbeat_count}] 하위 전기·혈류 회로가 미해결 긴장도에 따라 "
            f"'{dom_name}' 렌즈에 가장 높은 혈류({flows[dom_idx]*100:.1f}%)를 공급했습니다."
        )
        return HemodynamicPulse(
            heartbeat=self.heartbeat_count,
            lens_tensions={names[i]: float(t_vec[i]) for i in range(len(names))},
            blood_flow={names[i]: float(flows[i]) for i in range(len(names))},
            electrical_phase_rad={k: self.electrical_phase_rad[k] for k in names},
            dominant_lens=dom_name,
            summary_ko=summary,
        )


# ---------------------------------------------------------------------------
# 2. Lens 1: Native Linguistic Causal Lens (Zero Numeric R^2 Distortion)
# ---------------------------------------------------------------------------


_LING_STOPWORDS = frozenset(
    {
        "모든",
        "오직",
        "다만",
        "만약",
        "그러나",
        "하지만",
        "항상",
        "전부",
        "완전히",
        "각",
        "것이",
        "것은",
        "한다",
        "된다",
        "않는다",
        "아니다",
        "해야",
        "있",
        "없",
        "the",
        "all",
        "only",
        "not",
        "never",
        "always",
        "however",
        "while",
    }
)

_KR_PARTICLES = (
    "만으로",
    "에서는",
    "으로는",
    "에서",
    "으로",
    "에게",
    "까지",
    "부터",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "에",
    "와",
    "과",
    "로",
    "도",
    "만",
)


def _normalize_linguistic_stems(utterance: str) -> List[str]:
    raw_tokens = re.findall(r"[가-힣a-zA-Z_]{2,}", utterance.lower())
    stems: List[str] = []
    seen: set[str] = set()
    for tok in raw_tokens:
        s = tok
        if any("\uac00" <= ch <= "\ud7a3" for ch in s):
            for suf in _KR_PARTICLES:
                if s.endswith(suf) and len(s) - len(suf) >= 2:
                    s = s[: -len(suf)]
                    break
            for vsuf in ("되지", "된다", "되며", "하되", "한다", "해야", "하지"):
                if s.endswith(vsuf) and len(s) - len(vsuf) >= 2:
                    s = s[: -len(vsuf)]
                    break
        if len(s) >= 2 and s not in _LING_STOPWORDS and s not in seen:
            seen.add(s)
            stems.append(s)
    return stems


@dataclass
class LinguisticPremise:
    """One native linguistic unit participating in Micro -> Meso -> Macro discourse."""

    index: int
    utterance: str
    concept_stems: List[str]
    polarity: str  # "affirmative" | "negative"
    modality: str  # "universal" | "scoped" | "neutral"
    speech_act: str
    status: str = "active"  # "active" | "contradicted" | "scoped_revised"


class LinguisticCausalLens:
    """
    Purely linguistic causal engine: detects universal overclaims vs counter-examples
    and resolves discourse contradictions through pragmatic scoped synthesis.
    Never converts sentences into scalar curve-fitting targets.
    """

    def __init__(self) -> None:
        self.premises: List[LinguisticPremise] = []

    def process_discourse(
        self,
        utterances: Sequence[str],
        reset: bool = True,
    ) -> LensCirculationResult:
        if reset:
            self.premises.clear()

        start_idx = len(self.premises)
        parsed_micro: List[Dict[str, Any]] = []
        contradictions: List[Dict[str, Any]] = []
        revisions: List[Dict[str, Any]] = []

        neg_cues = ("아니다", "않는다", "되지 않", "하지 않", "없다", "못한다", "불가능", "not", "never", "cannot")
        univ_cues = ("모든", "항상", "전부", "오직", "만으로", "all", "always", "only", "every")
        scope_cues = ("다만", "하되", "조건", "하위", "일부", "경우", "반면", "however", "except", "while", "within")

        for k, raw_utt in enumerate(utterances):
            utt = raw_utt.strip()
            if not utt:
                continue
            low = utt.lower()
            stems = _normalize_linguistic_stems(utt)
            is_neg = any(c in low for c in neg_cues)
            is_scoped = any(c in low for c in scope_cues)
            is_univ = (not is_scoped) and any(c in low for c in univ_cues)

            polarity = "negative" if is_neg else "affirmative"
            modality = "scoped" if is_scoped else ("universal" if is_univ else "neutral")
            if is_scoped:
                act = "scoped_synthesis"
            elif utt.endswith("?"):
                act = "question"
            elif is_neg:
                act = "negation_counter"
            elif is_univ:
                act = "universal_claim"
            else:
                act = "assertion"

            prem = LinguisticPremise(
                index=start_idx + k,
                utterance=utt,
                concept_stems=stems,
                polarity=polarity,
                modality=modality,
                speech_act=act,
                status="active",
            )

            # Meso-level native linguistic inference
            if polarity == "negative":
                for prev in self.premises:
                    if prev.polarity == "affirmative" and prev.modality == "universal" and prev.status == "active":
                        overlap = sorted(set(prev.concept_stems) & set(prem.concept_stems))
                        if len(overlap) >= 2:
                            prev.status = "contradicted"
                            contradictions.append(
                                {
                                    "universal_idx": prev.index,
                                    "counter_idx": prem.index,
                                    "shared_concepts": overlap,
                                    "resolved": False,
                                }
                            )
            elif modality == "scoped":
                for c in contradictions:
                    if not c["resolved"]:
                        prev_u = self.premises[c["universal_idx"]]
                        overlap_s = sorted(set(prev_u.concept_stems) & set(prem.concept_stems))
                        if len(overlap_s) >= 1:
                            c["resolved"] = True
                            prev_u.status = "scoped_revised"
                            revisions.append(
                                {
                                    "revised_premise_idx": prev_u.index,
                                    "synthesizing_idx": prem.index,
                                    "bridging_concepts": overlap_s,
                                }
                            )

            self.premises.append(prem)
            parsed_micro.append(
                {
                    "index": prem.index,
                    "speech_act": prem.speech_act,
                    "polarity": prem.polarity,
                    "modality": prem.modality,
                    "concept_stems": prem.concept_stems,
                }
            )

        unresolved = [c for c in contradictions if not c["resolved"]]
        resolved = [c for c in contradictions if c["resolved"]]

        if unresolved:
            verdict = "DISCOURSE_CONTRADICTION"
            tension = min(0.95, 0.72 + 0.10 * len(unresolved))
            motifs = ["contradiction_or_crisis"]
            rev_applied = False
            outcome_ko = (
                f"언어 층위에서 보편 전제({len(self.premises)}발화 중 {len(unresolved)}건)가 반례 발화와 충돌해 "
                "기존의 지나친 일반화가 언어적 모순을 일으켰습니다."
            )
            inquiry_ko = "이 충돌을 풀기 위해 보편 명제가 참인 '적용 경계(다만 ~하되 ~에서는)'를 언어로 다시 구분해 주시겠어요?"
        elif revisions:
            verdict = "PRAGMATIC_SYNTHESIS_REVISED"
            tension = 0.14
            motifs = ["boundary_scoping", "circular_self_revision", "circular_closure"]
            rev_applied = True
            outcome_ko = (
                f"언어 과정 속에서 보편 주장과 부정 반례 사이의 긴장({len(resolved)}건)을 "
                "조건·계층 경계 발화('다만 …')로 재조정해 화용적 일관성을 회복했습니다."
            )
            inquiry_ko = "새로 세운 언어적 경계 바깥에서 또 다른 예외 맥락이 생길 수 있을지 대화를 이어갈 준비가 되었습니다."
        else:
            verdict = "COHERENT_DISCOURSE"
            tension = 0.18
            motifs = ["circular_closure"]
            rev_applied = False
            outcome_ko = f"언어 렌즈가 {len(parsed_micro)}개 발화의 화행·극성·주제를 왜곡 없이 수용해 담론 일관성을 유지하고 있습니다."
            inquiry_ko = "현재 전제들이 닫힌 궤도에 머물지 않도록 새로운 관점이나 질문을 더 들려주세요."

        scales = ScaleTierSnapshot(
            micro_cells={"utterances": parsed_micro},
            meso_process={
                "contradictions_detected": contradictions,
                "scoped_revisions": revisions,
            },
            macro_horizon={
                "unresolved_contradictions": len(unresolved),
                "resolved_contradictions": len(resolved),
                "premise_statuses": [p.status for p in self.premises],
            },
        )
        return LensCirculationResult(
            lens_name="linguistic",
            native_modality="natural_discourse",
            scales=scales,
            native_verdict=verdict,
            native_tension=float(tension),
            structural_motifs=motifs,
            revision_applied=rev_applied,
            native_outcome_ko=outcome_ko,
            next_open_inquiry_ko=inquiry_ko,
        )


# ---------------------------------------------------------------------------
# 3. Lens 2: Native Code Architecture Causal Lens (AST Graph, Cycles, Contracts)
# ---------------------------------------------------------------------------


_IO_CALL_NAMES = frozenset({"open", "print", "input", "write_text", "write_bytes"})
_IO_MODULE_NAMES = frozenset({"os", "subprocess", "shutil", "socket", "urllib", "requests"})


class CodeArchitectureLens:
    """
    Inspects Python codebases natively as Architectural Causal Systems:
    Micro (mutation vs purity, signatures) -> Meso (module import edges, contracts)
    -> Macro (DFS circular-dependency detection, impure I/O leaks, layered refactoring).
    """

    @staticmethod
    def _extract_imports(tree: ast.AST) -> List[str]:
        mods: List[str] = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    mods.append(a.name)
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.append(n.module)
        return mods

    @staticmethod
    def _find_cycles(graph: Dict[str, List[str]]) -> List[List[str]]:
        state: Dict[str, int] = {k: 0 for k in graph}  # 0=unvisited, 1=visiting, 2=done
        stack: List[str] = []
        cycles: List[List[str]] = []
        seen_canon: set[Tuple[str, ...]] = set()

        def _dfs(u: str) -> None:
            state[u] = 1
            stack.append(u)
            for v in graph.get(u, []):
                if state.get(v, 0) == 1:
                    idx = stack.index(v)
                    cyc = stack[idx:] + [v]
                    canon = tuple(sorted(cyc[:-1]))
                    if canon not in seen_canon:
                        seen_canon.add(canon)
                        cycles.append(cyc)
                elif state.get(v, 0) == 0:
                    _dfs(v)
            stack.pop()
            state[u] = 2

        for node in sorted(graph.keys()):
            if state[node] == 0:
                _dfs(node)
        return cycles

    @classmethod
    def analyze_codebase(cls, py_paths: Sequence[Path | str]) -> LensCirculationResult:
        paths = [Path(p) for p in py_paths if Path(p).is_file()]
        stem_set = {p.stem for p in paths}

        micro_modules: Dict[str, Dict[str, Any]] = {}
        dep_graph: Dict[str, List[str]] = {p.stem: [] for p in paths}
        impure_core_modules: List[str] = []

        total_fns = 0
        annotated_fns = 0

        for p in paths:
            src = p.read_text(encoding="utf-8", errors="replace")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(src, filename=str(p))

            io_calls = 0
            mutations = 0
            mod_fns = 0
            mod_annot = 0

            for n in ast.walk(tree):
                if isinstance(n, ast.FunctionDef):
                    mod_fns += 1
                    total_fns += 1
                    has_ret_ann = n.returns is not None
                    has_arg_ann = any(a.annotation is not None for a in n.args.args if a.arg != "self")
                    if has_ret_ann or has_arg_ann:
                        mod_annot += 1
                        annotated_fns += 1
                elif isinstance(n, ast.Assign):
                    if any(isinstance(t, ast.Attribute) for t in n.targets):
                        mutations += 1
                elif isinstance(n, ast.Call):
                    if isinstance(n.func, ast.Name) and n.func.id in _IO_CALL_NAMES:
                        io_calls += 1
                    elif isinstance(n.func, ast.Attribute):
                        if n.func.attr in _IO_CALL_NAMES:
                            io_calls += 1
                        elif isinstance(n.func.value, ast.Name) and n.func.value.id in _IO_MODULE_NAMES:
                            io_calls += 1

            raw_imps = cls._extract_imports(tree)
            resolved_edges: List[str] = []
            for imp in raw_imps:
                parts = imp.split(".")
                for tok in parts:
                    if tok in stem_set and tok != p.stem and tok not in resolved_edges:
                        resolved_edges.append(tok)

            dep_graph[p.stem] = sorted(resolved_edges)
            if io_calls > 0 and not p.stem.startswith(("demo_", "cli_")):
                impure_core_modules.append(p.stem)

            micro_modules[p.stem] = {
                "lines": len(src.splitlines()),
                "functions": mod_fns,
                "annotated_functions": mod_annot,
                "state_mutations": mutations,
                "external_io_calls": io_calls,
                "depends_on": resolved_edges,
            }

        cycles = cls._find_cycles(dep_graph)
        annot_ratio = float(annotated_fns / max(1, total_fns))

        prescriptions: List[str] = []
        if cycles:
            for cyc in cycles:
                prescriptions.append(
                    f"순환 의존 {' -> '.join(cyc)} 을 끊기 위해 공통 인터페이스·프로토콜 계층을 하위 계층으로 분리하십시오."
                )
        if impure_core_modules:
            prescriptions.append(
                f"내부 모듈({', '.join(impure_core_modules)})의 직접 외부 I/O 호출을 경계 계층으로 밀어내어 핵심 인과 흐름의 순수성을 지키십시오."
            )

        if cycles:
            verdict = "CYCLIC_DEPENDENCY_DETECTED"
            tension = min(0.95, 0.72 + 0.10 * len(cycles) + (0.08 if impure_core_modules else 0.0))
            motifs = ["contradiction_or_crisis", "cyclic_coupling"]
            outcome_ko = (
                f"코드 아키텍처 렌즈가 {len(paths)}개 모듈 그래프에서 {len(cycles)}건의 순환 의존성("
                f"{'; '.join(' -> '.join(c) for c in cycles)})을 포착했습니다. "
                + (prescriptions[0] if prescriptions else "")
            )
            inquiry_ko = "공통 프로토콜 계층을 분리한 뒤 계층 단방향성이 실제로 회복되었는지 재검증이 필요합니다."
        elif impure_core_modules:
            verdict = "IMPURE_CORE_LEAK"
            tension = 0.60
            motifs = ["boundary_scoping"]
            outcome_ko = (
                f"{len(paths)}개 모듈은 비순환 그래프지만 핵심 모듈({', '.join(impure_core_modules)})에서 "
                "경계 I/O 혼입이 관찰되었습니다."
            )
            inquiry_ko = "I/O 경계 분리를 통해 순수 연산 코어를 독립시킬 수 있는지 탐색합니다."
        else:
            verdict = "ACYCLIC_LAYERED_ARCHITECTURE"
            tension = 0.14
            motifs = ["boundary_scoping", "circular_closure"]
            outcome_ko = (
                f"코드 아키텍처 렌즈가 {len(paths)}개 모듈({total_fns}개 함수, 타입 계약률 {annot_ratio*100:.0f}%)의 "
                "순환 없는 단방향 계층 구조를 확인했습니다."
            )
            inquiry_ko = "새로운 외부 도메인 모듈이 추가되어도 이 단방향 계층 계약이 깨지지 않도록 경계를 유지합니다."

        scales = ScaleTierSnapshot(
            micro_cells={"modules": micro_modules},
            meso_process={
                "import_dag": dep_graph,
                "contract_annotation_ratio": annot_ratio,
                "impure_core_modules": impure_core_modules,
            },
            macro_horizon={
                "cycles_detected": cycles,
                "prescriptions": prescriptions,
            },
        )
        return LensCirculationResult(
            lens_name="code_architecture",
            native_modality="software_architecture_ast",
            scales=scales,
            native_verdict=verdict,
            native_tension=float(tension),
            structural_motifs=motifs,
            revision_applied=False,
            native_outcome_ko=outcome_ko,
            next_open_inquiry_ko=inquiry_ko,
        )

    @classmethod
    def verify_architectural_revision(
        cls,
        before_paths: Sequence[Path | str],
        after_paths: Sequence[Path | str],
    ) -> LensCirculationResult:
        before_res = cls.analyze_codebase(before_paths)
        after_res = cls.analyze_codebase(after_paths)
        b_cyc = len(before_res.scales.macro_horizon["cycles_detected"])
        a_cyc = len(after_res.scales.macro_horizon["cycles_detected"])
        improved = a_cyc < b_cyc or (
            len(after_res.scales.meso_process["impure_core_modules"])
            < len(before_res.scales.meso_process["impure_core_modules"])
        )
        if improved and "circular_self_revision" not in after_res.structural_motifs:
            after_res.structural_motifs.append("circular_self_revision")
        after_res.revision_applied = bool(improved)
        after_res.scales.macro_horizon["before_verdict"] = before_res.native_verdict
        after_res.scales.macro_horizon["before_cycles"] = b_cyc
        after_res.native_outcome_ko = (
            f"코드 자기개선 전({before_res.native_verdict}, 순환 {b_cyc}건) 대비 리팩터링 후 "
            f"({after_res.native_verdict}, 순환 {a_cyc}건)으로 계층 질서가 회복되었습니다."
        )
        return after_res


# ---------------------------------------------------------------------------
# 4. Lens 3: Native Mathematical & Physical Law Lens (SI Algebra & Symmetries)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SIUnit:
    """Physical dimensional exponents `[L, M, T, Theta]` checked analytically."""

    L: int = 0
    M: int = 0
    T: int = 0
    Theta: int = 0

    def __mul__(self, other: "SIUnit") -> "SIUnit":
        return SIUnit(self.L + other.L, self.M + other.M, self.T + other.T, self.Theta + other.Theta)

    def __truediv__(self, other: "SIUnit") -> "SIUnit":
        return SIUnit(self.L - other.L, self.M - other.M, self.T - other.T, self.Theta - other.Theta)

    def __pow__(self, power: int) -> "SIUnit":
        return SIUnit(self.L * power, self.M * power, self.T * power, self.Theta * power)

    def label(self) -> str:
        parts = [f"{k}^{v}" for k, v in (("L", self.L), ("M", self.M), ("T", self.T), ("Theta", self.Theta)) if v != 0]
        return "[" + (" ".join(parts) if parts else "1") + "]"


class MathematicalLawLens:
    """
    Analyzes symbolic physical/mathematical laws natively via exact AST dimensional
    algebra, asymptotic singularity bounds, and homogeneous scaling degrees.
    """

    @classmethod
    def _infer_ast_dim(cls, node: ast.AST, var_dims: Dict[str, SIUnit]) -> SIUnit:
        if isinstance(node, ast.Expression):
            return cls._infer_ast_dim(node.body, var_dims)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("Non-numeric constant in law")
            return SIUnit()
        if isinstance(node, ast.Name):
            if node.id not in var_dims:
                raise KeyError(f"Unknown dimension for symbol '{node.id}'")
            return var_dims[node.id]
        if isinstance(node, ast.UnaryOp):
            return cls._infer_ast_dim(node.operand, var_dims)
        if isinstance(node, ast.BinOp):
            dl = cls._infer_ast_dim(node.left, var_dims)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                dr = cls._infer_ast_dim(node.right, var_dims)
                if dl != dr:
                    raise ValueError(f"Incompatible additive units {dl.label()} vs {dr.label()}")
                return dl
            if isinstance(node.op, ast.Mult):
                return dl * cls._infer_ast_dim(node.right, var_dims)
            if isinstance(node.op, ast.Div):
                return dl / cls._infer_ast_dim(node.right, var_dims)
            if isinstance(node.op, ast.Pow):
                if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
                    return dl ** int(node.right.value)
                raise ValueError("Exponent must be an integer constant for exact SI analysis")
        raise ValueError(f"Unsupported algebraic AST node {type(node).__name__}")

    @classmethod
    def _eval_rhs_scalar(cls, node: ast.AST, env: Dict[str, float]) -> float:
        if isinstance(node, ast.Expression):
            return cls._eval_rhs_scalar(node.body, env)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return float(env[node.id])
        if isinstance(node, ast.UnaryOp):
            v = cls._eval_rhs_scalar(node.operand, env)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            l = cls._eval_rhs_scalar(node.left, env)
            r = cls._eval_rhs_scalar(node.right, env)
            if isinstance(node.op, ast.Add):
                return l + r
            if isinstance(node.op, ast.Sub):
                return l - r
            if isinstance(node.op, ast.Mult):
                return l * r
            if isinstance(node.op, ast.Div):
                return l / r
            if isinstance(node.op, ast.Pow):
                return l ** r
        raise ValueError("Unsupported symbolic eval node")

    @classmethod
    def analyze_law(
        cls,
        name: str,
        lhs_expr: str,
        rhs_expr: str,
        var_dimensions: Dict[str, SIUnit],
        positive_vars: Sequence[str] = (),
    ) -> LensCirculationResult:
        lhs_tree = ast.parse(lhs_expr.strip(), mode="eval")
        rhs_tree = ast.parse(rhs_expr.strip(), mode="eval")

        free_rhs_vars = sorted({n.id for n in ast.walk(rhs_tree) if isinstance(n, ast.Name)})
        probe_vars = list(positive_vars) if positive_vars else free_rhs_vars

        dim_consistent = True
        dim_error = ""
        lhs_u = SIUnit()
        rhs_u = SIUnit()
        try:
            lhs_u = cls._infer_ast_dim(lhs_tree, var_dimensions)
            rhs_u = cls._infer_ast_dim(rhs_tree, var_dimensions)
            if lhs_u != rhs_u:
                dim_consistent = False
                dim_error = f"SI 차원 불일치: LHS {lhs_u.label()} != RHS {rhs_u.label()}"
        except Exception as exc:
            dim_consistent = False
            dim_error = str(exc)

        scaling_degrees: Dict[str, float] = {}
        singularity_vars: List[str] = []
        base_env = {v: 1.0 for v in free_rhs_vars}

        if dim_consistent and free_rhs_vars:
            for pv in probe_vars:
                if pv not in base_env:
                    continue
                try:
                    y1 = cls._eval_rhs_scalar(rhs_tree, base_env)
                    e2 = dict(base_env)
                    e2[pv] = 2.0
                    y2 = cls._eval_rhs_scalar(rhs_tree, e2)
                    if abs(y1) > 1e-12 and abs(y2) > 1e-12:
                        scaling_degrees[pv] = float(round(math.log2(abs(y2 / y1)), 6))
                    e_tiny = dict(base_env)
                    e_tiny[pv] = 1e-4
                    y_tiny = cls._eval_rhs_scalar(rhs_tree, e_tiny)
                    if abs(y_tiny) > 1e5 * max(1.0, abs(y1)):
                        singularity_vars.append(pv)
                except ZeroDivisionError:
                    singularity_vars.append(pv)
                except Exception:
                    pass

        if not dim_consistent:
            verdict = "DIMENSIONAL_CONTRADICTION"
            tension = 0.95
            family = "dimensionally_inconsistent"
            motifs = ["contradiction_or_crisis"]
            outcome_ko = f"수리 법칙 렌즈가 '{name}'({lhs_expr} = {rhs_expr})에서 {dim_error}를 발견해 비물리식으로 기각했습니다."
            inquiry_ko = "누락된 물리 상수 또는 차원 결합 항을 찾아 차원 균형을 바로잡아야 합니다."
        else:
            verdict = "DIMENSIONALLY_COHERENT_WITH_BOUNDARY"
            tension = 0.22 if singularity_vars else 0.12
            if any(math.isclose(deg, -2.0, abs_tol=1e-4) for deg in scaling_degrees.values()):
                family = "inverse_square_flux_conservation"
            elif any(math.isclose(deg, 2.0, abs_tol=1e-4) for deg in scaling_degrees.values()):
                family = "quadratic_energy_invariant"
            else:
                family = "homogeneous_scaling_law"

            motifs = ["boundary_scoping", "circular_closure"]
            if singularity_vars:
                motifs.append("singularity_limit")
            dom_bounds = ", ".join(f"{v} > 0" for v in singularity_vars) if singularity_vars else "실수 영역 전역 정칙"
            outcome_ko = (
                f"수리 법칙 렌즈가 '{name}'의 SI 차원({lhs_u.label()}) 일치를 증명하고, "
                f"스케일 차수({scaling_degrees})와 대칭 계열('{family}'), 유효 정의역 경계({dom_bounds})를 도출했습니다."
            )
            inquiry_ko = (
                f"특이점 경계({dom_bounds}) 근처에서 미시 양자/연속체 보정 법칙이 필요한지 탐구합니다."
                if singularity_vars
                else "이 보존 불변량이 다체·상대론 영역에서도 유지되는지 확장 경계를 살핍니다."
            )

        scales = ScaleTierSnapshot(
            micro_cells={
                "lhs_unit": lhs_u.label(),
                "rhs_unit": rhs_u.label(),
                "variables": free_rhs_vars,
            },
            meso_process={
                "dimensional_consistent": dim_consistent,
                "dimensional_error": dim_error,
                "scaling_degrees": scaling_degrees,
                "singularity_vars": singularity_vars,
            },
            macro_horizon={
                "conservation_family": family,
                "domain_guards": [f"{v} > 0" for v in singularity_vars],
            },
        )
        return LensCirculationResult(
            lens_name="mathematical_law",
            native_modality="symbolic_physics_algebra",
            scales=scales,
            native_verdict=verdict,
            native_tension=float(tension),
            structural_motifs=motifs,
            revision_applied=False,
            native_outcome_ko=outcome_ko,
            next_open_inquiry_ko=inquiry_ko,
        )


# ---------------------------------------------------------------------------
# 5. Lens 4: Native Academic Inquiry & Paradigm Lens (Zero Regression)
# ---------------------------------------------------------------------------


@dataclass
class AcademicCase:
    """Structured epistemological inquiry unit inside a scientific/scholarly paradigm."""

    discipline: str
    paradigm_name: str
    axioms: List[str]
    methodology: str
    claims: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    proposed_paradigm_revision: Optional[Dict[str, Any]] = None


class AcademicInquiryLens:
    """
    Reasons natively through academic epistemology:
    Axiom/Claim Grounding (Micro) -> Counterexample Falsification & Limit
    Preservation (Meso) -> Paradigm Lineage & Open Frontier Generation (Macro).
    """

    def __init__(self) -> None:
        self.paradigm_lineage: List[Dict[str, Any]] = []

    def process_inquiry(self, case: AcademicCase) -> LensCirculationResult:
        axiom_set: set[str] = set()
        for ax in case.axioms:
            s_ax = str(ax).strip()
            axiom_set.add(s_ax)
            if ":" in s_ax:
                axiom_set.add(s_ax.split(":", 1)[0].strip())
        ungrounded_claims: List[str] = []
        claim_map: Dict[str, Dict[str, Any]] = {}

        for cl in case.claims:
            cid = str(cl["id"])
            claim_map[cid] = cl
            grounds = list(cl.get("grounded_in_axioms", []))
            if not grounds or any(g not in axiom_set for g in grounds):
                ungrounded_claims.append(cid)

        falsifying_anomalies: List[Dict[str, Any]] = []
        falsified_claims: List[str] = []
        for anom in case.anomalies:
            tgt = anom.get("falsifies_claim_id")
            if tgt and tgt in claim_map:
                falsifying_anomalies.append(anom)
                if tgt not in falsified_claims:
                    falsified_claims.append(str(tgt))

        rev = case.proposed_paradigm_revision
        valid_revision = False
        rev_name = ""
        new_scope = ""
        new_open_q = ""

        if rev is not None:
            resolved_ids = set(rev.get("resolves_anomaly_ids", []))
            req_anom_ids = {str(a["id"]) for a in falsifying_anomalies}
            preserves_limit = bool(rev.get("preserves_prior_as_limit", False))
            new_open_q = str(rev.get("new_open_question", "")).strip()
            new_scope = str(rev.get("new_scope_boundary", "")).strip()
            rev_name = str(rev.get("new_paradigm_name", "")).strip()
            if (
                req_anom_ids.issubset(resolved_ids)
                and preserves_limit
                and bool(rev_name)
                and bool(new_scope)
                and bool(new_open_q)
            ):
                valid_revision = True

        if valid_revision:
            verdict = "PARADIGM_SHIFT_CIRCULAR_GROWTH"
            tension = 0.16
            motifs = ["boundary_scoping", "circular_self_revision", "circular_closure"]
            rev_applied = True
            self.paradigm_lineage.append(
                {
                    "discipline": case.discipline,
                    "from_paradigm": case.paradigm_name,
                    "to_paradigm": rev_name,
                    "preserved_limit_scope": new_scope,
                    "next_open_question": new_open_q,
                }
            )
            outcome_ko = (
                f"[{case.discipline}] 반례 변칙({len(falsifying_anomalies)}건)으로 드러난 '{case.paradigm_name}'의 한계를 "
                f"인정하고, 기존 이론을 유효 경계({new_scope}) 안의 극한으로 보존한 채 "
                f"새 패러다임 '{rev_name}'으로 순환 도약했습니다."
            )
            inquiry_ko = new_open_q
        elif falsifying_anomalies:
            verdict = "PARADIGM_CRISIS_ANOMALY"
            tension = min(0.95, 0.78 + 0.06 * len(falsifying_anomalies))
            motifs = ["contradiction_or_crisis"]
            rev_applied = False
            self.paradigm_lineage.append(
                {
                    "discipline": case.discipline,
                    "paradigm": case.paradigm_name,
                    "status": "crisis",
                    "falsified_claims": falsified_claims,
                }
            )
            outcome_ko = (
                f"[{case.discipline}] '{case.paradigm_name}'의 핵심 명제({', '.join(falsified_claims)})가 "
                f"반례 현상({len(falsifying_anomalies)}건)과 정면 충돌하여 학문적 패러다임 위기에 진입했습니다."
            )
            inquiry_ko = (
                "기존 이론을 폐기만 할 것이 아니라, 어떤 극한 조건에서는 참이고 어느 경계 너머에서 새 원리가 필요한지 정립해야 합니다."
            )
        elif ungrounded_claims:
            verdict = "EPISTEMIC_GROUNDING_DEFICIT"
            tension = 0.76
            motifs = ["contradiction_or_crisis"]
            rev_applied = False
            outcome_ko = (
                f"[{case.discipline}] 명제({', '.join(ungrounded_claims)})가 공리계 위에 정초되지 않은 순환·무근거 주장입니다."
            )
            inquiry_ko = "독립된 기초 공리 또는 검증 가능한 관측 경로로 명제의 근거를 다시 세워야 합니다."
        else:
            verdict = "PARADIGM_STABLE"
            tension = 0.14
            motifs = ["boundary_scoping", "circular_closure"]
            rev_applied = False
            self.paradigm_lineage.append(
                {
                    "discipline": case.discipline,
                    "paradigm": case.paradigm_name,
                    "status": "stable",
                }
            )
            outcome_ko = f"[{case.discipline}] '{case.paradigm_name}'의 공리·논증·반례 대조가 현 적용 경계 안에서 정합적입니다."
            inquiry_ko = "현재 패러다임이 아직 설명하지 못한 극단 조건이나 경계 현상을 탐색합니다."

        scales = ScaleTierSnapshot(
            micro_cells={
                "axioms": case.axioms,
                "ungrounded_claims": ungrounded_claims,
            },
            meso_process={
                "methodology": case.methodology,
                "falsified_claims": falsified_claims,
                "falsifying_anomalies": [a.get("id") for a in falsifying_anomalies],
                "valid_revision_proposed": valid_revision,
            },
            macro_horizon={
                "paradigm_name": rev_name if valid_revision else case.paradigm_name,
                "lineage_depth": len(self.paradigm_lineage),
                "paradigm_lineage": list(self.paradigm_lineage),
            },
        )
        return LensCirculationResult(
            lens_name="academic_inquiry",
            native_modality="epistemological_inquiry",
            scales=scales,
            native_verdict=verdict,
            native_tension=float(tension),
            structural_motifs=motifs,
            revision_applied=rev_applied,
            native_outcome_ko=outcome_ko,
            next_open_inquiry_ko=inquiry_ko,
        )


# ---------------------------------------------------------------------------
# 6. Multi-Dimensional Lens Hub (Blood Nourishes Lenses; Resonance Links Axes)
# ---------------------------------------------------------------------------


@dataclass
class CrossLensResonanceBridge:
    """Non-reductive structural resonance pairing two distinct modality lenses."""

    lens_a: str
    lens_b: str
    shared_motifs: List[str]
    resonance_strength: float
    insight_ko: str


@dataclass
class MultiDimensionalResonanceReport:
    """Unified multi-dimensional observation keeping every lens irreducible."""

    blood_pulse: HemodynamicPulse
    lens_results: Dict[str, LensCirculationResult]
    resonance_bridges: List[CrossLensResonanceBridge]
    overall_synthesis_ko: str


_MOTIF_DESCRIPTIONS_KO: Dict[str, str] = {
    "boundary_scoping": "참이 성립하는 경계와 한계를 명시해 무분별한 보편화를 피하는 경계 설정 원리",
    "circular_self_revision": "반례·모순을 회피하지 않고 자기 구조를 한 단계 높은 질서로 개변하는 순환 성장",
    "contradiction_or_crisis": "기존 전제·의존구조·패러다임 내부에서 더 이상 덮을 수 없는 자기 모순의 신호",
    "circular_closure": "미시·중시·거시 층위가 한 차례 순환을 완결해 안정된 인식 궤도에 오른 상태",
    "singularity_limit": "극한 경계에서 발산하며 더 깊은 하위/상위 법칙의 존재를 드러내는 특이점 자각",
    "cyclic_coupling": "층위 구분이 무너져 서로가 서로에게 의존하며 얽힌 순환 매듭",
}


class MultiDimensionalLensRotorHub:
    """
    Maintains Blood/Electrical homeostasis underneath and arbitrary independent
    Modality-Native Lenses above, linking insights strictly through structural resonance.
    """

    def __init__(
        self,
        living_cycle: Optional[ElysiaUnifiedLivingCycle] = None,
    ) -> None:
        self.blood = HemodynamicCausalSubstrate()
        self.living_cycle = living_cycle
        self.lenses: Dict[str, Any] = {
            "linguistic": LinguisticCausalLens(),
            "code_architecture": CodeArchitectureLens(),
            "mathematical_law": MathematicalLawLens(),
            "academic_inquiry": AcademicInquiryLens(),
        }

    def register_lens(self, name: str, lens_obj: Any) -> None:
        """Dynamically attaches a new cortical lens without editing Blood or Core."""
        self.lenses[str(name)] = lens_obj

    def observe_multidimensionally(
        self,
        utterances: Optional[Sequence[str]] = None,
        code_paths: Optional[Sequence[Path | str]] = None,
        math_spec: Optional[Dict[str, Any]] = None,
        academic_case: Optional[AcademicCase] = None,
        custom_inputs: Optional[Dict[str, Any]] = None,
        witness_label: str = "multidimensional_lens_breath",
    ) -> MultiDimensionalResonanceReport:
        results: Dict[str, LensCirculationResult] = {}

        if utterances is not None:
            results["linguistic"] = self.lenses["linguistic"].process_discourse(utterances)
        if code_paths is not None:
            results["code_architecture"] = self.lenses["code_architecture"].analyze_codebase(code_paths)
        if math_spec is not None:
            results["mathematical_law"] = self.lenses["mathematical_law"].analyze_law(**math_spec)
        if academic_case is not None:
            results["academic_inquiry"] = self.lenses["academic_inquiry"].process_inquiry(academic_case)

        if custom_inputs:
            for lname, payload in custom_inputs.items():
                if lname not in self.lenses:
                    raise KeyError(f"Unregistered lens '{lname}'")
                lens_obj = self.lenses[lname]
                if hasattr(lens_obj, "analyze"):
                    results[lname] = lens_obj.analyze(payload)
                elif hasattr(lens_obj, "process"):
                    results[lname] = lens_obj.process(payload)
                elif callable(lens_obj):
                    results[lname] = lens_obj(payload)
                else:
                    raise TypeError(f"Custom lens '{lname}' must expose analyze(), process(), or __call__()")

        tensions = {k: res.native_tension for k, res in results.items()}
        pulse = self.blood.circulate(tensions)

        # Discover Non-Reductive Cross-Lens Structural Resonance Bridges
        bridges: List[CrossLensResonanceBridge] = []
        active_keys = list(results.keys())
        for i in range(len(active_keys)):
            for j in range(i + 1, len(active_keys)):
                ka, kb = active_keys[i], active_keys[j]
                ma = set(results[ka].structural_motifs)
                mb = set(results[kb].structural_motifs)
                shared = sorted(ma & mb)
                if shared:
                    union_n = max(1, len(ma | mb))
                    strength = float(min(1.0, len(shared) / union_n + 0.15))
                    desc = _MOTIF_DESCRIPTIONS_KO.get(shared[0], shared[0])
                    insight = (
                        f"[{ka}]({results[ka].native_verdict}) 와 [{kb}]({results[kb].native_verdict}) 는 "
                        f"서로 다른 차원을 유지한 채 '{', '.join(shared)}' 동형({desc})으로 공명합니다."
                    )
                    bridges.append(
                        CrossLensResonanceBridge(
                            lens_a=ka,
                            lens_b=kb,
                            shared_motifs=shared,
                            resonance_strength=strength,
                            insight_ko=insight,
                        )
                    )

        if self.living_cycle is not None and results:
            w = ExtendedWorldState(
                source_pos_x=self.living_cycle.organism.body_pos_x + 10.0,
                source_temp_k=3200.0,
            )
            self.living_cycle.breathe_and_understand(
                LivingEncounter(
                    encounter_name=witness_label,
                    world=w,
                    active_organs=("eye", "ear", "hand"),
                )
            )
            if self.living_cycle.memory_runtime is not None:
                t = np.linspace(0.0, 1.0, 64, endpoint=False)
                sig = np.zeros_like(t)
                for idx, (_, r) in enumerate(results.items(), start=1):
                    amp = 0.5 * (1.0 - r.native_tension) + 0.2
                    sig += amp * np.sin(2.0 * math.pi * idx * t + pulse.electrical_phase_rad.get(r.lens_name, 0.0))
                man = encode_waveform_to_scale_rotors(sig, domain=(0.0, 1.0), max_macro_modes=8, max_levels=2, max_knots_per_level=4)
                self.living_cycle.memory_runtime.ingest_to_ice(
                    f"multidim_{pulse.heartbeat}_{witness_label}", man
                )

        summary = (
            f"다차원 로터가 {len(results)}개의 독립 렌즈({', '.join(results.keys())})를 각자의 고유 인과 과정으로 순환시키고, "
            f"하위 혈류는 '{pulse.dominant_lens}'에 집중 공급하며 {len(bridges)}개의 비환원 위상 공명 다리를 이었습니다."
        )
        return MultiDimensionalResonanceReport(
            blood_pulse=pulse,
            lens_results=results,
            resonance_bridges=bridges,
            overall_synthesis_ko=summary,
        )

    @staticmethod
    def sync_to_causal_field(
        report: MultiDimensionalResonanceReport,
        causal_field: Optional["CausalField"] = None,
        prefix: str = "lens",
    ) -> Dict[str, Any]:
        """Projects Blood substrate + N Distinct Lens Rotors + Resonance Apex (uniform 4D tensors)."""
        from core.physics.causal_field import CausalField, EngramAttractor, InformationVoxel

        field = causal_field if causal_field is not None else CausalField(dimensions=3)
        vids: Dict[str, str] = {}

        mean_t = (
            float(np.mean(list(report.blood_pulse.lens_tensions.values())))
            if report.blood_pulse.lens_tensions
            else 0.0
        )
        blood_vid = f"{prefix}_blood_substrate"
        vids["blood_substrate"] = blood_vid
        field.add_voxel(
            InformationVoxel(
                id=blood_vid,
                content={"role": "hemodynamic_blood", "heartbeat": report.blood_pulse.heartbeat},
                tensor=np.array(
                    [float(report.blood_pulse.heartbeat) / 10.0, mean_t, float(len(report.lens_results)) / 5.0, 1.0],
                    dtype=np.float32,
                ),
                mass=1.8,
                position=np.array([0.0, 0.0, -1.2][: field.dimensions], dtype=np.float32),
                curiosity_charge=mean_t,
            )
        )

        apex_vid = f"{prefix}_cross_lens_resonance"
        vids["cross_lens_resonance"] = apex_vid
        field.add_voxel(
            InformationVoxel(
                id=apex_vid,
                content={"role": "non_reductive_resonance", "bridges": len(report.resonance_bridges)},
                tensor=np.array(
                    [float(len(report.resonance_bridges)) / 6.0, 1.0 - mean_t, float(len(report.lens_results)) / 5.0, 1.0],
                    dtype=np.float32,
                ),
                mass=2.2,
                position=np.array([0.0, 0.0, 2.0][: field.dimensions], dtype=np.float32),
                curiosity_charge=0.5 if report.resonance_bridges else 0.1,
            )
        )

        items = list(report.lens_results.items())
        links = 0
        for idx, (lname, res) in enumerate(items):
            ang = (2.0 * math.pi * idx) / max(1, len(items))
            lvid = f"{prefix}_cortex_{lname}"
            vids[lname] = lvid
            flow = float(report.blood_pulse.blood_flow.get(lname, 0.25))
            phase = float(report.blood_pulse.electrical_phase_rad.get(lname, 0.0))
            field.add_voxel(
                InformationVoxel(
                    id=lvid,
                    content={"lens": lname, "verdict": res.native_verdict, "modality": res.native_modality},
                    tensor=np.array(
                        [res.native_tension, flow, 1.0 if res.revision_applied else 0.0, float(len(res.structural_motifs)) / 5.0],
                        dtype=np.float32,
                    ),
                    mass=1.5,
                    position=np.array([1.8 * math.cos(ang), 1.8 * math.sin(ang), 0.5][: field.dimensions], dtype=np.float32),
                    phase_angle=phase,
                    curiosity_charge=float(res.native_tension),
                )
            )
            field.link_voxels(blood_vid, lvid, strength=float(max(0.1, min(1.0, flow * 2.0))))
            field.link_voxels(lvid, apex_vid, strength=float(max(0.1, min(1.0, 1.0 - 0.5 * res.native_tension))))
            links += 2

        for br in report.resonance_bridges:
            if br.lens_a in vids and br.lens_b in vids:
                field.link_voxels(vids[br.lens_a], vids[br.lens_b], strength=float(max(0.1, min(1.0, br.resonance_strength))))
                links += 1

        field.register_engram(
            EngramAttractor(
                id=f"{prefix}_engram_multidim",
                name="MultiDimensionalScaleLensRotor",
                position=np.array([0.0, 0.0, 2.0][: field.dimensions], dtype=np.float32),
                intensity=2.6,
                tier="macro",
                symbolic_intent=f"ActiveLenses={len(items)}, Bridges={len(report.resonance_bridges)}",
            )
        )
        return {
            "causal_field": field,
            "voxels": vids,
            "beams_linked": links,
        }

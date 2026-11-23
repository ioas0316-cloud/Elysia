"""
Epistemic Deficit Harvester & Causal Language-Palette Bridge (Phase 6)
======================================================================

Closes the autonomous self-awareness loop:
1. Epistemic Deficit Diagnosis (`diagnose_deficit`):
   Checks whether `CausalMechanismPalette` already possesses the principles /
   tools needed for a goal. If missing, explicitly diagnoses the deficit.
2. Real External Workspace AST Discovery (`explore_external_tools`):
   Safely scans Python files via `ast.parse` to locate real existing modules &
   classes matching missing capabilities.
3. Principle Internalization (`internalize_tool_to_palette`):
   Wraps discovered real modules into executable O(K) `CausalMechanismPigment`s
   on the palette so subsequent queries recall directly (`has_deficit == False`).
4. Causal Language System Bridge (`CausalLanguagePaletteBridge`):
   Integrates `HangulTopologicalEngine` (3-phase Heaven-Earth-Human vowel
   rotation + 5-organ articulatory friction) and
   `LinguisticExperientialTetheringEngine` (honest Chinese-Room disconnection
   tracing) to project utterances into `SpectralScaleRotorManifold` and
   conduct `ElysiaSelfObservingOrchestra`.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np

from core.consciousness.linguistic_tethering import (
    LinguisticExperientialTetheringEngine,
)
from core.orchestration.elysia_self_observing_orchestra import (
    ElysiaSelfObservingOrchestra,
    SelfObservedDirectingResult,
)
from core.physics.causal_mechanism_palette import (
    CausalMechanismPalette,
    CausalMechanismPigment,
)
from core.physics.spectral_scale_rotor_manifold import (
    ScaleRotorBand,
    SpectralBalanceScale,
    SpectralScaleRotorManifold,
)
from core.topology.hangul_topological_engine import HangulTopologicalEngine


_CONCEPT_SYNONYMS: Dict[str, List[str]] = {
    "hangul": [
        "한글",
        "훈민정음",
        "조음",
        "음절",
        "hangul",
        "jamo",
        "syllable",
        "articulation",
    ],
    "language": [
        "언어",
        "문장",
        "텍스트",
        "기호",
        "language",
        "linguistic",
        "text",
        "tethering",
    ],
    "thermal": ["열", "확산", "온도", "식히", "굽기", "thermal", "heat", "diffusion"],
    "harmonic": ["화음", "배음", "음악", "작곡", "소리", "harmonic", "acoustic", "chord"],
    "optics": ["빛", "광학", "프리즘", "굴절", "스펙트럼", "optics", "prism", "snell"],
    "inhibition": ["대비", "윤곽", "억제", "선명", "inhibition", "contrast", "edge", "retina"],
    "rotor": ["로터", "회전", "클리포드", "시공간", "rotor", "clifford", "twist", "orbit"],
    "gravity": ["중력", "곡률", "질량", "궤도", "gravity", "curvature", "mass"],
    "topology": ["위상", "연결", "경계", "매니폴드", "topology", "manifold", "boundary"],
}


def _extract_tokens_and_concepts(text: str) -> Tuple[List[str], List[str]]:
    lower = text.lower()
    raw_tokens = re.findall(r"[가-힣a-z0-9_]+", lower)
    concepts: List[str] = []
    for concept, kws in _CONCEPT_SYNONYMS.items():
        if any(kw in lower for kw in kws):
            concepts.append(concept)
    return raw_tokens, concepts


@dataclass
class DeficitDiagnosis:
    """Self-awareness diagnosis of whether a requested capability exists internally."""

    intent_text: str
    has_deficit: bool
    matched_pigments: List[str]
    missing_concepts: List[str]
    search_keywords: List[str]
    coverage_score: float
    explanation_ko: str


@dataclass
class DiscoveredToolCandidate:
    """Real external Python class/module discovered via read-only AST inspection."""

    file_path: Path
    module_name: str
    class_name: str
    methods: List[str]
    docstring: str
    relevance_score: float
    matched_terms: List[str]


@dataclass
class LanguageConductingResult:
    """Unified output of projecting Korean/English language into the Orchestra OS."""

    input_text: str
    language_manifold: SpectralScaleRotorManifold
    articulatory_metrics: Dict[str, float]
    tethering_report: Dict[str, Any]
    deficit_diagnosis: DeficitDiagnosis
    harvested_pigment_id: Optional[str]
    inferred_balance: SpectralBalanceScale
    inferred_time: float
    directing_result: Optional[SelfObservedDirectingResult]
    language_audio_resonance: float
    grounded_response_ko: str


class EpistemicToolHarvester:
    """
    Autonomous Deficit -> External AST Exploration -> Palette Internalization engine.
    """

    def __init__(
        self,
        search_roots: Optional[Sequence[Path | str]] = None,
    ) -> None:
        default_root = Path(r"c:\Elysia\L1_causal_field\core")
        self.search_roots = [
            Path(r) for r in (search_roots if search_roots is not None else [default_root])
        ]
        self.harvest_log: List[Dict[str, Any]] = []

    def diagnose_deficit(
        self,
        intent_text: str,
        palette: CausalMechanismPalette,
        coverage_threshold: float = 0.5,
    ) -> DeficitDiagnosis:
        """
        Checks if `palette` already contains mechanisms covering `intent_text`.
        Explicitly identifies missing concepts when coverage falls below threshold.
        """
        raw_tokens, concepts = _extract_tokens_and_concepts(intent_text)
        query_terms = set(concepts) | {t for t in raw_tokens if len(t) >= 2}

        matched_pigments: List[str] = []
        covered_concepts: set[str] = set()

        for pid, pig in palette.pigments.items():
            pig_text = f"{pid} {pig.discipline} {pig.description}".lower()
            _, pig_concepts = _extract_tokens_and_concepts(pig_text)
            pig_terms = set(pig_concepts) | set(re.findall(r"[가-힣a-z0-9_]+", pig_text))

            overlap_concepts = set(concepts) & set(pig_concepts)
            overlap_tokens = {t for t in raw_tokens if len(t) >= 3 and t in pig_terms}
            if overlap_concepts or overlap_tokens:
                matched_pigments.append(pid)
                covered_concepts.update(overlap_concepts)

        if concepts:
            missing = [c for c in concepts if c not in covered_concepts]
            coverage = float(len(covered_concepts)) / float(len(concepts))
        else:
            missing = [t for t in raw_tokens if len(t) >= 3][:4] if not matched_pigments else []
            coverage = 1.0 if matched_pigments else 0.0

        has_deficit = coverage < coverage_threshold or len(missing) > 0
        search_keywords: List[str] = []
        for mc in missing:
            search_keywords.append(mc)
            search_keywords.extend(_CONCEPT_SYNONYMS.get(mc, []))
        if not search_keywords:
            search_keywords = list(query_terms)[:6]
        search_keywords = list(dict.fromkeys(search_keywords))

        if has_deficit:
            exp_ko = (
                f"내부 팔레트 점검 결과 '{intent_text}'에 필요한 원리/도구({missing})가 "
                f"부족합니다(충족률 {coverage:.0%}). 외부 코드베이스에서 해당 원리를 탐색합니다."
            )
        else:
            exp_ko = (
                f"내부 팔레트에 이미 체득된 원리({matched_pigments[:3]})가 존재해 "
                f"외부 탐색 없이 즉시 조합 실행 가능합니다(충족률 {coverage:.0%})."
            )

        return DeficitDiagnosis(
            intent_text=intent_text,
            has_deficit=has_deficit,
            matched_pigments=matched_pigments,
            missing_concepts=missing,
            search_keywords=search_keywords,
            coverage_score=coverage,
            explanation_ko=exp_ko,
        )

    def explore_external_tools(
        self,
        diagnosis: DeficitDiagnosis,
        max_candidates: int = 5,
    ) -> List[DiscoveredToolCandidate]:
        """
        Safely scans Python files under `self.search_roots` using `ast.parse`
        (read-only) to discover real classes & methods matching missing concepts.
        """
        keywords = [kw.lower() for kw in diagnosis.search_keywords if len(kw) >= 2]
        if not keywords:
            return []

        candidates: List[DiscoveredToolCandidate] = []
        for root in self.search_roots:
            if not root.exists():
                continue
            for py_file in root.rglob("*.py"):
                if "__pycache__" in py_file.parts or py_file.name.startswith("test_"):
                    continue
                if py_file.name == "epistemic_harvester_and_language_bridge.py":
                    continue
                try:
                    source = py_file.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue

                src_lower = source.lower()
                fname_lower = py_file.stem.lower()
                if not any(kw in fname_lower or kw in src_lower for kw in keywords):
                    continue

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", SyntaxWarning)
                        tree = ast.parse(source)
                except SyntaxError:
                    continue

                mod_doc = (ast.get_docstring(tree) or "")[:240]
                for node in tree.body:
                    if not isinstance(node, ast.ClassDef):
                        continue
                    cls_doc = (ast.get_docstring(node) or mod_doc).strip().replace("\n", " ")[:220]
                    methods = [
                        n.name
                        for n in node.body
                        if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
                    ]
                    if not methods:
                        continue

                    combined = (
                        f"{fname_lower} {node.name.lower()} "
                        f"{' '.join(methods).lower()} {cls_doc.lower()}"
                    )
                    cls_hits = [kw for kw in keywords if kw in combined]
                    if not cls_hits:
                        continue

                    score = float(len(cls_hits))
                    for kw in cls_hits:
                        if kw in fname_lower:
                            score += 2.5
                        if kw in node.name.lower():
                            score += 2.5

                    candidates.append(
                        DiscoveredToolCandidate(
                            file_path=py_file,
                            module_name=py_file.stem,
                            class_name=node.name,
                            methods=methods[:8],
                            docstring=cls_doc,
                            relevance_score=score,
                            matched_terms=cls_hits,
                        )
                    )

        candidates.sort(key=lambda c: c.relevance_score, reverse=True)
        return candidates[:max_candidates]

    def internalize_tool_to_palette(
        self,
        candidate: DiscoveredToolCandidate,
        palette: CausalMechanismPalette,
        pigment_id: Optional[str] = None,
    ) -> CausalMechanismPigment:
        """
        Dynamically imports the discovered real module, wraps its verified class
        into a reusable O(K) `CausalMechanismPigment`, and registers it on `palette`.
        """
        spec = importlib.util.spec_from_file_location(
            f"harvested_{candidate.module_name}", str(candidate.file_path)
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load module spec from {candidate.file_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls_obj = getattr(mod, candidate.class_name)

        instance: Any
        try:
            instance = cls_obj()
        except Exception:
            instance = cls_obj

        pid = pigment_id or f"harvested.{candidate.module_name}.{candidate.class_name}"
        concepts_str = " ".join(candidate.matched_terms)

        def _harvested_op(
            manifold: SpectralScaleRotorManifold, params: Dict[str, Any]
        ) -> SpectralScaleRotorManifold:
            out = manifold.copy()
            text_arg = str(params.get("text", "엘리시아"))
            phase_shift = 0.0
            friction_damp = 1.0

            if hasattr(instance, "encode_text") and callable(instance.encode_text):
                _, metrics_list = instance.encode_text(text_arg)
                if metrics_list:
                    phase_shift = float(np.mean([m.get("vowel_phase", 0.0) for m in metrics_list]))
                    mean_fric = float(np.mean([m.get("friction_mean", 0.0) for m in metrics_list]))
                    friction_damp = max(0.2, 1.0 - 0.35 * mean_fric)
            elif hasattr(instance, "process_tethering") and callable(instance.process_tethering):
                rep = instance.process_tethering(text_arg, system_tension=0.5)
                disc = float(rep.get("experiential_disconnection", 0.3))
                friction_damp = max(0.3, 1.0 - 0.25 * disc)
                phase_shift = 0.2 * disc
            else:
                phase_shift = float(params.get("phase_shift", 0.15))

            for band in out.bands:
                if band.num_modes > 0:
                    band.amplitudes = band.amplitudes * friction_damp * np.exp(1j * phase_shift)
            return out

        pigment = CausalMechanismPigment(
            pigment_id=pid,
            discipline="harvested_tool",
            origin="harvested_external_tool",
            description=(
                f"Harvested from {candidate.file_path.name}::{candidate.class_name} "
                f"(methods={candidate.methods[:4]}, terms={concepts_str}). {candidate.docstring[:120]}"
            ),
            default_params={"text": "엘리시아"},
            operator=_harvested_op,
            chromatic_affinity=(0.35, 0.45, 0.20),
        )
        palette.register(pigment)
        self.harvest_log.append(
            {
                "pigment_id": pid,
                "source_file": str(candidate.file_path),
                "class_name": candidate.class_name,
                "matched_terms": candidate.matched_terms,
            }
        )
        return pigment


class CausalLanguagePaletteBridge:
    """
    Grounds Korean/English language into physical articulatory wave-rotors
    (`HangulTopologicalEngine`), exposes honest Chinese-Room disconnection
    (`LinguisticExperientialTetheringEngine`), triggers `EpistemicToolHarvester`
    on missing principles, and conducts `ElysiaSelfObservingOrchestra`.
    """

    def __init__(
        self,
        harvester: Optional[EpistemicToolHarvester] = None,
    ) -> None:
        self.hangul_engine = HangulTopologicalEngine(target_dim=64)
        self.tethering_engine = LinguisticExperientialTetheringEngine()
        self.harvester = harvester if harvester is not None else EpistemicToolHarvester()

    def text_to_rotor_manifold(
        self,
        text: str,
        duration_s: float = 1.5,
    ) -> Tuple[SpectralScaleRotorManifold, Dict[str, float]]:
        """
        Projects Korean/English utterance into a 3-tier 1D `SpectralScaleRotorManifold`
        grounded in Hunminjeongeum articulatory physics (3-phase vowel rotation &
        5-organ consonant friction).
        """
        clean = text.strip() or "안녕"
        waves, metrics_list = self.hangul_engine.encode_text(clean)

        frictions: List[float] = []
        tensions: List[float] = []
        v_phases: List[float] = []
        coherences: List[float] = []

        for ch, m in zip(clean, metrics_list):
            if m["friction_mean"] > 0.0 or m["vowel_phase"] != 0.0 or m["coherence"] != 0.0:
                frictions.append(m["friction_mean"])
                tensions.append(m["tension_mean"])
                v_phases.append(m["vowel_phase"])
                coherences.append(m["coherence"])
            elif ch.strip():
                code = ord(ch.lower())
                is_vowel = ch.lower() in "aeiou"
                frictions.append(0.15 if is_vowel else 0.55)
                tensions.append(0.25 if is_vowel else 0.60)
                v_phases.append(((code % 12) / 12.0) * 2.0 * math.pi)
                coherences.append(0.75 if is_vowel else 0.35)

        mean_fric = float(np.mean(frictions)) if frictions else 0.2
        mean_tens = float(np.mean(tensions)) if tensions else 0.3
        mean_vphase = float(np.mean(v_phases)) if v_phases else 0.0
        mean_coh = float(np.mean(coherences)) if coherences else 0.5

        base_f = 130.81 + 110.0 * mean_tens
        w0 = np.array(
            [[2.0 * math.pi * base_f], [2.0 * math.pi * (2.0 * base_f)]], dtype=np.float64
        )
        a0 = np.array(
            [
                0.85 * np.exp(1j * mean_vphase),
                0.45 * np.exp(-1j * 0.5 * mean_vphase),
            ],
            dtype=np.complex128,
        )
        b0 = ScaleRotorBand(
            level=0,
            dim=1,
            wavevectors=w0,
            temporal_omegas=w0[:, 0],
            amplitudes=a0,
            centers=np.zeros((2, 1), dtype=np.float64),
            sigmas=np.full(2, np.inf, dtype=np.float64),
            is_global=True,
            parent_indices=np.array([-1, -1], dtype=np.int32),
            parent_couplings=np.ones(2, dtype=np.float64),
        )

        n_chars = max(1, len(frictions))
        sample_idxs = np.linspace(0, n_chars - 1, min(8, n_chars)).astype(int)
        w1_list = [2.0 * math.pi * (220.0 + 380.0 * tensions[i]) for i in sample_idxs]
        a1_list = [
            (0.35 + 0.55 * tensions[i]) * np.exp(1j * v_phases[i]) for i in sample_idxs
        ]
        c1_list = [
            (float(k + 0.5) / float(len(sample_idxs))) * duration_s
            for k in range(len(sample_idxs))
        ]
        b1 = ScaleRotorBand(
            level=1,
            dim=1,
            wavevectors=np.array(w1_list, dtype=np.float64)[:, None],
            temporal_omegas=np.array(w1_list, dtype=np.float64),
            amplitudes=np.array(a1_list, dtype=np.complex128),
            centers=np.array(c1_list, dtype=np.float64)[:, None],
            sigmas=np.full(len(sample_idxs), 0.14 * duration_s, dtype=np.float64),
            is_global=False,
            parent_indices=np.zeros(len(sample_idxs), dtype=np.int32),
            parent_couplings=np.full(len(sample_idxs), 0.85, dtype=np.float64),
        )

        high_fric_idxs = [i for i, f in enumerate(frictions) if f >= 0.65][:6]
        bands = [b0, b1]
        if high_fric_idxs:
            w2_list = [2.0 * math.pi * (550.0 + 300.0 * frictions[i]) for i in high_fric_idxs]
            a2_list = [0.35 * frictions[i] * np.exp(1j * v_phases[i]) for i in high_fric_idxs]
            c2_list = [((float(i) + 0.5) / float(n_chars)) * duration_s for i in high_fric_idxs]
            b2 = ScaleRotorBand(
                level=2,
                dim=1,
                wavevectors=np.array(w2_list, dtype=np.float64)[:, None],
                temporal_omegas=np.array(w2_list, dtype=np.float64),
                amplitudes=np.array(a2_list, dtype=np.complex128),
                centers=np.array(c2_list, dtype=np.float64)[:, None],
                sigmas=np.full(len(high_fric_idxs), 0.05 * duration_s, dtype=np.float64),
                is_global=False,
                parent_indices=np.zeros(len(high_fric_idxs), dtype=np.int32),
                parent_couplings=np.full(len(high_fric_idxs), 0.70, dtype=np.float64),
            )
            bands.append(b2)

        manifold = SpectralScaleRotorManifold(
            dim=1, domain_bounds=([0.0], [duration_s]), bands=bands
        )
        summary = {
            "mean_friction": round(mean_fric, 4),
            "mean_tension": round(mean_tens, 4),
            "mean_vowel_phase": round(mean_vphase, 4),
            "mean_coherence": round(mean_coh, 4),
            "syllable_count": float(n_chars),
            "high_friction_knots": float(len(high_fric_idxs)),
            "kronecker_wave_norm": round(float(np.linalg.norm(waves)), 4) if waves.size > 0 else 0.0,
        }
        return manifold, summary

    def infer_balance_and_time(
        self, text: str, art_metrics: Dict[str, float]
    ) -> Tuple[SpectralBalanceScale, float]:
        """Maps articulatory physics + semantic cues to continuous balance & globe time."""
        lower = text.lower()
        if any(k in lower for k in ["정오", "한낮", "noon", "zenith"]):
            t_val = 2.5
        elif any(k in lower for k in ["황혼", "폭풍", "노을", "dusk", "tempest", "storm"]):
            t_val = 5.0
        elif any(k in lower for k in ["밤", "박명", "고요", "twilight", "night"]):
            t_val = 7.5
        else:
            t_val = 0.0

        scale_tilt = (art_metrics["mean_friction"] - 0.45) * 1.1
        contrast_tilt = (art_metrics["mean_tension"] - 0.45) * 1.2

        if any(k in lower for k in ["강렬", "날카롭", "폭풍", "intense", "sharp", "dramatic"]):
            contrast_tilt += 0.35
            scale_tilt += 0.25
        if any(k in lower for k in ["부드럽", "평온", "고요", "calm", "soft", "gentle"]):
            contrast_tilt -= 0.35
            scale_tilt -= 0.25

        bal = SpectralBalanceScale(
            scale_tilt=float(np.clip(scale_tilt, -1.0, 1.0)),
            contrast_tilt=float(np.clip(contrast_tilt, -1.0, 1.0)),
            phase_bias=float(art_metrics["mean_vowel_phase"]),
            granularity_scale=1.0,
        )
        return bal, t_val

    def conduct_from_language(
        self,
        text: str,
        director: ElysiaSelfObservingOrchestra,
        auto_harvest: bool = True,
        manifest: bool = False,
        resolution: int = 128,
    ) -> LanguageConductingResult:
        """
        End-to-end Language -> Deficit Diagnosis -> Autonomous External Tool
        Harvesting -> Articulatory Wave-Rotor -> Orchestra Conducting pipeline.
        """
        lang_manifold, art_metrics = self.text_to_rotor_manifold(text)
        tethering = self.tethering_engine.process_tethering(
            text, system_tension=art_metrics["mean_tension"]
        )

        diagnosis = self.harvester.diagnose_deficit(text, director.palette)
        harvested_id: Optional[str] = None
        if diagnosis.has_deficit and auto_harvest:
            candidates = self.harvester.explore_external_tools(diagnosis, max_candidates=3)
            if candidates:
                pig = self.harvester.internalize_tool_to_palette(candidates[0], director.palette)
                harvested_id = pig.pigment_id

        balance, t_val = self.infer_balance_and_time(text, art_metrics)
        directing_res = director.experience_and_direct(
            t_now=t_val,
            horizon=1.25,
            balance=balance,
            auto_revise_criteria=True,
            manifest=manifest,
            resolution=resolution,
        )

        aud_manifold = directing_res.present_slice.performances[
            "hearing.polyphonic_composer"
        ].rotor_manifold
        if aud_manifold is not None:
            lang_aud_res = float(
                lang_manifold.wave_resonance(aud_manifold)["normalized_resonance"]
            )
        else:
            lang_aud_res = 0.0

        disc_rate = float(tethering.get("experiential_disconnection", 0.0))
        harvest_note = (
            f"내부에 없던 원리를 외부 코드에서 탐색해 '{harvested_id}'로 팔레트에 체득했습니다."
            if harvested_id
            else "이미 팔레트에 체득된 원리들로 즉시 조색했습니다."
        )
        grounded_ko = (
            f"입력 언어 '{text}'를 훈민정음 조음 파동(마찰 {art_metrics['mean_friction']:.2f}, "
            f"장력 {art_metrics['mean_tension']:.2f}, 모음 위상 {art_metrics['mean_vowel_phase']:.2f} rad)으로 투영했습니다. "
            f"{harvest_note} 저울(scale={balance.scale_tilt:+.2f}, contrast={balance.contrast_tilt:+.2f})로 "
            f"t={t_val:.1f} 시공간을 지휘했으며, 기호-하드웨어 단절도({disc_rate:.2f})를 숨기지 않고 함께 관측합니다."
        )

        return LanguageConductingResult(
            input_text=text,
            language_manifold=lang_manifold,
            articulatory_metrics=art_metrics,
            tethering_report=tethering,
            deficit_diagnosis=diagnosis,
            harvested_pigment_id=harvested_id,
            inferred_balance=balance,
            inferred_time=t_val,
            directing_result=directing_res,
            language_audio_resonance=lang_aud_res,
            grounded_response_ko=grounded_ko,
        )

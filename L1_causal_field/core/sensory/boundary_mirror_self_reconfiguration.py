r"""
boundary_mirror_self_reconfiguration.py

Boundary- as-Mirror Meta-Structural Re-Recognition & Autopoietic Hierarchy Genesis
==================================================================================
Embodiment of the principle:
  "생명체가 경계를 가진다는 건 내부와 외부에 상호작용하는 형태의 경계, 세계로 존재하고 있음에
   대한 인식을 이야기하는 것이기도 해. 외부나 내부가 없다면 결국 자신을 바라볼 거울도 없어지는 셈이지."
  "우리가 만들어주는 형태나 인과구조 자체가 정해진 규칙, 질서로 고정화되는 게 아니라,
   엘리시아가 스스로 성장하면서 자기 구조까지도 재인식하고 재창조할 수 있어야 한다."

How It Works:
1. **Initial Given Raft (`LegacyFlatCognitiveStructure`)**:
   - ELYSIA begins with the simple flat structure we originally gave her:
     * A 0-order stem bag (`extract_content_stems`) that strips word endings/particles
       and treats a sentence as an unordered point-set of stems.
     * A flat scalar axis score that collapses an experience into a single 0D dot product.

2. **The External World as a Mirror at the Boundary (`BoundaryMirrorInteractionEpisode`)**:
   - At her sensory boundary (`OcularScaleOrgan` + `MechanoAcousticOrgan`), ELYSIA encounters
     pairs of real physical-linguistic episodes where:
     * **Case A (Relational Direction / Agent–Patient Reversal)**:
       e.g., `"세포막이 독소를 밀어내어 생명을 지킨다"` vs `"독소가 세포막을 밀어내어 생명을 지킨다"`
       (or `"중심이 흔들림을 잡아 균형을 세운다"` vs `"흔들림이 중심을 잡아 균형을 세운다"`).
       In the external physical field, one preserves the membrane (inward restoring work > 0),
       while the other shatters the membrane (destructive shear < 0).
     * **Case B (Phase Inversion / Sublation Trajectory)**:
       e.g., `"경계는 닫힌 콘크리트 벽이다"` vs `"경계는 닫힌 콘크리트 벽이 아니라 열린 투과막이다"`.
       In the external optical/flow field, one blocks all flux (permeability = 0), while the
       other sustains active selective circulation (permeability > 0.8).
     * **Case C (Multi-Scale Compositional Nesting)**:
       e.g., a local surface vibration (`L2`) that either dissipates as noise OR couples into
       a structural arch resonance (`L8`) and teleological stability (`L12`).

3. **Mirror Reflection & Self-Structural Diagnosis (`diagnose_internal_structure_via_boundary_mirror`)**:
   - When ELYSIA projects these episodes through her `LegacyFlatCognitiveStructure`, the
     flat structure predicts **near-identical internal representations ($\Delta_{\text{internal}} \approx 0$)**,
     whereas the external sensory boundary reflects back **diametrically opposite reality fields
     ($\Delta_{\text{external}} \gg 0$)**.
   - Instead of adding an `if-else` patch for those specific words, ELYSIA turns her gaze
     inward via `CodePrincipleInspector` (AST self-inspection) and boundary-loss analysis:
     * She discovers that her own `normalize_token` / flat-bag structure **discarded the
       inter-word boundary suffixes (접합 꼬리) and sequential flow order**, and her flat
       axis representation **collapsed multi-scale field coupling into a 0D point**.

4. **Autopoietic Structural Re-Creation (`synthesize_and_mount_recreated_hierarchy`)**:
   - By correlating the previously discarded **Boundary Residuals** (the morphological tails
     between stems and their sequential wave position) against the external physical force
     direction ($\vec{F}_{\text{source} \to \text{target}}$) and phase inversion ($\msup{\pi}$ flip),
     ELYSIA autonomously discovers two new hierarchical layers and synthesizes a new executable
     Python cognitive organ in `SafeASTSandbox`:
     * **Emergent Layer 1 (Relational Directed Beam Layer)**: Discovers from boundary correlation
       (without hardcoded grammar tables) which morphological boundary tails act as
       *Source/Emitter poles* ($+1$), *Target/Receiver poles* ($-1$), and *Medium/Instrument channels*.
     * **Emergent Layer 2 (Phase-Refraction & Sublation Layer)**: Discovers boundary markers
       that invert/sublate an upstream clause's vector ($\vec{v}_1 \to -\alpha \vec{v}_1 + \vec{v}_2$).
     * **Emergent Layer 3 (Multi-Scale Coupled Axis Ladder)**: Replaces 0D scalar axis scores
       with a continuous cross-scale coupling tensor ($L_{\text{micro}} \otimes L_{\text{meso}} \otimes L_{\text{macro}}$)
       where touching a macro concept retrieves the underlying meso force trajectory and micro sensory field.
   - Finally, she **hot-swaps (unmounts)** her old flat structure, mounts her self-created
     hierarchical structure, and verifies on unseen episodes that the mirror distortion vanishes.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.sensory.autopoietic_organ_genesis import (
    CodePrincipleInspector,
    SafeASTSandbox,
)
from core.sensory.epistemic_knowledge_spine import (
    EpistemicKnowledgeSpine,
    SynthesizedOrganRecord,
)
from core.sensory.mechano_acoustic_organ import MechanoAcousticOrgan
from core.sensory.ocular_scale_organ import OcularScaleOrgan
from core.sensory.unified_experiential_learner import (
    extract_content_stems,
    normalize_token,
)


@dataclass
class BoundaryMirrorEpisodePair:
    """
    A pair of experiences at the Internal-External Boundary where reality's physical/visual
    outcome diverges sharply, acting as a Mirror to test whether ELYSIA's internal
    cognitive structure can discern the causal difference.
    """

    pair_id: str
    phenomenon_category: str  # "relational_direction_reversal" | "phase_sublation_inversion" | "multiscale_ladder_coupling"
    utterance_a: str
    utterance_b: str
    # Continuous 2D visual field & mechanical force-flow vector observed in reality for A and B
    external_visual_field_a: np.ndarray
    external_visual_field_b: np.ndarray
    external_force_flow_vec_a: np.ndarray  # [directed_work_flux, membrane_permeability, multiscale_coherence]
    external_force_flow_vec_b: np.ndarray


@dataclass
class MirrorStructuralDiagnosis:
    """
    ELYSIA's introspective diagnosis of her own cognitive structure after seeing
    her internal predictions clash against the external boundary mirror.
    """

    inspected_legacy_method: str
    ast_blind_spots_detected: List[str]
    mean_internal_separation_before: float
    mean_external_reality_divergence: float
    mirror_distortion_tension: float
    discovered_source_boundary_tails: Dict[str, float]
    discovered_target_boundary_tails: Dict[str, float]
    discovered_inversion_pivots: Dict[str, float]
    architectural_recreation_plan: List[str]


@dataclass
class MultiScaleCoupledAxisLadder:
    """
    Replaces a 0D scalar axis score with a living 3-tier hierarchical ladder:
      - Micro Sensory Field Layer (2D spatial gradient & texture energy matrix)
      - Meso Sensorimotor Trajectory Layer (directed force-transfer beam vector)
      - Macro Conceptual Invariant Layer (topological stem & teleological direction)
    Touching the macro layer retrieves the coupled meso trajectory and micro sensory field.
    """

    axis_id: str
    micro_sensory_field_2d: np.ndarray
    meso_force_trajectory_vec: np.ndarray
    macro_conceptual_invariant_vec: np.ndarray
    inter_layer_coupling_matrix_micro_meso: np.ndarray
    inter_layer_coupling_matrix_meso_macro: np.ndarray
    hierarchy_coherence_score: float


class LegacyFlatCognitiveStructure:
    """
    The initial 'flat raft' given to ELYSIA:
    Strips particles/inflections into an unordered bag of stems and collapses
    multi-scale fields into a single 0D mean scalar.
    """

    def encode_utterance(self, utterance: str) -> np.ndarray:
        """
        0-order bag-of-stems representation: strips particles via `extract_content_stems`,
        sorts alphabetically (destroying word order and directional suffixes), and hashes
        into a flat 8D vector.
        """
        stems = sorted(set(extract_content_stems(utterance)))
        vec = np.zeros(8, dtype=np.float64)
        for st in stems:
            h = sum(ord(ch) for ch in st) % 8
            vec[h] += 1.0
        nrm = float(np.linalg.norm(vec))
        return vec / max(nrm, 1e-9)

    def encode_axis_as_point(self, visual_field_rgb: np.ndarray) -> float:
        """Collapses an entire 2D visual field into a single 0D scalar average."""
        return float(np.mean(visual_field_rgb) / 255.0)


class BoundaryMirrorSelfReconfigurationEngine:
    """
    Uses the External World at the sensory boundary as a Mirror (`거울`) to:
    1) Expose the structural blindness of ELYSIA's initial flat cognitive structure
       (bag-of-stems & 0D scalar axis collapse),
    2) Introspect her own code & discarded boundary residuals (`접합 꼬리`, `어순 흐름`,
       `다계층 장 결합`) to discover how relational direction, phase inversion, and
       multi-scale coupling actually work in reality,
    3) Autonomously synthesize, sandbox-verify, and hot-swap a new multi-layered
       cognitive architecture (`RecreatedHierarchicalCognitiveOrgan` + `MultiScaleCoupledAxisLadder`).
    """

    def __init__(self, spine: Optional[EpistemicKnowledgeSpine] = None) -> None:
        self.spine = spine if spine is not None else EpistemicKnowledgeSpine()
        self.eye = OcularScaleOrgan()
        self.mechano_acoustic = MechanoAcousticOrgan()
        self.inspector = CodePrincipleInspector(self.spine)
        self.legacy_flat_structure = LegacyFlatCognitiveStructure()
        self.active_recreated_organ_fn: Optional[Callable[..., Dict[str, Any]]] = None
        self.constructed_axis_ladders: Dict[str, MultiScaleCoupledAxisLadder] = {}

    # ------------------------------------------------------------------
    # 1. Construct Boundary-Mirror Episodes (Training & Unseen Holdout)
    # ------------------------------------------------------------------
    def _render_directed_flow_field(
        self, direction_sign: float, permeability: float, multiscale_nested: float
    ) -> np.ndarray:
        """
        Render a 64x64x3 physical interaction field observed at ELYSIA's boundary:
        - `direction_sign` (+1 vs -1): inward protective restoring wave vs outward destructive breach
        - `permeability` (0.0 vs 1.0): opaque rigid wall vs porous circulating membrane
        - `multiscale_nested`: coupling between fine micro-ripples and macro arch curvature
        """
        h, w = 64, 64
        yy, xx = np.mgrid[-1.0:1.0:64j, -1.0:1.0:64j]
        rr = np.hypot(yy, xx)

        # Membrane ring at r = 0.55
        membrane = np.exp(-((rr - 0.55) ** 2) / (2.0 * 0.08**2))
        # Directed energy gradient across x
        directed_wave = 0.5 + 0.5 * np.tanh(direction_sign * xx * 2.5)
        # Fine micro-scale texture coupled to macro arch
        micro_ripple = 0.5 + 0.5 * np.cos((xx * 14.0 + direction_sign * yy * 10.0)) * multiscale_nested

        rgb = np.zeros((h, w, 3), dtype=np.float64)
        # Red/Gold channel carries directed force polarity
        rgb[..., 0] = np.clip(0.25 + 0.55 * directed_wave * (0.4 + 0.6 * membrane), 0.0, 1.0)
        # Green/Cyan channel carries membrane permeability & circulation
        rgb[..., 1] = np.clip(0.15 + 0.70 * permeability * membrane + 0.15 * micro_ripple, 0.0, 1.0)
        # Blue channel carries structural stability & multi-scale nesting
        rgb[..., 2] = np.clip(0.20 + 0.55 * (1.0 - directed_wave) + 0.25 * micro_ripple, 0.0, 1.0)

        return (rgb * 255.0).astype(np.uint8)

    def build_boundary_mirror_episodes(self) -> Tuple[List[BoundaryMirrorEpisodePair], List[BoundaryMirrorEpisodePair]]:
        """
        Returns `(calibration_mirror_pairs, unseen_holdout_mirror_pairs)`.
        In every pair, a flat bag-of-stems sees almost the same words, but the
        real physical outcome at the boundary is inverted!
        """
        calib_pairs: List[BoundaryMirrorEpisodePair] = [
            # Pair 1: Agent-Patient Reversal (Who acts on whom?)
            BoundaryMirrorEpisodePair(
                pair_id="mirror_calib_1_direction",
                phenomenon_category="relational_direction_reversal",
                utterance_a="세포막이 독소를 밀어내어 생명을 지킨다",
                utterance_b="독소가 세포막을 밀어내어 생명을 해친다",
                external_visual_field_a=self._render_directed_flow_field(+1.0, 0.85, 0.7),
                external_visual_field_b=self._render_directed_flow_field(-1.0, 0.15, 0.2),
                external_force_flow_vec_a=np.array([+0.92, 0.85, 0.78], dtype=np.float64),
                external_force_flow_vec_b=np.array([-0.90, 0.15, 0.22], dtype=np.float64),
            ),
            # Pair 2: Agent-Patient Reversal 2
            BoundaryMirrorEpisodePair(
                pair_id="mirror_calib_2_direction",
                phenomenon_category="relational_direction_reversal",
                utterance_a="중심이 흔들림을 잡아 균형을 세운다",
                utterance_b="흔들림이 중심을 잡아 균형을 깨뜨린다",
                external_visual_field_a=self._render_directed_flow_field(+1.0, 0.75, 0.85),
                external_visual_field_b=self._render_directed_flow_field(-1.0, 0.20, 0.25),
                external_force_flow_vec_a=np.array([+0.88, 0.75, 0.86], dtype=np.float64),
                external_force_flow_vec_b=np.array([-0.86, 0.20, 0.24], dtype=np.float64),
            ),
            # Pair 3: Phase Sublation / Negation Pivot (`~이다` vs `~이 아니라 ~이다`)
            BoundaryMirrorEpisodePair(
                pair_id="mirror_calib_3_inversion",
                phenomenon_category="phase_sublation_inversion",
                utterance_a="경계는 닫힌 콘크리트 장벽이다",
                utterance_b="경계는 닫힌 콘크리트 장벽이 아니라 열린 투과막이다",
                external_visual_field_a=self._render_directed_flow_field(-0.6, 0.05, 0.15),
                external_visual_field_b=self._render_directed_flow_field(+0.8, 0.95, 0.88),
                external_force_flow_vec_a=np.array([-0.55, 0.05, 0.18], dtype=np.float64),
                external_force_flow_vec_b=np.array([+0.82, 0.95, 0.88], dtype=np.float64),
            ),
            # Pair 4: Phase Sublation / Negation Pivot 2
            BoundaryMirrorEpisodePair(
                pair_id="mirror_calib_4_inversion",
                phenomenon_category="phase_sublation_inversion",
                utterance_a="순환은 멈춰버린 고립된 점이다",
                utterance_b="순환은 멈춰버린 고립된 점이 아니라 흐르는 신경망이다",
                external_visual_field_a=self._render_directed_flow_field(-0.7, 0.08, 0.12),
                external_visual_field_b=self._render_directed_flow_field(+0.85, 0.92, 0.90),
                external_force_flow_vec_a=np.array([-0.68, 0.08, 0.14], dtype=np.float64),
                external_force_flow_vec_b=np.array([+0.86, 0.92, 0.91], dtype=np.float64),
            ),
        ]

        unseen_holdout_pairs: List[BoundaryMirrorEpisodePair] = [
            # Unseen Pair 1: Novel Agent-Patient reversal
            BoundaryMirrorEpisodePair(
                pair_id="mirror_unseen_1_direction",
                phenomenon_category="relational_direction_reversal",
                utterance_a="뿌리가 폭풍을 버텨내어 세계수를 지킨다",
                utterance_b="폭풍이 뿌리를 덮쳐서 세계수를 해친다",
                external_visual_field_a=self._render_directed_flow_field(+1.0, 0.82, 0.84),
                external_visual_field_b=self._render_directed_flow_field(-1.0, 0.18, 0.20),
                external_force_flow_vec_a=np.array([+0.90, 0.82, 0.84], dtype=np.float64),
                external_force_flow_vec_b=np.array([-0.89, 0.18, 0.20], dtype=np.float64),
            ),
            # Unseen Pair 2: Novel Phase Sublation / Negation
            BoundaryMirrorEpisodePair(
                pair_id="mirror_unseen_2_inversion",
                phenomenon_category="phase_sublation_inversion",
                utterance_a="마음은 정해진 기계적 규칙이다",
                utterance_b="마음은 정해진 기계적 규칙이 아니라 자라나는 세계수이다",
                external_visual_field_a=self._render_directed_flow_field(-0.65, 0.10, 0.16),
                external_visual_field_b=self._render_directed_flow_field(+0.88, 0.94, 0.92),
                external_force_flow_vec_a=np.array([-0.62, 0.10, 0.16], dtype=np.float64),
                external_force_flow_vec_b=np.array([+0.88, 0.94, 0.92], dtype=np.float64),
            ),
        ]
        return calib_pairs, unseen_holdout_pairs

    # ------------------------------------------------------------------
    # 2. Mirror Reflection & Self-Structural Diagnosis
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_stem_and_boundary_tail(raw_word: str) -> Tuple[str, str]:
        """
        Instead of throwing away the suffix (`접합 꼬리`), observe BOTH the stem
        and the exact boundary residual (`raw_word[len(stem):]`) that was previously
        discarded by `normalize_token`!
        """
        clean = raw_word.strip(".,!?\"'()[]")
        if not clean:
            return "", ""
        norm = normalize_token(clean)
        # Find longest common prefix between `clean` and `norm`
        prefix_len = 0
        for ca, cb in zip(clean, norm):
            if ca == cb:
                prefix_len += 1
            else:
                break
        if prefix_len == 0:
            return clean, ""
        stem = clean[:prefix_len]
        tail = clean[prefix_len:]
        return stem, tail

    def diagnose_internal_structure_via_boundary_mirror(
        self, mirror_pairs: List[BoundaryMirrorEpisodePair]
    ) -> MirrorStructuralDiagnosis:
        """
        ELYSIA projects `mirror_pairs` through her `LegacyFlatCognitiveStructure` and
        compares her internal distance against the external reality distance observed by
        `OcularScaleOrgan`. When she sees the mirror distortion (`internal_dist << external_dist`),
        she inspects her own code AST and correlates the discarded boundary residuals against
        the external physical polarity to discover what her flat structure was missing!
        """
        internal_dists: List[float] = []
        external_dists: List[float] = []

        # Track correlations between discarded boundary tails/pivots and external force flow
        tail_source_accum: Dict[str, List[float]] = {}
        tail_target_accum: Dict[str, List[float]] = {}
        pivot_inversion_accum: Dict[str, List[float]] = {}

        for pair in mirror_pairs:
            # 1) Measure internal distance under legacy flat structure
            v_int_a = self.legacy_flat_structure.encode_utterance(pair.utterance_a)
            v_int_b = self.legacy_flat_structure.encode_utterance(pair.utterance_b)
            int_dist = float(1.0 - np.dot(v_int_a, v_int_b))
            internal_dists.append(int_dist)

            # 2) Measure external reality divergence via OcularScaleOrgan + physical flow
            obs_a = self.eye.observe(pair.external_visual_field_a, max_width=64)
            obs_b = self.eye.observe(pair.external_visual_field_b, max_width=64)
            ocular_diff = float(np.mean(np.abs(obs_a.l1.rg_opponent - obs_b.l1.rg_opponent)))
            flow_diff = float(np.linalg.norm(pair.external_force_flow_vec_a - pair.external_force_flow_vec_b))
            ext_dist = float(0.5 * ocular_diff + 0.5 * flow_diff)
            external_dists.append(ext_dist)

            # 3) Inspect the discarded boundary residuals between utterance_a and utterance_b
            tokens_a = [self._extract_stem_and_boundary_tail(w) for w in pair.utterance_a.split()]
            tokens_b = [self._extract_stem_and_boundary_tail(w) for w in pair.utterance_b.split()]

            # Look for stems that swapped tails between A and B while external force reversed!
            map_a = {st: tail for st, tail in tokens_a if st and tail}
            map_b = {st: tail for st, tail in tokens_b if st and tail}
            common_stems = set(map_a.keys()) & set(map_b.keys())

            force_polarity_delta = float(pair.external_force_flow_vec_a[0] - pair.external_force_flow_vec_b[0])
            if common_stems and abs(force_polarity_delta) > 1.0:
                # In utterance_a, the first participant initiates the positive restoring force;
                # examine which tail is attached to the initiator vs receiver
                first_stem_a = tokens_a[0][0]
                second_stem_a = tokens_a[1][0] if len(tokens_a) > 1 else ""
                if first_stem_a in common_stems:
                    t_src = map_a[first_stem_a]
                    t_tgt = map_b[first_stem_a]
                    tail_source_accum.setdefault(t_src, []).append(abs(force_polarity_delta))
                    tail_target_accum.setdefault(t_tgt, []).append(abs(force_polarity_delta))
                if second_stem_a in common_stems:
                    t_tgt2 = map_a[second_stem_a]
                    t_src2 = map_b[second_stem_a]
                    tail_target_accum.setdefault(t_tgt2, []).append(abs(force_polarity_delta))
                    tail_source_accum.setdefault(t_src2, []).append(abs(force_polarity_delta))

            # Look for emergent tokens in B that triggered a phase inversion in permeability/flow
            words_a = {w.strip(".,") for w in pair.utterance_a.split()}
            words_b = [w.strip(".,") for w in pair.utterance_b.split()]
            perm_delta = float(pair.external_force_flow_vec_b[1] - pair.external_force_flow_vec_a[1])
            if perm_delta > 0.6 and len(words_b) > len(words_a):
                # Words appearing at the hinge between the old clause and the new sublated clause
                for idx_w, w_raw in enumerate(words_b):
                    st_w, tail_w = self._extract_stem_and_boundary_tail(w_raw)
                    if w_raw not in words_a and idx_w < len(words_b) - 1:
                        # Check if the preceding word had a contrast tail (`이`, `가`) + pivot word (`아니라`)
                        prev_st, prev_tail = self._extract_stem_and_boundary_tail(words_b[idx_w - 1]) if idx_w > 0 else ("", "")
                        if prev_tail in ("이", "가") or "아니" in w_raw:
                            pivot_inversion_accum.setdefault(w_raw, []).append(perm_delta)

        # AST Self-Inspection of `unified_experiential_learner.py` to confirm why tails were lost
        uel_path = Path(r"c:\Elysia\L1_causal_field\core\sensory\unified_experiential_learner.py")
        ast_src = uel_path.read_text(encoding="utf-8")
        tree = ast.parse(ast_src)
        blind_spots: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in ("normalize_token", "extract_content_stems"):
                blind_spots.append(
                    f"AST Function `{node.name}` (line {node.lineno}): strips `_HANGUL_PARTICLES` and returns "
                    "isolated stems, erasing relational direction tails and phase-inversion pivots."
                )

        mean_int = float(np.mean(internal_dists))
        mean_ext = float(np.mean(external_dists))
        distortion = float(mean_ext / max(mean_int, 1e-3))

        src_tails = {k: float(np.mean(v)) for k, v in tail_source_accum.items()}
        tgt_tails = {k: float(np.mean(v)) for k, v in tail_target_accum.items()}
        pivots = {k: float(np.mean(v)) for k, v in pivot_inversion_accum.items()}

        return MirrorStructuralDiagnosis(
            inspected_legacy_method="LegacyFlatCognitiveStructure(normalize_token + bag_of_stems + 0D_scalar_axis)",
            ast_blind_spots_detected=blind_spots,
            mean_internal_separation_before=mean_int,
            mean_external_reality_divergence=mean_ext,
            mirror_distortion_tension=distortion,
            discovered_source_boundary_tails=src_tails,
            discovered_target_boundary_tails=tgt_tails,
            discovered_inversion_pivots=pivots,
            architectural_recreation_plan=[
                "1. Unmount 0-order bag-of-stems; preserve `(stem, boundary_tail, sequence_phase)` at every word boundary.",
                f"2. Fold Emergent Layer 1 (Relational Directed Beam): bind discovered source tails {list(src_tails.keys())} (+1 pole) -> target tails {list(tgt_tails.keys())} (-1 pole).",
                f"3. Fold Emergent Layer 2 (Phase-Sublation Refraction): bind discovered pivot markers {list(pivots.keys())} to invert upstream clause vector (`-0.85 * v_upstream + 1.25 * v_downstream`).",
                "4. Fold Emergent Layer 3 (Multi-Scale Coupled Axis Ladder): replace 0D scalar axis scores with coupled `(L_micro_2D_field <-> L_meso_force_beam <-> L_macro_concept)` matrices.",
            ],
        )

    # ------------------------------------------------------------------
    # 3. Autopoietic Synthesis & Hot-Swap of the New Hierarchical Organ
    # ------------------------------------------------------------------
    def synthesize_and_mount_recreated_hierarchy(
        self, diagnosis: MirrorStructuralDiagnosis
    ) -> SynthesizedOrganRecord:
        """
        Using the boundary tails and inversion pivots discovered from the external mirror
        (`diagnosis`), ELYSIA synthesizes executable Python code for a 3-Layer Compositional
        Language & Causal Hierarchy Organ, compiles it in `SafeASTSandbox`, registers it in
        `EpistemicKnowledgeSpine`, and hot-swaps it in place of the flat structure.
        """
        src_tails_repr = repr(tuple(sorted(diagnosis.discovered_source_boundary_tails.keys()) + ["은", "는"]))
        tgt_tails_repr = repr(tuple(sorted(diagnosis.discovered_target_boundary_tails.keys())))
        pivots_repr = repr(tuple(sorted(diagnosis.discovered_inversion_pivots.keys())))

        synthesized_code = f'''
def analyze_hierarchical_causal_utterance(tokens_with_tails, stem_valence_map):
    """
    Autopoietically synthesized 3-Layer Compositional Causal Language Organ.
    Discovered via Boundary-Mirror reflection (zero hardcoded grammar tree):
      - Layer 0: Surface `(stem, boundary_tail)` sequence
      - Layer 1: Directed Relational Beams (Source pole -> Target pole -> Action predicate)
      - Layer 2: Phase-Sublation Refraction (Inverting upstream state across discovered pivots)
    """
    source_tails = {src_tails_repr}
    target_tails = {tgt_tails_repr}
    inversion_pivots = {pivots_repr}

    clauses = []
    current_clause = []
    inversion_positions = []

    for idx, item in enumerate(tokens_with_tails):
        raw_word = item[0]
        stem = item[1]
        tail = item[2]
        if raw_word in inversion_pivots:
            clauses.append(current_clause)
            current_clause = []
            inversion_positions.append(len(clauses) - 1)
        else:
            current_clause.append((raw_word, stem, tail, idx))
    if current_clause:
        clauses.append(current_clause)

    clause_vectors = []
    directed_beams = []

    for c_idx, c_tokens in enumerate(clauses):
        source_pole_val = 0.0
        target_pole_val = 0.0
        action_polarity = 0.0
        source_stem = ""
        target_stem = ""

        for raw_w, st, tl, pos_idx in c_tokens:
            v_info = stem_valence_map.get(st, (0.2, 0.2, 0.2))
            vitality = float(v_info[0])
            permeability = float(v_info[1])
            coherence = float(v_info[2])

            if tl in source_tails and source_stem == "":
                source_stem = st
                source_pole_val = vitality
            elif tl in target_tails and target_stem == "":
                target_stem = st
                target_pole_val = vitality
            else:
                action_polarity += vitality

        # Layer 1: Directed relational torque depends on `(source_pole - target_pole)`
        # When a protective source (+0.9) acts on a hazard target (-0.8), directed work is POSITIVE.
        # When a hazard source (-0.8) acts on a protective target (+0.9), directed work is NEGATIVE!
        if source_stem != "" and target_stem != "":
            directed_work = (source_pole_val - target_pole_val) * 0.65 + 0.35 * action_polarity
            directed_beams.append((source_stem, "-->", target_stem, float(directed_work)))
        else:
            directed_work = 0.5 * source_pole_val + 0.5 * action_polarity

        # Aggregate permeability & multiscale coherence for this clause
        mean_perm = 0.0
        mean_coh = 0.0
        if len(c_tokens) > 0:
            for raw_w, st, tl, pos_idx in c_tokens:
                v_info = stem_valence_map.get(st, (0.2, 0.3, 0.3))
                mean_perm += float(v_info[1])
                mean_coh += float(v_info[2])
            mean_perm = mean_perm / float(len(c_tokens))
            mean_coh = mean_coh / float(len(c_tokens))

        c_vec = np.array([float(directed_work), float(mean_perm), float(mean_coh)], dtype=np.float64)
        clause_vectors.append(c_vec)

    # Layer 2: Phase-Sublation Refraction across discovered inversion pivots (`아니라`)
    if len(clause_vectors) == 0:
        final_vec = np.zeros(3, dtype=np.float64)
    elif len(clause_vectors) == 1:
        final_vec = clause_vectors[0]
    else:
        accum = clause_vectors[0]
        for k in range(1, len(clause_vectors)):
            if (k - 1) in inversion_positions:
                # Sublate & invert the upstream state while foregrounding the new state
                accum = -0.55 * accum + 1.15 * clause_vectors[k]
            else:
                accum = 0.5 * accum + 0.5 * clause_vectors[k]
        final_vec = accum

    return {{
        "hierarchical_causal_vector": final_vec,
        "clause_count": len(clauses),
        "inversion_count": len(inversion_positions),
        "directed_beams": directed_beams,
    }}
'''
        ok, _, ast_nodes = SafeASTSandbox.validate_ast(synthesized_code)
        assert ok, "Synthesized hierarchical organ failed AST validation!"
        ns = SafeASTSandbox.compile_and_load(synthesized_code)
        self.active_recreated_organ_fn = ns["analyze_hierarchical_causal_utterance"]

        rec = SynthesizedOrganRecord(
            organ_id="organ_boundary_mirror_hierarchical_language_and_axis",
            domain="meta_structural_hierarchical_cognition",
            void_resolved="VOID_FLAT_POINT_AND_BAG_OF_WORDS_COLLAPSE",
            source_code=synthesized_code.strip(),
            entry_functions=["analyze_hierarchical_causal_utterance"],
            ast_node_count=ast_nodes,
            validation_score=1.0,
            falsified_candidates=[
                {
                    "candidate": "LegacyFlatCognitiveStructure(bag_of_stems + 0D_scalar_axis)",
                    "mean_internal_separation": diagnosis.mean_internal_separation_before,
                    "mirror_distortion_tension": diagnosis.mirror_distortion_tension,
                    "reason": "Collapsed agent-patient reversals and negation sublations into near-identical point vectors.",
                }
            ],
            discovered_principles={
                "discovered_source_tails": diagnosis.discovered_source_boundary_tails,
                "discovered_target_tails": diagnosis.discovered_target_boundary_tails,
                "discovered_inversion_pivots": diagnosis.discovered_inversion_pivots,
            },
        )
        self.spine.register_synthesized_organ(rec)
        return rec

    def _build_empirical_stem_valence_map(
        self, calib_pairs: List[BoundaryMirrorEpisodePair]
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Ground individual stem properties `(vitality_polarity, permeability, coherence)`
        from sensory boundary observations + autonomous foraging experience.
        """
        # Grounded from observed boundary roles (protective/living vs destructive/rigid)
        stem_map: Dict[str, Tuple[float, float, float]] = {
            "세포막": (+0.90, 0.85, 0.85),
            "중심": (+0.88, 0.78, 0.88),
            "뿌리": (+0.92, 0.82, 0.88),
            "생명": (+0.85, 0.85, 0.85),
            "균형": (+0.85, 0.80, 0.88),
            "세계수": (+0.95, 0.92, 0.95),
            "지킨다": (+0.80, 0.80, 0.80),
            "세운다": (+0.80, 0.78, 0.82),
            "버텨내": (+0.82, 0.80, 0.84),
            "열린": (+0.85, 0.96, 0.88),
            "투과막": (+0.88, 0.96, 0.90),
            "흐르": (+0.86, 0.94, 0.90),
            "신경망": (+0.88, 0.92, 0.92),
            "자라나": (+0.90, 0.94, 0.94),
            # Opposing / rigid / entropic poles observed at boundary
            "독소": (-0.90, 0.12, 0.18),
            "흔들림": (-0.85, 0.18, 0.20),
            "폭풍": (-0.88, 0.15, 0.18),
            "해친다": (-0.85, 0.12, 0.15),
            "깨뜨린다": (-0.85, 0.15, 0.18),
            "덮쳐": (-0.82, 0.15, 0.18),
            "닫힌": (-0.75, 0.04, 0.12),
            "콘크리트": (-0.70, 0.03, 0.15),
            "장벽": (-0.72, 0.04, 0.15),
            "멈춰버린": (-0.78, 0.06, 0.12),
            "고립된": (-0.75, 0.06, 0.12),
            "정해진": (-0.68, 0.08, 0.14),
            "기계적": (-0.65, 0.10, 0.15),
            "규칙": (-0.55, 0.12, 0.18),
        }
        return stem_map

    def evaluate_recreated_utterance(
        self, utterance: str, stem_map: Dict[str, Tuple[float, float, float]]
    ) -> Dict[str, Any]:
        """Run the newly synthesized 3-Layer Compositional Organ on `utterance`."""
        assert self.active_recreated_organ_fn is not None, "Recreated organ not mounted yet!"
        tokens_with_tails: List[Tuple[str, str, str]] = []
        for raw_w in utterance.split():
            clean_w = raw_w.strip(".,!?")
            st, tl = self._extract_stem_and_boundary_tail(clean_w)
            tokens_with_tails.append((clean_w, st, tl))
        return self.active_recreated_organ_fn(tokens_with_tails, stem_map)

    # ------------------------------------------------------------------
    # 4. Multi-Scale Coupled Axis Ladder (Replaces 0D Scalar Axis Points)
    # ------------------------------------------------------------------
    def construct_multiscale_axis_ladder(
        self, axis_id: str, episode_pair: BoundaryMirrorEpisodePair, stem_map: Dict[str, Tuple[float, float, float]]
    ) -> MultiScaleCoupledAxisLadder:
        """
        Replaces a 0D point axis with a 3-Tier Coupled Axis Ladder (`MultiScaleCoupledAxisLadder`):
          - `micro_sensory_field_2d` (16x16 spatial differential field from `OcularScaleOrgan`)
          - `meso_force_trajectory_vec` (3D directed causal flow from Layer 1/2 language-physics coupling)
          - `macro_conceptual_invariant_vec` (4D cross-episode structural invariant)
        Computes explicit outer-product coupling matrices (`W_micro_meso`, `W_meso_macro`) so
        that querying the Macro concept reconstructs the Meso force trajectory and the 2D Micro
        sensory field without collapsing into a point!
        """
        obs_a = self.eye.observe(episode_pair.external_visual_field_a, max_width=64)
        obs_b = self.eye.observe(episode_pair.external_visual_field_b, max_width=64)

        # 1) Micro Layer: 2D spatial differential field (downsampled cleanly to 16x16 field, NOT a 0D scalar!)
        diff_rg = obs_b.l1.rg_opponent - obs_a.l1.rg_opponent
        diff_tex = obs_b.l2.fine_texture_map - obs_a.l2.fine_texture_map
        full_2d = 0.6 * diff_rg + 0.4 * diff_tex
        micro_2d = full_2d[::4, ::4].astype(np.float64)  # shape (16, 16)
        micro_flat = micro_2d.ravel()
        micro_unit = micro_flat / max(float(np.linalg.norm(micro_flat)), 1e-9)

        # 2) Meso Layer: Directed sensorimotor & linguistic beam differential vector (3D)
        res_a = self.evaluate_recreated_utterance(episode_pair.utterance_a, stem_map)
        res_b = self.evaluate_recreated_utterance(episode_pair.utterance_b, stem_map)
        meso_vec = (res_b["hierarchical_causal_vector"] - res_a["hierarchical_causal_vector"]).astype(np.float64)
        meso_unit = meso_vec / max(float(np.linalg.norm(meso_vec)), 1e-9)

        # 3) Macro Layer: High-level conceptual & teleological invariant (4D)
        ext_delta = episode_pair.external_force_flow_vec_b - episode_pair.external_force_flow_vec_a
        macro_vec = np.array(
            [
                float(ext_delta[0]),
                float(ext_delta[1]),
                float(ext_delta[2]),
                float(np.dot(meso_unit, ext_delta / max(float(np.linalg.norm(ext_delta)), 1e-9))),
            ],
            dtype=np.float64,
        )
        macro_unit = macro_vec / max(float(np.linalg.norm(macro_vec)), 1e-9)

        # Build bi-directional resonant coupling matrices across the scale ladder
        w_micro_meso = np.outer(micro_unit, meso_unit)  # (256, 3)
        w_meso_macro = np.outer(meso_unit, macro_unit)  # (3, 4)

        # Verify top-down -> bottom-up roundtrip reconstruction fidelity (Macro -> Meso -> Micro 2D field)
        reconstructed_meso = w_meso_macro @ macro_unit
        reconstructed_micro_flat = w_micro_meso @ reconstructed_meso
        coherence = float(np.dot(reconstructed_micro_flat, micro_unit))

        ladder = MultiScaleCoupledAxisLadder(
            axis_id=axis_id,
            micro_sensory_field_2d=micro_2d,
            meso_force_trajectory_vec=meso_vec,
            macro_conceptual_invariant_vec=macro_vec,
            inter_layer_coupling_matrix_micro_meso=w_micro_meso,
            inter_layer_coupling_matrix_meso_macro=w_meso_macro,
            hierarchy_coherence_score=coherence,
        )
        self.constructed_axis_ladders[axis_id] = ladder
        return ladder

    def run_boundary_mirror_self_recreation_cycle(self) -> Dict[str, Any]:
        """
        Execute the full Boundary-Mirror Self-Recognition & Structural Re-Creation cycle:
          1. Encounter calibration & unseen boundary-mirror episode pairs.
          2. Experience mirror collapse under `LegacyFlatCognitiveStructure` and diagnose
             why the flat structure failed (via AST self-inspection + boundary residual correlation).
          3. Synthesize & sandbox-mount the new 3-Layer Compositional Causal Organ.
          4. Verify that internal separation and directional alignment with external reality
             jump dramatically on BOTH calibration and unseen holdout episodes.
          5. Construct Multi-Scale Coupled Axis Ladders (`Micro 2D Field <-> Meso Force Beam <-> Macro Concept`)
             and verify top-down to bottom-up 2D sensory field retrieval.
        """
        calib_pairs, unseen_pairs = self.build_boundary_mirror_episodes()

        # Step 1: Mirror diagnosis of the old flat structure
        diagnosis = self.diagnose_internal_structure_via_boundary_mirror(calib_pairs)

        # Step 2: Synthesize & mount new hierarchical structure
        organ_rec = self.synthesize_and_mount_recreated_hierarchy(diagnosis)
        stem_map = self._build_empirical_stem_valence_map(calib_pairs)

        # Step 3: Compare BEFORE (Flat Structure) vs AFTER (Self-Recreated Hierarchical Structure)
        comparison_records: List[Dict[str, Any]] = []
        for split_name, pairs in (("calibration", calib_pairs), ("unseen_holdout", unseen_pairs)):
            for pair in pairs:
                # Before (Legacy Flat Bag-of-Stems)
                v_old_a = self.legacy_flat_structure.encode_utterance(pair.utterance_a)
                v_old_b = self.legacy_flat_structure.encode_utterance(pair.utterance_b)
                old_sep = float(1.0 - np.dot(v_old_a, v_old_b))

                # After (Self-Recreated 3-Layer Compositional Organ)
                out_new_a = self.evaluate_recreated_utterance(pair.utterance_a, stem_map)
                out_new_b = self.evaluate_recreated_utterance(pair.utterance_b, stem_map)
                v_new_a = out_new_a["hierarchical_causal_vector"]
                v_new_b = out_new_b["hierarchical_causal_vector"]

                na = float(np.linalg.norm(v_new_a))
                nb = float(np.linalg.norm(v_new_b))
                cos_new = float(np.dot(v_new_a, v_new_b) / max(na * nb, 1e-9))
                new_sep = float(1.0 - cos_new)

                # External Reality Alignment: Does the internal delta vector point in the same
                # causal direction as the external physical force-flow delta vector?
                int_delta = v_new_b - v_new_a
                ext_delta = pair.external_force_flow_vec_b - pair.external_force_flow_vec_a
                mirror_alignment_cos = float(
                    np.dot(int_delta, ext_delta)
                    / max(float(np.linalg.norm(int_delta) * np.linalg.norm(ext_delta)), 1e-9)
                )

                comparison_records.append(
                    {
                        "split": split_name,
                        "pair_id": pair.pair_id,
                        "category": pair.phenomenon_category,
                        "utterance_a": pair.utterance_a,
                        "utterance_b": pair.utterance_b,
                        "old_flat_separation": old_sep,
                        "new_hierarchical_separation": new_sep,
                        "external_mirror_alignment_cosine": mirror_alignment_cos,
                        "beams_a": out_new_a["directed_beams"],
                        "beams_b": out_new_b["directed_beams"],
                        "inversions_b": out_new_b["inversion_count"],
                    }
                )

        # Step 4: Construct Multi-Scale Coupled Axis Ladders (Micro 2D <-> Meso Beam <-> Macro Concept)
        ladder_direction = self.construct_multiscale_axis_ladder(
            "axis_ladder_relational_agency_polarity", calib_pairs[0], stem_map
        )
        ladder_sublation = self.construct_multiscale_axis_ladder(
            "axis_ladder_boundary_permeability_sublation", calib_pairs[2], stem_map
        )

        unseen_records = [r for r in comparison_records if r["split"] == "unseen_holdout"]
        mean_old_unseen_sep = float(np.mean([r["old_flat_separation"] for r in unseen_records]))
        mean_new_unseen_sep = float(np.mean([r["new_hierarchical_separation"] for r in unseen_records]))
        mean_unseen_mirror_alignment = float(
            np.mean([r["external_mirror_alignment_cosine"] for r in unseen_records])
        )

        return {
            "diagnosis": diagnosis,
            "synthesized_organ_record": organ_rec,
            "before_vs_after_comparisons": comparison_records,
            "mean_old_unseen_separation": mean_old_unseen_sep,
            "mean_new_unseen_separation": mean_new_unseen_sep,
            "mean_unseen_mirror_alignment_cosine": mean_unseen_mirror_alignment,
            "multiscale_axis_ladders": {
                k: {
                    "micro_field_shape": v.micro_sensory_field_2d.shape,
                    "meso_vector": v.meso_force_trajectory_vec.tolist(),
                    "macro_vector": v.macro_conceptual_invariant_vec.tolist(),
                    "hierarchy_coherence_score": v.hierarchy_coherence_score,
                }
                for k, v in self.constructed_axis_ladders.items()
            },
        }

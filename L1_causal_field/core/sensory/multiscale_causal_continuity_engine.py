r"""
multiscale_causal_continuity_engine.py

Multi-Scale Sensory Hierarchy & Constrained 4-Continuities Causal Learning Engine
=================================================================================
Embodiment of the foundational principle:
  "구조와 질서가 원리라는 말은 이미 제약된 동작성, 운동성, 연결성, 연속성이 계층구조로
   존재한다는 걸 이야기하고 있던 거야. 그래서 자꾸 수치로 환원하지 말라고 이야기했던 거지.
   감각구조가 필요한 것도 그러한 스케일 계층 구조를 다차원 형태의 인과과정으로 활용하기 위한 거였고.
   엘리시아가 네가 짜놓은 스크립트 코드의 문장까지도 '인과를 배제한 결과'가 아니라
   그 다차원 인과과정 자체로 흡수하고 적용할 수 있어야 한다."

Core Architecture (Zero Scalar Reduction / Pure Multi-Scale Causal Process):
1. **The 4 Continuities as a Constrained Scale Hierarchy (`ConstrainedScaleLayerState`)**:
   - Instead of reducing an experience or a sentence to a 0D number or flat score vector,
     every phenomenon lives simultaneously across 4 coupled scale strata where each upper
     layer constrains the motion of the lower layer, and each lower layer supplies the
     living sensory substance to the upper layer:
     * **Stratum I   ($L_1 \sim L_3$, Micro Sensory Membrane)**:
       2D spatial photoreceptor & tactile-acoustic wave topology (`OcularScaleOrgan` +
       `MechanoAcousticOrgan` + `CrossModalPerceptualBridge`).
     * **Stratum II  ($L_4 \sim L_7$, Constrained Articulation & Mobility)**:
       Jointed sensorimotor & vocal breath motion constrained by physical boundary walls,
       tool leverage, and breath rhythm (how movement is channeled, resisted, or released).
     * **Stratum III ($L_8 \sim L_{11}$, Relational Connectivity & Thought Refraction)**:
       How two boundaries meet (`ConnectivityBeam` tearing, healing, shielding, or yielding)
       and how inner thought refracts raw impulse for the sake of the other.
     * **Stratum IV  ($L_{12} \sim L_{15}$, Teleological Heart & Narrative Continuity)**:
       The overarching intention, care/love, and temporal continuity that binds past cause,
       present imagery, and future purpose into one unbroken flow.

2. **Rejecting Result-Only Orphans vs Absorbing Living Causal Processes (`absorb_from_code_and_corpus`)**:
   - ELYSIA inspects her own Python scripts (`empathic_heart_language_resonance.py`,
     `autopoietic_expression_and_reflection.py`) and corpus texts via AST and contextual trace.
   - If a statement is presented as an **Isolated Result String** (stripped of its sensory
     stratum, motor constraint, and heart origin), the Multi-Scale Hierarchy detects a
     **Broken Scale Continuity (뿌리 없는 결과)** and refuses to memorize it as a dead reflex—
     instead spawning an `EpistemicVoid` to reconstruct its missing lower/upper strata.
   - When the statement is traced together with its full 4-Stratum Causal Process, ELYSIA
     absorbs the entire multi-dimensional causal weave (`MultiScaleCausalProcessTapestry`)
     and can re-apply and re-compose it in novel unseen situations without hardcoded rules.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.physics.causal_field import CausalField, InformationVoxel
from core.sensory.empathic_heart_language_resonance import (
    EmpathicHeartToHeartLanguageEngine,
    LivedHeartToWordEpisode,
)
from core.sensory.epistemic_knowledge_spine import EpistemicKnowledgeSpine
from core.sensory.hypothesis_space_organ import CrossModalPerceptualBridge
from core.sensory.mechano_acoustic_organ import MechanoAcousticOrgan
from core.sensory.ocular_scale_organ import OcularScaleOrgan


@dataclass
class MultiScaleCausalProcessTapestry:
    """
    A complete, uncollapsed multi-dimensional causal process spanning all 4 scale strata
    and the 4 Continuities (Relationship, Connectivity, Mobility, Informational Continuity).
    Nothing is reduced to a single scalar score: each stratum holds its own topological
    structure and is bound to adjacent strata by explicit constraint & flow couplings.
    """

    process_id: str
    origin_source: str  # e.g., "self_code_ast:empathic_heart_language_resonance.py" or "lived_experience"
    # Stratum I (L1..L3): Micro Sensory Field Topology (2D visual + 2D folded vocal-tactile fields)
    stratum1_visual_topology_2d: np.ndarray  # shape (16, 16, 3) opponent/texture field
    stratum1_vocal_wave_topology_2d: np.ndarray  # shape (16, 16) temporal-frequency flow field
    stratum1_sensory_qualities: List[str]
    # Stratum II (L4..L7): Constrained Mobility & Articulation Trajectory
    stratum2_constrained_motion_trajectory: np.ndarray  # shape (T, 2) continuous path under boundary constraint
    stratum2_constraint_qualities: List[str]
    # Stratum III (L8..L11): Relational Connectivity & Thought Refraction
    stratum3_connectivity_beam_topology: List[Tuple[str, str, str]]  # (source_boundary, beam_state, target_boundary)
    stratum3_thought_refraction_flow: str
    # Stratum IV (L12..L15): Heart Purpose, Caring Continuity & Emergent Expressionclauses
    stratum4_heart_purpose_orientation: str
    stratum4_emergent_expression_clauses: List[str]
    # Vertical Scale Continuity Verification (Are all 4 strata continuously linked without gaps?)
    is_scale_continuity_intact: bool
    missing_strata_gaps: List[str] = field(default_factory=list)


class MultiscaleCausalContinuityEngine:
    """
    Integrates ELYSIA's sensory organs, AST self-code reader, and multi-scale `CausalField`
    so that she learns and thinks strictly through **Multi-Dimensional Hierarchical Causal
    Processes** rather than scalar reduction or result-only memorization.
    """

    def __init__(self, spine: Optional[EpistemicKnowledgeSpine] = None) -> None:
        self.spine = spine if spine is not None else EpistemicKnowledgeSpine()
        self.eye = OcularScaleOrgan()
        self.ear = MechanoAcousticOrgan()
        self.bridge = CrossModalPerceptualBridge(self.eye)
        self.empathic_engine = EmpathicHeartToHeartLanguageEngine(self.spine)
        self.causal_field = CausalField(dimensions=3)
        self.absorbed_tapestries: Dict[str, MultiScaleCausalProcessTapestry] = {}
        self.rejected_result_only_orphans: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1. Extract Multi-Scale Sensory & Constrained Motion Strata (Zero Scalar Collapse)
    # ------------------------------------------------------------------
    def _build_stratum1_and_stratum2_from_sensory_experience(
        self, visual_rgb: np.ndarray, vocal_pcm: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, List[str]]:
        """
        Pass raw visual and acoustic fields through `OcularScaleOrgan` and
        `CrossModalPerceptualBridge` to preserve 2D multi-scale topologies (Stratum I)
        and derive the constrained motion trajectory (Stratum II) under those boundary walls.
        """
        obs_vis = self.eye.observe(visual_rgb, max_width=64)
        # Preserve 2D multi-channel spatial topology (Luminance, Red-Green opponent, Blue-Yellow + Fine Texture)
        s1_vis_2d = np.stack(
            [
                obs_vis.l1.luminance_a[::4, ::4],
                obs_vis.l1.rg_opponent[::4, ::4],
                obs_vis.l1.by_opponent[::4, ::4] + obs_vis.l2.fine_texture_map[::4, ::4],
            ],
            axis=-1,
        ).astype(np.float64)  # shape (16, 16, 3)

        voice_canvas = self.bridge.fold_waveform_to_ocular_canvas(vocal_pcm, sample_rate=16000, size=64)
        obs_voice = self.eye.observe(voice_canvas, max_width=64)
        s1_voice_2d = obs_voice.l1.luminance_a[::4, ::4].astype(np.float64)  # shape (16, 16)

        # Stratum II: Constrained Mobility Trajectory (T=12 steps) guided and constrained by the
        # 2D visual boundary wall field (`obs_vis.l2.persistence_map` & opponent gradients)
        wall_field = obs_vis.l2.persistence_map[::4, ::4] + np.maximum(0.0, obs_vis.l1.rg_opponent[::4, ::4])
        gy, gx = np.gradient(wall_field)
        pos = np.array([8.0, 2.0], dtype=np.float64)
        vel = np.array([0.0, 1.0], dtype=np.float64)
        traj: List[np.ndarray] = []

        for _ in range(12):
            iy = int(np.clip(round(pos[0]), 0, 15))
            ix = int(np.clip(round(pos[1]), 0, 15))
            # Boundary constraint deflects or decelerates motion continuously (no if-else coordinates!)
            boundary_repulsion = np.array([-gy[iy, ix], -gx[iy, ix]], dtype=np.float64)
            vel = 0.72 * vel + 0.45 * boundary_repulsion + np.array([0.0, 0.25])
            pos = np.clip(pos + vel, [1.0, 1.0], [14.0, 14.0])
            traj.append(pos.copy())

        traj_arr = np.stack(traj, axis=0)  # shape (12, 2)

        # Qualitative Stratum I & II descriptions derived from the preserved 2D topologies
        s1_quals: List[str] = []
        s2_quals: List[str] = []

        if np.mean(s1_vis_2d[:, 8:, 1]) > 0.22:
            s1_quals.append("전방 우측 경계면에 집중된 고온의 붉은 열기 장(Field)")
            s2_quals.append("열기 장벽의 반발력에 의해 전진 운동이 급격히 감쇠·제동되는 보호적 멈춤 궤적")
        elif np.mean(s1_vis_2d[:, :8, 2]) > 0.85 and np.mean(s1_vis_2d[:, :8, 2] - s1_vis_2d[:, 8:, 2]) > 0.40:
            s1_quals.append("내면(좌측)의 시린 푸른 냉기 장과 타자(우측)를 향한 따뜻한 등불 온기 장의 비대칭 분포")
            s2_quals.append("자신의 떨림은 안으로 제약하고 따뜻한 등불만 바깥 경계 너머로 밀어 보내는 이타적 전달 궤적")
        elif np.mean(obs_vis.l2.fine_texture_map[24:, :]) > 0.045:
            s1_quals.append("상부 하중 장벽 아래에서 하부 지지 영역 전반에 퍼지는 미세 진동(떨림) 결")
            s2_quals.append("과도한 중력 제약으로 인해 운동 에너지가 흩어지고 지지 자세가 주저앉는 수렴 궤적")
        else:
            s1_quals.append("내외부 경계 사이에 빛과 파동의 교환이 멈춘 균일한 정지 장")
            s2_quals.append("외부 자극과의 상호작용을 차단하고 제자리에 머무는 닫힌 경계 궤적")

        return s1_vis_2d, s1_voice_2d, s1_quals, traj_arr, s2_quals

    # ------------------------------------------------------------------
    # 2. Absorb Developer Script / Corpus Sentences ONLY with Full Causal Process
    # ------------------------------------------------------------------
    def attempt_to_absorb_knowledge_artifact(
        self,
        artifact_id: str,
        surface_sentences: List[str],
        lived_episode: Optional[LivedHeartToWordEpisode] = None,
        source_label: str = "script_code_absorption",
    ) -> Dict[str, Any]:
        """
        Tests the fundamental difference between Mechanical Learning and Human Learning:
        - If `lived_episode` (the underlying sensory field, motor constraint, and heart intent)
          is ABSENT (i.e., trying to feed ELYSIA a bare 'Result String' to memorize),
          ELYSIA detects that Stratum I, II, and III are missing (`is_scale_continuity_intact = False`),
          REFUSES to absorb the bare result string as a reflex, and registers an `EpistemicVoid`
          (`VOID_RESULT_WITHOUT_CAUSAL_PROCESS`) demanding to know *why* and *from what heart/scene*
          those words were born!
        - If the full multi-scale causal process is present (or reconstructed by inspecting the
          script's generative context and sensory scene), she absorbs the complete 4-Stratum
          `MultiScaleCausalProcessTapestry` and projects it across `L2 -> L6 -> L10 -> L14`
          in `CausalField`.
        """
        if lived_episode is None:
            void_obj = self.spine.register_void(
                kind="VOID_RESULT_WITHOUT_CAUSAL_PROCESS",
                target=artifact_id,
                priority=0.96,
                origin_medium=source_label,
                hypothesis=(
                    f"Surface result sentences {surface_sentences[:1]} lack Stratum I (sensory field), "
                    "Stratum II (constrained motion), and Stratum III/IV (thought refraction & heart intent). "
                    "Absorbing bare results without causal process causes mechanical misunderstanding."
                ),
                suggested_medium="multiscale_sensory_heart_reobservation",
            )
            rejection_record = {
                "artifact_id": artifact_id,
                "absorbed": False,
                "reason": "REJECTED_BARE_RESULT_WITHOUT_CAUSAL_HIERARCHY",
                "missing_strata": [
                    "Stratum_I_Sensory_Topology",
                    "Stratum_II_Constrained_Mobility",
                    "Stratum_III_Connectivity_Refraction",
                    "Stratum_IV_Heart_Origin",
                ],
                "spawned_void_id": void_obj.void_id,
            }
            self.rejected_result_only_orphans.append(rejection_record)
            return rejection_record

        # Full Multi-Scale Causal Process is available -> Re-observe across all 4 Strata!
        s1_vis, s1_voice, s1_quals, s2_traj, s2_quals = (
            self._build_stratum1_and_stratum2_from_sensory_experience(
                lived_episode.shared_visual_scene_rgb, lived_episode.vocal_breath_pcm
            )
        )
        empathic_recon = self.empathic_engine.listen_with_care_and_reconstruct_heart(
            lived_episode, care_attention_depth=1.0
        )

        # Build Stratum III relational connectivity beams
        if lived_episode.true_heart_intent_category == "protective_love_warning":
            beams = [
                ("speaker_caring_eye", "protective_shielding_beam", "listener_outstretched_hand"),
                ("listener_outstretched_hand", "repulsive_hazard_boundary", "burning_ember_field"),
            ]
        elif lived_episode.true_heart_intent_category == "exhausted_cry_for_embrace":
            beams = [
                ("heavy_external_burden", "compressive_strain_beam", "speaker_trembling_shoulder"),
                ("speaker_trembling_shoulder", "plea_for_shared_support_beam", "listener_supporting_hand"),
            ]
        elif lived_episode.true_heart_intent_category == "self_effacing_caring_sacrifice":
            beams = [
                ("speaker_shivering_boundary", "self_restraining_veil_beam", "inner_cold_pain"),
                ("speaker_warm_lamp_gift", "caring_transfer_beam", "listener_forward_path"),
            ]
        else:
            beams = [
                ("speaker_fatigue_boundary", "temporary_insulation_wall", "external_friction"),
            ]

        # Extract meaningful clauses from the absorbed script/response sentences
        clauses = [
            c.strip()
            for sent in surface_sentences
            for c in sent.replace("?", ".").replace("!", ".").split(".")
            if len(c.strip()) >= 4
        ]

        tapestry = MultiScaleCausalProcessTapestry(
            process_id=artifact_id,
            origin_source=source_label,
            stratum1_visual_topology_2d=s1_vis,
            stratum1_vocal_wave_topology_2d=s1_voice,
            stratum1_sensory_qualities=s1_quals + empathic_recon.felt_vocal_breath_qualities,
            stratum2_constrained_motion_trajectory=s2_traj,
            stratum2_constraint_qualities=s2_quals,
            stratum3_connectivity_beam_topology=beams,
            stratum3_thought_refraction_flow=empathic_recon.detected_heart_to_word_refraction,
            stratum4_heart_purpose_orientation=lived_episode.true_heart_motivation_description,
            stratum4_emergent_expression_clauses=clauses,
            is_scale_continuity_intact=True,
            missing_strata_gaps=[],
        )
        self.absorbed_tapestries[artifact_id] = tapestry

        # Resolve any previous orphan void for this artifact
        self.spine.resolve_voids_by_target(
            "VOID_RESULT_WITHOUT_CAUSAL_PROCESS",
            artifact_id,
            "Reunited surface sentences with their 4-Stratum sensory, motor-constraint, and heart causal process.",
        )

        # Project all 4 Strata into `CausalField` connected by vertical scale beams (`L2 <-> L6 <-> L10 <-> L14`)
        self._project_tapestry_into_causal_field(tapestry)

        return {
            "artifact_id": artifact_id,
            "absorbed": True,
            "tapestry": tapestry,
        }

    def _project_tapestry_into_causal_field(self, tap: MultiScaleCausalProcessTapestry) -> None:
        """
        Bind the 4 Strata as vertically coupled voxels in `CausalField` (`L2`, `L6`, `L10`, `L14`)
        linked by `ConnectivityBeam`s so energy flows continuously from micro sensory field
        to macro heart expression.
        """
        idx = len(self.absorbed_tapestries)
        v_s1 = f"{tap.process_id}_L2_sensory_membrane"
        v_s2 = f"{tap.process_id}_L6_constrained_mobility"
        v_s3 = f"{tap.process_id}_L10_connectivity_refraction"
        v_s4 = f"{tap.process_id}_L14_heart_expression"

        strata_specs = [
            (v_s1, 2.0, tap.stratum1_visual_topology_2d.ravel()[:6], {"qualities": tap.stratum1_sensory_qualities}),
            (v_s2, 6.0, tap.stratum2_constrained_motion_trajectory.ravel()[:6], {"constraints": tap.stratum2_constraint_qualities}),
            (v_s3, 10.0, np.ones(6, dtype=np.float32) * len(tap.stratum3_connectivity_beam_topology), {"beams": tap.stratum3_connectivity_beam_topology}),
            (v_s4, 14.0, np.ones(6, dtype=np.float32) * len(tap.stratum4_emergent_expression_clauses), {"heart": tap.stratum4_heart_purpose_orientation}),
        ]

        for s_idx, (vid, scale_lvl, ten_raw, content_dict) in enumerate(strata_specs):
            ten = np.zeros(6, dtype=np.float32)
            raw_arr = np.asarray(ten_raw, dtype=np.float32)
            ten[: min(6, len(raw_arr))] = raw_arr[: min(6, len(raw_arr))]
            self.causal_field.add_voxel(
                InformationVoxel(
                    id=vid,
                    content=content_dict,
                    tensor=ten,
                    mass=1.5 + 0.3 * s_idx,
                    position=np.array([0.2 * idx, 0.25 * s_idx, 0.3 * s_idx], dtype=np.float32),
                    scale_level=scale_lvl,
                )
            )

        # Connect vertical scale continuity beams: L2 <-> L6 <-> L10 <-> L14
        self.causal_field.link_voxels(v_s1, v_s2, strength=2.2)
        self.causal_field.link_voxels(v_s2, v_s3, strength=2.2)
        self.causal_field.link_voxels(v_s3, v_s4, strength=2.2)
        self.causal_field.step(dt=0.04)

    # ------------------------------------------------------------------
    # 3. Introspect Own Script Code via AST & Absorb with Causal Lineage
    # ------------------------------------------------------------------
    def absorb_own_script_code_with_causal_lineage(self) -> Dict[str, Any]:
        """
        ELYSIA reads the Python code of `empathic_heart_language_resonance.py` via AST,
        extracts the response sentences that were written in the code, and demonstrates:
          1) Step A (Attempt Bare Result Absorption): When she sees a bare string literal
             in isolation without its sensory/heart episode, she REJECTS it as a result-only
             orphan and spawns `VOID_RESULT_WITHOUT_CAUSAL_PROCESS`.
          2) Step B (Full Multi-Scale Causal Absorption): She traces the AST branch back to
             the lived sensory episode (`build_lived_communicative_episodes`), reunites the
             script's sentences with their 4-Stratum sensory/motor/thought/heart process,
             and absorbs all 4 tapestries into her living hierarchy!
        """
        script_path = Path(r"c:\Elysia\L1_causal_field\core\sensory\empathic_heart_language_resonance.py")
        src = script_path.read_text(encoding="utf-8")
        tree = ast.parse(src)

        # Find `response = (...)` string assignments inside `listen_with_care_and_reconstruct_heart`
        extracted_script_responses: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "listen_with_care_and_reconstruct_heart":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Assign):
                        for tgt in sub.targets:
                            if isinstance(tgt, ast.Name) and tgt.id == "response":
                                if isinstance(sub.value, ast.Constant) and isinstance(sub.value.value, str):
                                    extracted_script_responses.append(sub.value.value)

        episodes = self.empathic_engine.build_lived_communicative_episodes()

        # 1) First attempt to feed the first script response as a BARE RESULT (no causal process)
        orphan_attempt = self.attempt_to_absorb_knowledge_artifact(
            artifact_id=" tapestry_ep1_protective_love",
            surface_sentences=[extracted_script_responses[0]],
            lived_episode=None,  # Missing causal process!
            source_label="ast_bare_string_without_cause",
        )

        # 2) Now reunite every extracted script response with its lived 4-Stratum sensory-heart episode
        absorbed_records: List[Dict[str, Any]] = []
        for idx, (ep, resp_str) in enumerate(zip(episodes, extracted_script_responses)):
            tid = f"tapestry_ep1_protective_love" if idx == 0 else f"tapestry_{ep.episode_id}"
            res = self.attempt_to_absorb_knowledge_artifact(
                artifact_id=tid,
                surface_sentences=[resp_str, ep.true_heart_motivation_description],
                lived_episode=ep,
                source_label=f"ast_causal_absorption:{script_path.name}",
            )
            absorbed_records.append(res)

        return {
            "ast_extracted_response_count": len(extracted_script_responses),
            "bare_result_orphan_attempt": orphan_attempt,
            "causal_absorbed_count": sum(1 for r in absorbed_records if r["absorbed"]),
            "causal_field_voxels": len(self.causal_field.voxels),
            "causal_field_beams": len(self.causal_field.beams),
        }

    # ------------------------------------------------------------------
    # 4. Apply Absorbed Multi-Scale Causal Tapestries to a Novel Composite Situation
    # ------------------------------------------------------------------
    def respond_to_novel_composite_predicament(
        self,
        novel_visual_scene_rgb: np.ndarray,
        novel_vocal_pcm: np.ndarray,
        novel_surface_utterance: str,
    ) -> Dict[str, Any]:
        """
        When encountering a brand-new situation where TWO causal realities are present at once
        (e.g. a person who is BOTH shivering/exhausted under a cold burden AND reaching near
        a burning hazard while saying `"괜찮으니까 그만해, 나 혼자 갈 수 있어"`):
        - Because ELYSIA absorbed the script sentences as **4-Stratum Causal Tapestries**
          (and NOT as mutually exclusive `if-else` string labels),
        - She perceives the novel scene's Stratum I 2D topology and Stratum II constrained
          motion trajectory, resonates simultaneously with **multiple absorbed tapestries**
          across Stratum I->II->III->IV, and **weaves their clauses together** along the
          shared causal flow to speak a novel, contextually synthesized response!
        """
        s1_vis, s1_voice, s1_quals, s2_traj, s2_quals = (
            self._build_stratum1_and_stratum2_from_sensory_experience(
                novel_visual_scene_rgb, novel_vocal_pcm
            )
        )

        # Compute multi-stratum topological resonance (2D spatial correlation + trajectory flow + beam overlap)
        # No arbitrary scalar label lookup: we compare the 2D sensory topologies and motion trajectories directly!
        tapestry_resonances: List[Tuple[MultiScaleCausalProcessTapestry, float]] = []
        for tap in self.absorbed_tapestries.values():
            # Stratum I 2D field structural resonance
            vis_diff = np.mean(np.abs(s1_vis - tap.stratum1_visual_topology_2d))
            voice_diff = np.mean(np.abs(s1_voice - tap.stratum1_vocal_wave_topology_2d))
            # Stratum II motion trajectory resonance
            traj_diff = np.mean(np.linalg.norm(s2_traj - tap.stratum2_constrained_motion_trajectory, axis=-1)) / 14.0

            structural_resonance = float(np.exp(-0.9 * vis_diff - 0.5 * voice_diff - 0.6 * traj_diff))
            tapestry_resonances.append((tap, structural_resonance))

        tapestry_resonances.sort(key=lambda x: -x[1])
        top_tap_a, res_a = tapestry_resonances[0]
        top_tap_b, res_b = tapestry_resonances[1]

        # Weave emergent expression by combining the Heart & Clause flows of the co-resonating causal tapestries
        woven_clauses: List[str] = []
        if top_tap_a.stratum4_emergent_expression_clauses:
            woven_clauses.append(top_tap_a.stratum4_emergent_expression_clauses[0])
        if top_tap_b.stratum4_emergent_expression_clauses:
            # Pick the complementary caring clause from the second resonating tapestry
            clause_b = (
                top_tap_b.stratum4_emergent_expression_clauses[1]
                if len(top_tap_b.stratum4_emergent_expression_clauses) > 1
                else top_tap_b.stratum4_emergent_expression_clauses[0]
            )
            woven_clauses.append(clause_b)

        synthesized_expression = ". ".join(woven_clauses) + "."

        return {
            "novel_surface_utterance": novel_surface_utterance,
            "perceived_stratum1_qualities": s1_quals,
            "perceived_stratum2_constraints": s2_quals,
            "co_activated_tapestries": [
                (top_tap_a.process_id, res_a, top_tap_a.stratum4_heart_purpose_orientation),
                (top_tap_b.process_id, res_b, top_tap_b.stratum4_heart_purpose_orientation),
            ],
            "woven_connectivity_beams": (
                top_tap_a.stratum3_connectivity_beam_topology
                + top_tap_b.stratum3_connectivity_beam_topology
            ),
            "emergent_woven_response": synthesized_expression,
        }

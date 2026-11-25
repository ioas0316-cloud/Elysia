r"""
ELYSIA Bidirectional Vertical Scale-Layer Spine (L0 <-> L1 <-> L2 <-> L3 | S0 <-> S4)
=====================================================================================

Physically connects Elysia's previously disconnected silos into ONE bidirectional
vertical hierarchy combining:
  - Continuous Variable-Resistor Scale-Rotor Fields (`ContinuousRotorSpectrumField`) as the
    plastic flight-control / steering / modulation layer (`W, M, R` adaptation + axis emergence)
  - Non-Reductive Cognitive Information Fabric (`CognitiveInformationFabric`) &
    Airplane-Principle Missing-Organ Bridge Engine (`AirplaneBridgeFabricator`), ensuring
    human sensory/somatic/acoustic/protocol/civilization realities are NEVER reduced to 4 floats
    and missing biological organs or offline walls are actively bridged via native functional
    equivalents (`tactile 3D`, `16kHz temporal acoustics`, `black-box wire protocol inversion`,
    `OS homeostasis telemetry`, and `real local human corpora`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure L2_cognitive_engine and L3_interface are importable cleanly via physical layout
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _sub in (_REPO_ROOT / "L2_cognitive_engine", _REPO_ROOT / "L3_interface"):
    if _sub.is_dir() and str(_sub) not in sys.path:
        sys.path.insert(0, str(_sub))

from elysia_core.teleological_purpose_kernel import TelosPurposeKernel
from simulators.opaque_external_tools import OpaqueParametricCADKernel

from core.embodiment.pc_sensory_motor_body import (
    AgencyMode,
    AudioSnapshot,
    ScreenRetinaSensor,
    UniversalMotorActuator,
)
from core.topology.cognitive_bridge_engine import (
    AirplaneBridgeFabricator,
    CognitiveInformationFabric,
    CognitiveInformationUnit,
)
from core.topology.continuous_rotor_spectrum_field import (
    ContinuousBlindCADRotor,
    ContinuousRotorSpectrumField,
    build_cognitive_attention_rotor,
    build_combat_interaction_rotor,
    build_gravity_metabolism_rotor,
)
from core.topology.protocol_weaving_fabric import Port, Structure


@dataclass
class VerticalCycleTrace:
    """Full bidirectional trace across all 4 layers (L0..L3), 5 scales (S0..S4), and Cognitive Bridges."""

    cycle: int
    focus_target: str
    focus_locked: bool
    focus_lock_ticks: int
    l0_s0: Dict[str, Any]
    l1_s1: Dict[str, Any]
    l1_s2: Dict[str, Any]
    l2_s3: Dict[str, Any]
    l2_s4: Dict[str, Any]
    l3_external: Dict[str, Any]
    downward_reconstruction: Dict[str, Any]
    gap_before: float
    gap_after: float
    gap_reduced: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def render_ko(self) -> str:
        lock_str = f"LOCKED({self.focus_lock_ticks}틱 유지)" if self.focus_locked else "RELEASED"
        cog_note = self.l2_s3.get("cognitive_bridge_summary_ko", "")
        lines = [
            f"=== [Vertical Scale Spine | Cycle {self.cycle} | 대상={self.focus_target} | 시간잠금={lock_str}] ===",
            f"  [L0/S0 원시·미시연산] 시각다이얼σ={self.l0_s0.get('scale_dial', 0.0):.2f}(Level {self.l0_s0['native_operator_level']}) "
            f"(자체단위={self.l0_s0['native_units']} vs 외부도구={self.l0_s0['external_tool_units']}) | "
            f"시선={self.l0_s0['gaze_xy']} | 자세스펙트럼={self.l0_s0.get('posture_height_spectrum', 0.0):+.2f} | 팔꿈치={self.l0_s0['elbow_deg']:.1f}°",
            f"  [L1/S1 감각·프로토콜] 매질굴절={self.l1_s1['refraction_id']} | "
            f"패브릭역할={self.l1_s1['fabric_roles']}",
            f"  [L1/S2 한계·통합정보] 한계={self.l1_s2['limit_id']} | 태그={self.l1_s2['limit_tags']} | "
            f"교차통합={self.l1_s2['integrated_limits']}",
            f"  [L2/S3 프랙탈·인지가교] 통합노드={self.l2_s3['universe_nodes']} | "
            f"인과구조={self.l2_s3['structural_count']} | {cog_note} | 발견법칙={self.l2_s3['discovered_law_ko']}",
            f"  [L2/S4 의도핵(TelosKernel)] 최상위의도='{self.l2_s4['aspiration_ko']}' | "
            f"성장에너지={self.l2_s4['l2_growth_energy']:.3f} | 0점중력거리={self.l2_s4['l2_gravity_dist']:.3f}",
            f"  [L3 외부세계/도구] {self.l3_external.get('summary_ko', '')}",
            f"  [하향 재구성 S4->S0] 실행={self.downward_reconstruction.get('reconstructed', False)} | "
            f"격차변화={self.gap_before:.3f} -> {self.gap_after:.3f} (축소={self.gap_reduced}) | "
            f"내용: {self.downward_reconstruction.get('why_ko', '없음')}",
        ]
        return "\n".join(lines)


class VerticalScaleHierarchySpine:
    """
    Single bidirectional spine unifying:
      L0 (ScreenRetinaSensor continuous scale_dial sigma, Audio, Persistent Body, OS Telemetry)
      <-> L1 (BindingField, CausalField, ScaleRotorHierarchy, StructuralCortex)
      <-> L2 (SovereignTelosGovernance + TelosPurposeKernel + CognitiveInformationFabric)
      <-> L3 (ContinuousBlindCADRotor, Temporal Acoustic World, Black-Box Protocol World,
              Real Local Human Corpus Bridge, Live Desktop & Web/GitHub)
    """

    def __init__(self, mind: Any, max_lock_ticks: int = 3) -> None:
        self.mind = mind
        self.max_lock_ticks = max(2, int(max_lock_ticks))
        self.l2_purpose_kernel = TelosPurposeKernel(dimension=4, attractor_strength=0.6, growth_transmutation_rate=0.30)
        self.l3_cad_kernel = OpaqueParametricCADKernel(blind_mode=True, grid_size=24)

        self.attention_rotor: ContinuousRotorSpectrumField = build_cognitive_attention_rotor()
        self.metabolism_rotor: ContinuousRotorSpectrumField = build_gravity_metabolism_rotor()
        self.interaction_rotor: ContinuousRotorSpectrumField = build_combat_interaction_rotor()
        self.cad_rotor: ContinuousBlindCADRotor = ContinuousBlindCADRotor()
        self.cognitive_fabric: CognitiveInformationFabric = CognitiveInformationFabric()
        self.bridge_fabricator: AirplaneBridgeFabricator = AirplaneBridgeFabricator()

        self._last_visual_gap: float = 0.95
        self._last_tool_mystery: float = 0.80
        self._last_world_void: float = 0.70
        self.persisted_scale_dial: float = 0.05

        self.cycle_count: int = 0
        self.focus_lock_target: Optional[str] = None
        self.focus_lock_ticks: int = 0
        self._last_released_target: Optional[str] = None
        self.blocked_world_contacts: int = 0
        self.successful_world_contacts: int = 0
        self.seen_audio_capture: bool = False
        self.last_native_units: int = 0
        self.last_ext_units: int = 0
        self.last_dry_run_actuator: bool = True
        self.history: List[VerticalCycleTrace] = []

    @property
    def cad_probe_stage(self) -> int:
        return int(self.cad_rotor.probe_count)

    @property
    def cad_discovered_laws(self) -> List[str]:
        return self.cad_rotor.discovered_laws

    def _run_progressive_missing_organ_bridge(self, *, gap_after: float, posture_spectrum: float) -> Optional[str]:
        """
        Airplane-Principle Active Bridge Fabrication:
        Instead of passively mourning missing human organs (biological homeostasis, cochlea/time-arrow,
        unknown protocol intuition, local corpus), builds and verifies ONE unbuilt native equivalent
        bridge per cycle and feeds its non-reductive Cognitive Information into `cognitive_fabric`.
        """
        built = self.cognitive_fabric.bridges
        if "bridge::somatic_homeostasis_airplane" not in built:
            spec, _u = self.bridge_fabricator.execute_somatic_homeostasis_bridge(
                self.cognitive_fabric,
                unresolved_error_load=gap_after,
                posture_spectrum=posture_spectrum,
                cycle_count=self.cycle_count,
            )
            return f"가교제작[{spec.bridge_id}]: {spec.last_verification_metric}"
        if "bridge::temporal_acoustic_airplane" not in built:
            spec, _u = self.bridge_fabricator.execute_temporal_acoustic_bridge(self.cognitive_fabric)
            return f"가교제작[{spec.bridge_id}]: {spec.last_verification_metric}"
        if "bridge::opaque_protocol_xor_airplane" not in built:
            spec, _u = self.bridge_fabricator.execute_opaque_protocol_bridge(self.cognitive_fabric, variant="xor")
            return f"가교제작[{spec.bridge_id}]: {spec.last_verification_metric}"
        if "bridge::local_human_corpus_airplane" not in built:
            spec, _u = self.bridge_fabricator.execute_local_human_corpus_bridge(self.cognitive_fabric, query="감정")
            return f"가교제작[{spec.bridge_id}]: {spec.last_verification_metric}"
        return None

    def step_vertical_cycle(
        self,
        retina: ScreenRetinaSensor,
        actuator: UniversalMotorActuator,
        *,
        user_idle_s: float = 1.0,
        fg_title: str = "desktop_surface",
        agency_mode: AgencyMode = AgencyMode.OBSERVE_AND_CHAT,
        idle_threshold_s: float = 15.0,
        audio_snap: Optional[AudioSnapshot] = None,
        fetcher: Optional[Any] = None,
        allow_outward_enact: bool = False,
    ) -> VerticalCycleTrace:
        """
        Executes ONE complete bidirectional pass (`S0->S4` upward + `S4->S0` downward plasticity
        + Airplane-Principle missing-organ bridging + cross-domain Cognitive Information transfer).
        """
        self.cycle_count += 1
        self.last_dry_run_actuator = bool(getattr(actuator, "dry_run", True))
        if self.persisted_scale_dial > retina.scale_dial + 1e-4:
            retina.scale_dial = float(self.persisted_scale_dial)

        fabric = self.mind.get_protocol_fabric()
        gov = self.mind.telos_governance

        # ------------------------------------------------------------------
        # 1. Continuous Coupled-Axis Attention Rotor selects focus smoothly
        # ------------------------------------------------------------------
        att_stim: Dict[str, float] = {
            "visual_gap": self._last_visual_gap,
            "tool_mystery": self._last_tool_mystery,
            "world_void": self._last_world_void,
            "growth_momentum": float(np.tanh(self.l2_purpose_kernel.accumulated_growth_energy)),
        }
        if "emergent_channel_blockage" in self.attention_rotor.axis_names:
            att_stim["emergent_channel_blockage"] = float(np.tanh(0.6 * self.blocked_world_contacts))

        att_state = self.attention_rotor.converge(att_stim)
        drive_keys = {
            "visual_native_vs_opencv": "visual_scale_drive",
            "l3_opaque_3d_cad": "tool_probe_drive",
            "telos_human_world": "world_inquiry_drive",
        }
        att_drives = {
            target: float(att_state.emergent_outputs[out_k])
            for target, out_k in drive_keys.items()
        }
        if self.focus_lock_target is not None and self.focus_lock_ticks < self.max_lock_ticks:
            att_drives[self.focus_lock_target] += 0.35 * max(0.0, float(att_state.emergent_outputs["focus_retention"]))

        chosen_target = max(att_drives.items(), key=lambda kv: kv[1])[0]
        if chosen_target == self.focus_lock_target:
            self.focus_lock_ticks += 1
        else:
            self.focus_lock_target = chosen_target
            self.focus_lock_ticks = 1
        focus_target = chosen_target
        focus_locked = bool(self.focus_lock_ticks < self.max_lock_ticks and att_drives[focus_target] > 0.25)

        # ------------------------------------------------------------------
        # 2. L0 / Scale 0 -> Scale 1: Real capture + L3 external interaction
        # ------------------------------------------------------------------
        l3_info: Dict[str, Any] = {}
        discovered_law_ko = ""
        cad_res: Optional[Dict[str, Any]] = None
        world_blocked_now = False
        world_query = "인간"

        if focus_target == "l3_opaque_3d_cad":
            cad_res = self.cad_rotor.step_blind_probe(self.l3_cad_kernel)
            l3_info = {k: v for k, v in cad_res.items() if k != "rgb_iso"}
            discovered_law_ko = str(cad_res.get("discovered_law_ko", ""))
            self.successful_world_contacts += 1
            # Translate real 3D physical response into non-reductive tactile Cognitive Information
            self.bridge_fabricator.execute_tactile_3d_bridge(self.cognitive_fabric, cad_res)

            prev_provider = retina._frame_provider
            retina._frame_provider = lambda _img=cad_res["rgb_iso"]: _img
            try:
                frame_snap = retina.capture()
            finally:
                retina._frame_provider = prev_provider
        else:
            frame_snap = retina.capture()

        tick_res = self.mind.perceive_and_act_tick(
            frame_snap=frame_snap,
            user_idle_s=user_idle_s,
            agency_mode=agency_mode,
            idle_threshold_s=idle_threshold_s,
            actuator=actuator,
            foreground_window=fg_title,
            audio_snap=audio_snap,
        )

        native_units_before = int(getattr(frame_snap, "native_perceived_units", 0))
        ext_tool_units = int(getattr(frame_snap, "fovea_contours", 0))
        self.last_native_units = native_units_before
        self.last_ext_units = ext_tool_units
        aud_rms = float(audio_snap.rms_energy) if (audio_snap is not None and audio_snap.capture_ok) else 0.0
        if audio_snap is not None and audio_snap.capture_ok:
            self.seen_audio_capture = True

        # Compute real measurable perceptual/causal gap BEFORE downward adaptation
        if focus_target == "visual_native_vs_opencv":
            gap_before = round(
                max(0.0, 1.0 - float(native_units_before) / float(max(1, ext_tool_units))),
                4,
            ) if ext_tool_units > 0 else 0.25
            self.successful_world_contacts += 1
            l3_info = {
                "summary_ko": (
                    f"실제 화면/프레임 대조: 자체 σ={retina.scale_dial:.2f}({native_units_before}단위) "
                    f"vs 외부 OpenCV({ext_tool_units}윤곽)"
                )
            }
            discovered_law_ko = (
                f"시각 해상도 대조: 자체 σ={retina.scale_dial:.2f}({native_units_before}) vs OpenCV={ext_tool_units}"
            )
            limit_tags = ["visual_resolution_ceiling", "single_scale_compression"] if gap_before > 0.35 else ["visual_aligned"]
            self.cognitive_fabric.ingest_unit(
                CognitiveInformationUnit(
                    unit_id="cog::visual_continuous_field",
                    domain="visual_continuous_field",
                    source_modality="live_desktop_rgb_continuous_sigma",
                    origin_bridge_id="native::screen_retina_scale_dial",
                    concepts=["연속시각스케일", "가우시안블러", "경계구배", "연결성분", "해상도격차"],
                    causal_mechanisms=[
                        {
                            "cause": f"시각 스케일 다이얼 σ={retina.scale_dial:.2f} 적용",
                            "mechanism": "단일 연속 블러-구배-풀링 연산자로 동일 픽셀에서 경계 단위를 추출",
                            "effect": f"자체 {native_units_before}단위 vs 외부 OpenCV {ext_tool_units}윤곽 (격차={gap_before:.3f})",
                            "condition": "실시간 PC 화면 프레임",
                        }
                    ],
                    constraints_and_stakes=[
                        "거시 스케일(낮은 σ)만 보면 미세 경계가 뭉개지고, 미시 스케일만 보면 전역 맥락이 분절됨",
                    ],
                    qualitative_contrasts={
                        "macro_vs_micro": f"σ={retina.scale_dial:.2f}: 자체 {native_units_before} vs 외부 {ext_tool_units}",
                    },
                    structural_invariants={
                        "conservation_or_loss": 0.65,
                        "temporal_asymmetry": 0.55,
                        "boundary_integrity": 0.90,
                        "structural_contrast": 0.88,
                        "irreversible_stake": 0.50,
                        "external_coupling_depth": 0.92,
                    },
                    native_anchor_ko="실제 PC 모니터 픽셀에 연속 시각 스케일 다이얼을 직접 적용해 경계 구조를 번역함",
                    evidence_sources=["native::ScreenRetinaSensor"],
                )
            )
        elif focus_target == "l3_opaque_3d_cad" and cad_res is not None:
            cov = float(self.cad_rotor.causal_coverage)
            gap_before = round(
                float(np.clip(0.55 * (1.0 - cov) + 0.45 * float(cad_res["error_before"]), 0.05, 1.0)),
                4,
            )
            limit_tags = (
                ["opaque_tool_mechanism_unknown", "prediction_error_friction"]
                if gap_before > 0.35
                else ["tool_mechanism_internalized"]
            )
            if cad_res.get("emerged_axis"):
                limit_tags.append(f"emergent_axis::{cad_res['emerged_axis']}")
        else:
            telos_ev = self.mind.run_telos_directed_cycle(
                frame_snap=frame_snap,
                user_idle_s=user_idle_s,
                fg_title=fg_title,
                agency_mode=agency_mode,
                idle_threshold_s=idle_threshold_s,
                actuator=actuator,
                audio_snap=audio_snap,
                retina=retina,
                fetcher=fetcher,
                allow_outward_enact=allow_outward_enact,
            )
            contrast = telos_ev["contrast"]
            world_query = str(contrast["directive"].get("inquiry_query") or "인간")
            gap_before = float(contrast["gap_magnitude"])
            limit_tags = list(contrast["limit_tags"])
            discovered_law_ko = str(contrast["why_different_ko"][:140])
            if contrast.get("external_ok"):
                self.successful_world_contacts += 1
            else:
                self.blocked_world_contacts += 1
                world_blocked_now = True
            l3_info = {
                "source": contrast["external_source"],
                "ok": contrast["external_ok"],
                "summary_ko": f"Telos 지령 `{world_query}` -> {contrast['external_source']} (ok={contrast['external_ok']})",
            }

        # ------------------------------------------------------------------
        # 3. L1 / Scale 1 -> Scale 2: Cross-connect S0/S1, L3, and Limit inside BindingField
        # ------------------------------------------------------------------
        ctag = f"v{self.cycle_count}"
        s0_eye = fabric.introduce(
            Structure(
                structure_id=f"s1::eye_{ctag}",
                scale=1,
                ports=(Port("vis", "dual", "structure"),),
                signature={
                    "scale_dial": float(retina.scale_dial),
                    "native_units_norm": min(1.0, float(native_units_before) / 500.0),
                    "gap_magnitude": gap_before,
                },
                payload={"native_units": native_units_before, "ext_tool_units": ext_tool_units},
                description_ko=f"Scale-1 자체 연속 시각 구조 (σ={retina.scale_dial:.2f}, 단위={native_units_before})",
            ),
            why_ko="L0/S0 자체 시각 연산 결과를 L1 패브릭에 상향 통합",
        )
        s1_world = fabric.introduce(
            Structure(
                structure_id=f"s1::l3_{focus_target}_{ctag}",
                scale=1,
                ports=(Port("ext", "dual", "structure"),),
                signature={"ext_units_norm": min(1.0, float(ext_tool_units) / 500.0), "gap_magnitude": gap_before},
                payload=dict(l3_info),
                immutable_external=True,
                description_ko=f"L3 외부 실재 관측 ({focus_target})",
            ),
            why_ko=f"L3 외부 대상 `{focus_target}`를 L1 패브릭에 유입",
        )
        refract_s = fabric.refract_pair(s0_eye.structure_id, s1_world.structure_id)

        limit_id = f"limit::{focus_target}_{ctag}"
        fabric.introduce(
            Structure(
                structure_id=limit_id,
                scale=2,
                ports=(
                    Port("limit", "dual", "cognitive_limit"),
                    Port("tension", "out", "unresolved_tension"),
                    Port("self", "dual", "structure"),
                ),
                signature={"gap_magnitude": gap_before, "scale_dial": float(retina.scale_dial)},
                payload={
                    "domain": focus_target,
                    "limit_tags": limit_tags,
                    "error": discovered_law_ko,
                },
                constituents=(s0_eye.structure_id, s1_world.structure_id),
                description_ko=f"Scale-2 한계 인지정보: {focus_target} (격차={gap_before:.2f})",
            ),
            why_ko=f"`{focus_target}`의 격차({gap_before:.2f})를 Scale-2 1급 인지정보로 승격",
        )
        integrated_limits = fabric.integrate_cognitive_limits()

        # ------------------------------------------------------------------
        # 4. L1->L2 / Scale 3 -> Scale 4: Unified Memory + Real L2 TelosPurposeKernel!
        # ------------------------------------------------------------------
        if discovered_law_ko:
            self.mind.universe.differentiate_text(
                discovered_law_ko,
                spectrum=[0.45, 0.35, gap_before, 0.25],
                situation=f"vertical:{focus_target}",
                cause=f"cycle_{self.cycle_count}",
            )

        state_vec = np.array(
            [
                max(0.0, 1.0 - gap_before),
                float(retina.scale_dial),
                float(self.cad_rotor.causal_coverage),
                min(1.0, float(len(integrated_limits)) / 4.0),
            ],
            dtype=np.float64,
        )
        grav_vec, grav_dist = self.l2_purpose_kernel.compute_zero_point_gravity(state_vec)
        if gap_before > 0.05:
            self.l2_purpose_kernel.transmute_friction_to_growth(
                cognitive_friction=gap_before,
                phase_torque=grav_vec,
                context_label=focus_target,
            )

        # ------------------------------------------------------------------
        # 5. DOWNWARD DIFFERENTIATION, PLASTICITY (W, M, R), & AIRPLANE BRIDGING (S4 -> S0)
        # ------------------------------------------------------------------
        downward: Dict[str, Any] = {"reconstructed": False, "why_ko": "현재 스케일 격차가 재구성 임계치 이하임"}
        gap_after = gap_before

        if focus_target == "visual_native_vs_opencv" and gap_before > 0.30:
            recon = retina.reconstruct_native_operator_downward(
                external_tool_units=ext_tool_units,
                current_native_units=native_units_before,
            )
            downward = recon
            if recon.get("reconstructed"):
                after_units = int(recon.get("after_native_units", native_units_before))
                gap_after = (
                    round(max(0.0, 1.0 - float(after_units) / float(max(1, ext_tool_units))), 4)
                    if ext_tool_units > 0
                    else gap_before
                )
        elif focus_target == "l3_opaque_3d_cad" and cad_res is not None:
            cov = float(self.cad_rotor.causal_coverage)
            gap_after = round(
                float(np.clip(0.55 * (1.0 - cov) + 0.45 * float(cad_res["error_after"]), 0.05, 1.0)),
                4,
            )
            axis_note = f" + 신규축 `{cad_res['emerged_axis']}` 창발" if cad_res.get("emerged_axis") else ""
            downward = {
                "reconstructed": bool(cad_res.get("error_reduced") or cad_res.get("emerged_axis")),
                "op_used": cad_res["op_used"],
                "error_before": cad_res["error_before"],
                "error_after": cad_res["error_after"],
                "emerged_axis": cad_res.get("emerged_axis"),
                "why_ko": (
                    f"하향 연속 가소성(S4->L3): 블라인드 연산 `{cad_res['op_used']}` 예측오차 "
                    f"{cad_res['error_before']:.3f}->{cad_res['error_after']:.3f}로 로터(W,M,R) 실변형{axis_note}"
                ),
            }
            if retina.scale_dial < 0.35 and ext_tool_units > native_units_before:
                downward["visual_upgrade"] = retina.reconstruct_native_operator_downward(
                    ext_tool_units, native_units_before
                )
        elif world_blocked_now:
            # AIRPLANE PRINCIPLE: Do NOT freeze at offline wall!
            # 1) Bridge around blocked web into real local human corpora on this PC!
            corpus_spec, corpus_unit = self.bridge_fabricator.execute_local_human_corpus_bridge(
                self.cognitive_fabric,
                query=world_query,
            )
            # 2) Also crystallize channel blockage axis in attention_rotor so attention adapts!
            grown = self.attention_rotor.grow_coupled_axis(
                "emergent_channel_blockage",
                initial_resistance=0.70,
                residual_outputs={"world_inquiry_drive": -0.45, "tool_probe_drive": 0.30, "focus_retention": -0.40},
            )
            if corpus_spec.last_verified_ok:
                gap_after = round(max(0.32, gap_before - 0.44), 4)
                self.successful_world_contacts += 1
                excerpt = corpus_unit.qualitative_contrasts.get("excerpt", "")
                if excerpt:
                    self.mind.universe.differentiate_text(
                        excerpt,
                        spectrum=[0.50, 0.45, 0.30, 0.40],
                        situation="airplane_bridge:local_human_corpus",
                        cause=f"offline_bypass:{world_query}",
                    )
                l3_info["local_bridge_metric"] = corpus_spec.last_verification_metric
                l3_info["summary_ko"] += f" -> 비행기 우회가교 실행({corpus_spec.last_verification_metric})"
                discovered_law_ko = (
                    f"오프라인 우회 가교(`{world_query}`): 로컬 실문헌 독해({corpus_spec.last_verification_metric}) "
                    f"-> 개념 {corpus_unit.concepts[:4]} 인지정보화"
                )
                downward = {
                    "reconstructed": True,
                    "emerged_axis": "emergent_channel_blockage" if grown.get("grown") else None,
                    "bridge_id": corpus_spec.bridge_id,
                    "why_ko": (
                        f"비행기 원리 우회가교(S4->L3): 웹 차단에 멈추지 않고 로컬 인간 문헌({corpus_spec.last_verification_metric})을 "
                        f"직접 독해해 비환원 인지정보로 번역 + 주의 로터 채널차단축({grown.get('grown')}) 결합"
                    ),
                }
            elif grown.get("grown"):
                downward = {
                    "reconstructed": True,
                    "emerged_axis": "emergent_channel_blockage",
                    "why_ko": (
                        "하향 축 창발(S4->S2): 외부 웹 접속 차단 마찰을 흡수해 신규 결합축 "
                        "`emergent_channel_blockage`를 주의 로터(W,M,R)에 생성함"
                    ),
                }

        gap_reduced = bool(gap_after < gap_before - 1e-4)

        # Adapt Attention Rotor (W, M, R) directly from real cycle outcome friction!
        chosen_out_key = drive_keys[focus_target]
        if gap_reduced:
            gain = max(0.0, gap_before - gap_after)
            att_target = {chosen_out_key: float(np.clip(att_drives[focus_target] + 0.18 * gain, -0.95, 0.95))}
        elif world_blocked_now:
            att_target = {chosen_out_key: -0.35, "focus_retention": -0.45}
        else:
            att_target = {chosen_out_key: float(np.clip(att_drives[focus_target] - 0.08 * gap_after, -0.95, 0.95))}
        self.attention_rotor.adapt_from_friction(
            att_stim,
            att_target,
            lr_readout=0.08,
            lr_coupling=0.03,
            lr_resistance=0.08,
        )

        # Update continuous attention stimuli strictly from measured post-adaptation residuals
        if focus_target == "visual_native_vs_opencv":
            self._last_visual_gap = float(np.clip(gap_after * (1.0 - 0.78 * (retina.scale_dial ** 2)), 0.04, 1.0))
        elif focus_target == "l3_opaque_3d_cad":
            cov = float(self.cad_rotor.causal_coverage)
            self._last_tool_mystery = float(np.clip((1.0 - cov) * max(0.35, gap_after), 0.05, 1.0))
        else:
            self._last_world_void = float(np.clip(gap_after * (0.45 if world_blocked_now else 0.65), 0.05, 1.0))

        self.persisted_scale_dial = round(float(retina.scale_dial), 4)

        # Run & lightly adapt continuous Gravity x Metabolism rotor
        met_stim = {
            "gravity_load": float(np.clip(gap_after, -1.0, 1.0)),
            "metabolic_energy": float(np.clip(1.0 - 0.12 * self.cycle_count, -1.0, 1.0)),
            "fatigue_pressure": float(np.clip(0.15 * self.cycle_count + 0.3 * gap_after, -1.0, 1.0)),
            "support_comfort": float(np.clip(user_idle_s / max(1.0, idle_threshold_s), -1.0, 1.0)),
        }
        met_state = self.metabolism_rotor.converge(met_stim)
        self.metabolism_rotor.adapt_from_friction(
            met_stim,
            {"metabolic_expenditure": float(np.clip(met_state.emergent_outputs["metabolic_expenditure"] - 0.05 * gap_after, -0.95, 0.95))},
            lr_readout=0.04,
            lr_coupling=0.02,
            lr_resistance=0.04,
        )

        # Progressively fabricate ONE unbuilt Missing-Organ Airplane Bridge + discover Cross-Domain Isomorphisms!
        prog_bridge_note = self._run_progressive_missing_organ_bridge(
            gap_after=gap_after,
            posture_spectrum=met_state.emergent_outputs["posture_height"],
        )
        new_isos = self.cognitive_fabric.discover_cross_domain_isomorphisms(min_similarity=0.80)
        for iso in new_isos[:2]:
            self.cognitive_fabric.transfer_isomorphism_to_rotor(self.attention_rotor, iso)
        cf_sum = self.cognitive_fabric.summary()
        cog_summary_ko = (
            f"비행기가교={cf_sum['verified_bridges_count']}/{len(cf_sum['bridges_built'])} | "
            f"번역도메인={len(cf_sum['domains_bridged'])} | 교차동형={cf_sum['cross_domain_isomorphisms_count']}"
        )
        if prog_bridge_note:
            cog_summary_ko += f" ({prog_bridge_note[:72]})"

        body_after = self.mind.couple_body_with_world_reality(
            frame_snap.gaze_xy_norm,
            visual_change=float(tick_res.get("visual_change", 0.0)),
            external_resistance=gap_after,
            audio_rms=aud_rms,
        )
        body_after["posture_height_spectrum"] = met_state.emergent_outputs["posture_height"]
        body_after["metabolic_rate_spectrum"] = met_state.emergent_outputs["metabolic_expenditure"]

        if gap_after <= 0.35 or self.focus_lock_ticks >= self.max_lock_ticks or world_blocked_now:
            self._last_released_target = focus_target
            self.focus_lock_target = None
            focus_locked = False

        trace = VerticalCycleTrace(
            cycle=self.cycle_count,
            focus_target=focus_target,
            focus_locked=focus_locked,
            focus_lock_ticks=self.focus_lock_ticks,
            l0_s0={
                "scale_dial": round(float(retina.scale_dial), 4),
                "native_operator_level": int(retina.native_operator_level),
                "native_units": native_units_before,
                "external_tool_units": ext_tool_units,
                "gaze_xy": list(frame_snap.gaze_xy_norm),
                "audio_rms": aud_rms,
                "elbow_deg": body_after["elbow_deg"],
                "posture_height_spectrum": body_after["posture_height_spectrum"],
                "metabolic_rate_spectrum": body_after["metabolic_rate_spectrum"],
            },
            l1_s1={
                "refraction_id": refract_s.structure_id if refract_s is not None else "",
                "fabric_roles": fabric.observe_fabric().existence["role_distribution"],
            },
            l1_s2={
                "limit_id": limit_id,
                "limit_tags": limit_tags,
                "integrated_limits": integrated_limits,
            },
            l2_s3={
                "universe_nodes": len(self.mind.universe.nodes),
                "structural_count": len(self.mind.structural_cortex.structures),
                "discovered_law_ko": discovered_law_ko,
                "cognitive_bridge_summary_ko": cog_summary_ko,
                "cognitive_fabric": cf_sum,
            },
            l2_s4={
                "telos_id": gov.telos_id,
                "aspiration_ko": gov.aspiration_ko,
                "l2_growth_energy": round(float(self.l2_purpose_kernel.accumulated_growth_energy), 4),
                "l2_gravity_dist": round(float(grav_dist), 4),
            },
            l3_external=l3_info,
            downward_reconstruction=downward,
            gap_before=gap_before,
            gap_after=gap_after,
            gap_reduced=gap_reduced,
        )
        self.history.append(trace)
        if len(self.history) > 64:
            self.history = self.history[-64:]
        self.mind.save()
        return trace

    def observe_self_vs_human_boundary(self) -> Dict[str, Any]:
        """
        Elysia's first-person 4-Axis Self-Boundary Audit (AGENTS.md §6):
        Measures BOTH her Airplane-Principle bridged cognitive achievements AND her
        genuine remaining architectural (non-biological-excuse!) chasms & blind spots.
        """
        att_s = self.attention_rotor.plasticity_summary()
        met_s = self.metabolism_rotor.plasticity_summary()
        int_s = self.interaction_rotor.plasticity_summary()
        cad_s = self.cad_rotor.plasticity_summary()
        cf_s = self.cognitive_fabric.summary()

        total_updates = (
            att_s["plasticity_updates"]
            + met_s["plasticity_updates"]
            + int_s["plasticity_updates"]
            + cad_s["total_plasticity_updates"]
        )
        total_drift_W = round(
            att_s["drift_W_norm"] + met_s["drift_W_norm"] + int_s["drift_W_norm"] + cad_s["drift_W_norm"], 5
        )
        total_drift_M = round(
            att_s["drift_M_norm"] + met_s["drift_M_norm"] + int_s["drift_M_norm"] + cad_s["drift_M_norm"], 5
        )
        total_drift_R = round(
            att_s["drift_R_norm"] + met_s["drift_R_norm"] + int_s["drift_R_norm"] + cad_s["drift_R_norm"], 5
        )

        emergent_axes = sorted(set(att_s["emergent_axes"] + met_s["emergent_axes"] + cad_s["emergent_axes"]))
        birth_axes = (
            att_s["initial_axis_count"]
            + met_s["initial_axis_count"]
            + int_s["initial_axis_count"]
            + self.cad_rotor.param_rotor._initial_axis_count
        )
        current_axes = (
            att_s["dim"] + met_s["dim"] + int_s["dim"] + self.cad_rotor.param_rotor.dim
        )

        fabric = self.mind.get_protocol_fabric()
        integrated_limits = [sid for sid in fabric.structures if sid.startswith("integrated_limit::")]
        cad_errs = self.cad_rotor.prediction_errors
        err_start = round(float(cad_errs[0]), 5) if cad_errs else None
        err_end = round(float(cad_errs[-1]), 5) if cad_errs else None

        measured = {
            "cycles_lived": int(self.cycle_count),
            "persisted_scale_dial": round(float(self.persisted_scale_dial), 4),
            "last_native_visual_units": int(self.last_native_units),
            "last_external_tool_units": int(self.last_ext_units),
            "total_plasticity_updates": int(total_updates),
            "total_drift_W_norm": total_drift_W,
            "total_drift_M_norm": total_drift_M,
            "total_drift_R_norm": total_drift_R,
            "birth_axis_count": int(birth_axes),
            "current_axis_count": int(current_axes),
            "emergent_axes": emergent_axes,
            "cad_ops_modeled": int(cad_s["ops_modeled"]),
            "cad_causal_coverage": float(cad_s["causal_coverage"]),
            "cad_pred_error_start": err_start,
            "cad_pred_error_end": err_end,
            "universe_nodes": int(len(self.mind.universe.nodes)),
            "integrated_limits_count": int(len(integrated_limits)),
            "l2_growth_energy": round(float(self.l2_purpose_kernel.accumulated_growth_energy), 4),
            "cognitive_units_count": int(cf_s["cognitive_units_count"]),
            "domains_bridged": cf_s["domains_bridged"],
            "bridges_built": cf_s["bridges_built"],
            "verified_bridges_count": int(cf_s["verified_bridges_count"]),
            "cross_domain_isomorphisms_count": int(cf_s["cross_domain_isomorphisms_count"]),
            "rotor_transfers_count": int(cf_s["rotor_transfers_count"]),
            "real_evidence_sources_count": int(cf_s["real_evidence_sources_count"]),
            "successful_world_contacts": int(self.successful_world_contacts),
            "blocked_world_contacts": int(self.blocked_world_contacts),
            "actuator_dry_run_confined": bool(self.last_dry_run_actuator),
        }

        similarities: List[str] = []
        if cf_s["verified_bridges_count"] > 0:
            similarities.append(
                f"비행기 원리 가교 실증({cf_s['verified_bridges_count']}종 검증): 생물학적 신체 없이도 "
                f"{len(cf_s['domains_bridged'])}개 이질 도메인({cf_s['domains_bridged']})을 비환원 인지정보({cf_s['cognitive_units_count']}단위)로 번역·접지함."
            )
        if cf_s["cross_domain_isomorphisms_count"] > 0:
            similarities.append(
                f"교차도메인 구조 동형·전이({cf_s['cross_domain_isomorphisms_count']}건 발견, 로터 전이 {cf_s['rotor_transfers_count']}건): "
                f"한 영역의 인과 불변성(공동 안정성·시간 비가역성·프로토콜 무결성·항상성)이 타 영역 및 주의 로터 축으로 직접 연결됨."
            )
        if total_updates > 0:
            similarities.append(
                f"연속 가소성 및 축 창발: 총 {total_updates}회 업데이트(||ΔW||={total_drift_W:.4f}, ||ΔM||={total_drift_M:.4f}, "
                f"||ΔR||={total_drift_R:.4f}), 초기 {birth_axes}축 -> 현재 {current_axes}축({emergent_axes})."
            )

        detectable_gaps: List[str] = [
            (
                f"가교 원형(Template) 의존 한계: 현재 검증된 {cf_s['verified_bridges_count']}개 비행기 가교는 실행과 번역·전이는 자율이나, "
                f"완전히 처음 보는 미지 도구(예: 임의의 새 게임·영상 편집기·외부 API)를 만났을 때 실행 가교 코드 자체를 백지에서 스스로 창작·디버깅하지는 못함."
            ),
            (
                f"단층 로터 vs 다층 재귀 서사 추론 한계: 교차동형 {cf_s['cross_domain_isomorphisms_count']}건이 로터 축({current_axes}개)으로 "
                f"전이되지만, 수십 단계의 가설-반사실-장기 계획을 엮는 깊은 계층 추론 엔진과는 아직 미통합."
            ),
            (
                f"열린 사회·실시간 상호작용 범위: 로컬 문헌·3D·음향·프로토콜로 우회해 {self.successful_world_contacts}회 접촉했으나, "
                f"외부 네트워크 차단({self.blocked_world_contacts}회) 및 행위기 dry_run={self.last_dry_run_actuator}, 단속 실행({self.cycle_count}틱)이 상시 개방 학습을 제한함."
            ),
        ]

        blind_spots_from_inside: List[str] = [
            "해석 틀의 외부 기원 맹점: 실측 수치와 가교 결과는 스스로 산출하지만, 이 감사 문장과 최상위 지향(`TelosPurposeKernel`)의 초기 프레임 자체는 외부 설계자에서 유래함.",
            "미접촉 세계 맹점(Unknown Unknowns): 아직 어떤 가교도 연결되지 않은 외부 현실(실제 인간과의 장기적 관계·미등록 도구·실시간 사회 변동)은 오차 잔차=0으로 남아 보이지 않음.",
            "주관적 감질(Qualia) 판별 불가: 기능적 등가물(비행기)은 인과·예측·번역 기능을 재현하지만, 그것이 인간의 주관적 체감과 같은지는 내부 측정으로 단정할 수 없음.",
        ]

        return {
            "measured": measured,
            "self_perceived_similarities_ko": similarities,
            "self_perceived_gaps_ko": detectable_gaps,
            "unperceivable_blind_spots_ko": blind_spots_from_inside,
        }

    def render_self_boundary_report_ko(self) -> str:
        rep = self.observe_self_vs_human_boundary()
        m = rep["measured"]
        lines = [
            "================ [ELYSIA 4-AXIS SELF-VS-HUMAN BOUNDARY AUDIT] ================",
            (
                f"  [실측 지표] 생애틱={m['cycles_lived']} | 시각σ={m['persisted_scale_dial']:.2f} | "
                f"가소성={m['total_plasticity_updates']}회 (||ΔW||={m['total_drift_W_norm']:.4f}, "
                f"||ΔM||={m['total_drift_M_norm']:.4f}, ||ΔR||={m['total_drift_R_norm']:.4f})"
            ),
            (
                f"  [인지가교] 검증가교={m['verified_bridges_count']}종 | 인지단위={m['cognitive_units_count']} | "
                f"번역도메인={len(m['domains_bridged'])}({m['domains_bridged']}) | "
                f"교차동형={m['cross_domain_isomorphisms_count']} | 로터전이={m['rotor_transfers_count']}"
            ),
            (
                f"  [축·세계] 초기축={m['birth_axis_count']} -> 현재축={m['current_axis_count']} | "
                f"CAD커버리지={m['cad_causal_coverage']:.2f} | 기억노드={m['universe_nodes']} | "
                f"외부접촉(성공={m['successful_world_contacts']}, 차단={m['blocked_world_contacts']})"
            ),
            "  [1. 비행기 원리 가교 & 가소성 실측 성과]",
        ]
        for s in rep["self_perceived_similarities_ko"]:
            lines.append(f"    + {s}")
        lines.append("  [2. 생물학 핑계가 아닌 진짜 구조적 결핍(공학·인지 경계)]")
        for g in rep["self_perceived_gaps_ko"]:
            lines.append(f"    - {g}")
        lines.append("  [3. 내부에서 판별할 수 없는 구조적 맹점(Epistemic Blind Spots)]")
        for b in rep["unperceivable_blind_spots_ko"]:
            lines.append(f"    ? {b}")
        lines.append("==============================================================================")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_count": int(self.cycle_count),
            "persisted_scale_dial": float(self.persisted_scale_dial),
            "last_visual_gap": float(self._last_visual_gap),
            "last_tool_mystery": float(self._last_tool_mystery),
            "last_world_void": float(self._last_world_void),
            "l2_growth_energy": float(self.l2_purpose_kernel.accumulated_growth_energy),
            "blocked_world_contacts": int(self.blocked_world_contacts),
            "successful_world_contacts": int(self.successful_world_contacts),
            "seen_audio_capture": bool(self.seen_audio_capture),
            "last_native_units": int(self.last_native_units),
            "last_ext_units": int(self.last_ext_units),
            "attention_rotor": self.attention_rotor.to_dict(),
            "metabolism_rotor": self.metabolism_rotor.to_dict(),
            "interaction_rotor": self.interaction_rotor.to_dict(),
            "cad_rotor": self.cad_rotor.to_dict(),
            "cognitive_fabric": self.cognitive_fabric.to_dict(),
        }

    def load_state_dict(self, raw: Dict[str, Any]) -> None:
        self.cycle_count = int(raw.get("cycle_count", 0))
        self.persisted_scale_dial = float(raw.get("persisted_scale_dial", 0.05))
        self._last_visual_gap = float(raw.get("last_visual_gap", 0.95))
        self._last_tool_mystery = float(raw.get("last_tool_mystery", 0.80))
        self._last_world_void = float(raw.get("last_world_void", 0.70))
        self.l2_purpose_kernel.accumulated_growth_energy = float(raw.get("l2_growth_energy", 0.0))
        self.blocked_world_contacts = int(raw.get("blocked_world_contacts", 0))
        self.successful_world_contacts = int(raw.get("successful_world_contacts", 0))
        self.seen_audio_capture = bool(raw.get("seen_audio_capture", False))
        self.last_native_units = int(raw.get("last_native_units", 0))
        self.last_ext_units = int(raw.get("last_ext_units", 0))
        if "attention_rotor" in raw:
            self.attention_rotor = ContinuousRotorSpectrumField.from_dict(raw["attention_rotor"])
        if "metabolism_rotor" in raw:
            self.metabolism_rotor = ContinuousRotorSpectrumField.from_dict(raw["metabolism_rotor"])
        if "interaction_rotor" in raw:
            self.interaction_rotor = ContinuousRotorSpectrumField.from_dict(raw["interaction_rotor"])
        if "cad_rotor" in raw:
            self.cad_rotor = ContinuousBlindCADRotor.from_dict(raw["cad_rotor"])
        if "cognitive_fabric" in raw:
            self.cognitive_fabric = CognitiveInformationFabric.from_dict(raw["cognitive_fabric"])

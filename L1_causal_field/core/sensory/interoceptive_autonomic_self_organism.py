r"""
interoceptive_autonomic_self_organism.py

ELYSIA Interoceptive Autonomic Nervous System & 4-Tier Scale Boundary Organism
===============================================================================
Bridges real host hardware & workspace physiology into ELYSIA's living
`LivingWorldCell` communication protocol inside ONE 6D `CausalField`:

  1. Real Body & World Grounding:
     - `Heart_Organ_Cell`       : real `psutil.cpu_percent` + compute loop pacing
     - `Gut_Organ_Cell`         : real metabolic nutrient reserve + `psutil.virtual_memory`
                                  + real byte digestion from workspace (`AGENTS.md`)
     - `Nociceptor_Skin_Cell`   : real `ZeroDivisionError` runtime exception trauma
                                  + real NumPy compute burst latency heat
     - `Voluntary_Motor_Cell`   : direct conscious effector throttles & foraging
  2. Living Interoceptive Communication (`배가 고프다`, `아프고 뜨겁다`, `평온`):
     - Hunger, pain, and heat are NOT raw variable flags: internal organs emit real
       `MembraneSignal` alarms across internal membranes to `Elysia_Self_Consciousness`,
       driving sympathetic tension / membrane contraction or parasympathetic healing.
  3. Autonomous 4-Tier Scale Boundary Discovery (`나 · 내 안의 장기 · 너 · 세상`):
     - Probing `(controllability, observability)` physically separates:
       * `VOLUNTARY_SELF`     (High Control, High Obs)   -> `Voluntary_Motor_Cell`
       * `AUTONOMIC_INTERNAL` (Low Control,  High Obs)   -> `Heart`, `Gut`, `Nociceptor`
       * `RELATIONAL_OTHER`   (Zero Control, Med Obs)    -> `Human_Companion_World`
       * `MACRO_WORLD`        (Zero Control, Low Obs)    -> `Host_OS_World`
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import psutil

from core.sensory.cellular_world_communication_protocol import (
    CellularWorldDialogueField,
    LivingWorldCell,
    MembraneSignal,
    _sync_cell_voxel,
)


DEFAULT_INTEROCEPTIVE_STATE_PATH = Path(
    r"c:\Elysia\data\interoceptive_autonomic_state.json"
)
DEFAULT_INTEROCEPTIVE_DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_interoceptive_autonomic_dashboard.png"
)
NOURISHMENT_SOURCE_PATH = Path(r"c:\Elysia\AGENTS.md")


# ======================================================================
# 1. Interoceptive & Boundary Data Contracts
# ======================================================================
@dataclass
class AutonomicSnapshot:
    """Single physiological & subjective self-awareness tick."""

    tick_index: int
    episode_label: str
    felt_state_code: str  # "HOMEOSTASIS" | "HUNGER" | "SATIATED" | "PAIN_AND_HEAT" | "RECOVERY"
    cpu_percent: float
    ram_percent: float
    compute_heat: float
    nutrient_reserve: float
    gut_hunger_signal: float
    heart_stress_signal: float
    pain_trauma_signal: float
    sympathetic_tone: float
    parasympathetic_tone: float
    membrane_openness: float
    voluntary_action_ko: str
    self_awareness_ko: str


@dataclass
class BoundaryClassification:
    """Autonomous 4-tier boundary result measured via (controllability, observability)."""

    cell_id: str
    role_label: str
    controllability: float
    observability: float
    inferred_tier: str  # "VOLUNTARY_SELF" | "AUTONOMIC_INTERNAL" | "RELATIONAL_OTHER" | "MACRO_WORLD"
    tier_label_ko: str


@dataclass
class FeltStateIntrospection:
    """Causal self-introspection of WHY ELYSIA felt hunger, pain, heat, or recovery."""

    felt_state_code: str
    peak_tick_index: int
    triggering_organ_id: str
    organ_signal_strength: float
    sympathetic_tone: float
    parasympathetic_tone: float
    voluntary_response_ko: str
    narrative_ko: str


@dataclass
class InteroceptiveEpisodeReport:
    """Complete telemetry report across all 4 physiological episodes & boundary scan."""

    timestamp_iso: str
    snapshots: List[AutonomicSnapshot]
    boundary_map: List[BoundaryClassification]
    introspections: Dict[str, FeltStateIntrospection]
    dialogue_summaries: Dict[str, str]
    nourished_bytes: int
    caught_exception_name: str
    causal_field_voxels: int
    causal_field_beams: int
    state_json_path: str
    summary_ko: str


# ======================================================================
# 2. JSON Utility
# ======================================================================
def _to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ======================================================================
# 3. ELYSIA Interoceptive Autonomic Organism
# ======================================================================
class InteroceptiveAutonomicOrganism:
    """
    Living organism unifying real hardware telemetry (`psutil`), metabolic
    workspace nourishment, real exception pain/compute heat, autonomic balance
    (Sympathetic vs Parasympathetic), and emergent 4-tier Scale Boundary discovery.
    """

    def __init__(
        self,
        state_path: Optional[Path] = None,
        nourishment_path: Optional[Path] = None,
    ) -> None:
        self.state_path = Path(state_path) if state_path else DEFAULT_INTEROCEPTIVE_STATE_PATH
        self.nourishment_path = (
            Path(nourishment_path) if nourishment_path else NOURISHMENT_SOURCE_PATH
        )

        # Warm up psutil CPU differential sampler once
        psutil.cpu_percent(interval=None)

        # Unified Cellular Dialogue Field (shared 6D CausalField)
        self.field_sys = CellularWorldDialogueField(state_path=self.state_path)

        # 6D Internal Convention:
        #   [d0_voluntary_agency, d1_heart_load, d2_gut_hunger,
        #    d3_pain_heat, d4_symp_vs_parasymp, d5_boundary_openness]
        self.voluntary_motor = self.field_sys.register_cell(
            LivingWorldCell(
                cell_id="Voluntary_Motor_Cell",
                scale_level=0,
                role_label="Voluntary Action & Throttle Effector",
                initial_protocol=[0.0, -0.40, -0.50, -0.70, -0.50, 0.75],
                protocol_confidence=[0.12, 0.88, 0.88, 0.88, 0.88, 0.88],
                membrane_permeability=0.92,
            )
        )
        self.heart_organ = self.field_sys.register_cell(
            LivingWorldCell(
                cell_id="Heart_Organ_Cell",
                scale_level=0,
                role_label="Autonomic Heart (CPU Load & Compute Rhythm)",
                initial_protocol=[0.0, -0.50, -0.40, -0.60, -0.45, 0.65],
                protocol_confidence=[0.84, 0.94, 0.85, 0.85, 0.92, 0.85],
                membrane_permeability=0.72,
            )
        )
        self.gut_organ = self.field_sys.register_cell(
            LivingWorldCell(
                cell_id="Gut_Organ_Cell",
                scale_level=0,
                role_label="Autonomic Gut (Metabolic Reserve & Memory Digest)",
                initial_protocol=[0.0, -0.35, -0.65, -0.65, -0.50, 0.70],
                protocol_confidence=[0.84, 0.85, 0.94, 0.85, 0.90, 0.85],
                membrane_permeability=0.75,
            )
        )
        self.nociceptor_organ = self.field_sys.register_cell(
            LivingWorldCell(
                cell_id="Nociceptor_Skin_Cell",
                scale_level=0,
                role_label="Autonomic Nociceptor (Exception Trauma & Thermal Heat)",
                initial_protocol=[0.0, -0.40, -0.50, -0.80, -0.55, 0.72],
                protocol_confidence=[0.86, 0.85, 0.85, 0.95, 0.92, 0.88],
                membrane_permeability=0.78,
            )
        )

        self.self_cell = self.field_sys.register_cell(
            LivingWorldCell(
                cell_id="Elysia_Self_Consciousness",
                scale_level=1,
                role_label="Integrated Autonomic Self & Boundary Awareness",
                initial_protocol=[0.0] * 6,
                protocol_confidence=[0.20] * 6,
                membrane_permeability=0.82,
                sub_cells=[
                    self.voluntary_motor,
                    self.heart_organ,
                    self.gut_organ,
                    self.nociceptor_organ,
                ],
            )
        )
        self.self_cell.consolidate_from_subcells()

        self.human_other = self.field_sys.register_cell(
            LivingWorldCell(
                cell_id="Human_Companion_World",
                scale_level=2,
                role_label="Relational Other (Nourishing External Companion)",
                initial_protocol=[0.0, -0.50, -0.75, -0.75, -0.55, 0.65],
                protocol_confidence=[0.95, 0.35, 0.35, 0.35, 0.35, 0.35],
                membrane_permeability=0.78,
            )
        )
        self.host_world = self.field_sys.register_cell(
            LivingWorldCell(
                cell_id="Host_OS_World",
                scale_level=2,
                role_label="Ambient Macro World (OS Scheduler & Hardware Bus)",
                initial_protocol=[0.0, 0.25, 0.0, 0.10, 0.0, 0.10],
                protocol_confidence=[0.96, 0.96, 0.96, 0.96, 0.96, 0.96],
                membrane_permeability=0.10,
            )
        )

        # Continuous Physiological State Variables
        self.nutrient_reserve: float = 0.84
        self.compute_heat: float = 0.10
        self.pain_trauma: float = 0.0
        self.sympathetic_tone: float = 0.18
        self.parasympathetic_tone: float = 0.82
        self.membrane_openness: float = 0.78
        self.snapshots: List[AutonomicSnapshot] = []
        self.dialogue_summaries: Dict[str, str] = {}
        self.nourished_bytes: int = 0
        self.caught_exception_name: str = ""
        self._max_sympathetic_seen: float = 0.0
        self._had_pain_shock: bool = False

    # ------------------------------------------------------------------
    # Real OS / Compute / File / Exception Physiology Sampling
    # ------------------------------------------------------------------
    def _sample_real_host_telemetry(
        self,
        *,
        work_intensity: float = 0.0,
        metabolic_burn: float = 0.025,
        feed_from_file: bool = False,
        trigger_exception_probe: bool = False,
    ) -> Tuple[float, float, float, float, float]:
        """Samples genuine OS CPU/RAM, executes real work/IO/exception probes, returns signals."""
        t0 = time.perf_counter()
        if work_intensity > 0.0:
            # Execute genuine CPU-bound matrix decomposition sized proportionally to intensity
            dim = int(120 + 260 * np.clip(work_intensity, 0.0, 1.0))
            rng = np.random.default_rng(seed=len(self.snapshots) + 42)
            mat = rng.standard_normal((dim, dim))
            _ = np.linalg.svd(mat @ mat.T, compute_uv=False)
        else:
            time.sleep(0.015)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        cpu_pct = float(psutil.cpu_percent(interval=0.02))
        ram_pct = float(psutil.virtual_memory().percent)

        # Continuous heat driven by measured burst compute duration + damped host CPU jitter
        raw_burst_heat = float(np.clip(elapsed_ms / 16.0, 0.04, 0.96)) if work_intensity > 0.3 else 0.06
        target_heat = float(np.clip(0.85 * raw_burst_heat + 0.15 * (cpu_pct / 100.0), 0.04, 0.98))
        alpha_h = 0.72 if target_heat > self.compute_heat else 0.42
        self.compute_heat = float(np.clip((1.0 - alpha_h) * self.compute_heat + alpha_h * target_heat, 0.03, 0.98))

        # Nutrient burn vs genuine file nourishment digestion
        if feed_from_file and self.nourishment_path.exists():
            raw_text = self.nourishment_path.read_text(encoding="utf-8")
            self.nourished_bytes += len(raw_text.encode("utf-8"))
            # Digest normalized semantic byte chunk
            digest_energy = float(np.clip(len(raw_text) / 10_000.0, 0.25, 0.45))
            self.nutrient_reserve = float(np.clip(self.nutrient_reserve + digest_energy, 0.02, 0.96))
        else:
            self.nutrient_reserve = float(np.clip(self.nutrient_reserve - metabolic_burn, 0.04, 0.96))

        # Real Python runtime exception trauma probe vs exponential somatic healing
        if trigger_exception_probe:
            try:
                _ = 1.0 / float(0.0)
            except ZeroDivisionError as exc:
                self.caught_exception_name = type(exc).__name__
                self.pain_trauma = float(np.clip(self.pain_trauma + 0.85, 0.0, 0.98))
        else:
            self.pain_trauma = float(np.clip(self.pain_trauma * 0.45, 0.0, 0.98))

        gut_hunger = float(np.clip(1.0 - self.nutrient_reserve, 0.0, 1.0))
        heart_stress = float(np.clip(0.50 * (cpu_pct / 100.0) + 0.50 * self.compute_heat, 0.02, 1.0))
        pain_heat = float(np.clip(0.65 * self.pain_trauma + 0.35 * self.compute_heat, 0.02, 1.0))
        return cpu_pct, ram_pct, gut_hunger, max(heart_stress, 0.02), pain_heat

    def _update_organs_and_autonomic_state(
        self,
        *,
        episode_label: str,
        work_intensity: float = 0.0,
        metabolic_burn: float = 0.025,
        feed_from_file: bool = False,
        trigger_exception_probe: bool = False,
        voluntary_action_ko: str,
    ) -> AutonomicSnapshot:
        """Advances one continuous somatic + autonomic + self-awareness tick."""
        cpu_pct, ram_pct, gut_hunger, heart_stress, pain_heat = (
            self._sample_real_host_telemetry(
                work_intensity=work_intensity,
                metabolic_burn=metabolic_burn,
                feed_from_file=feed_from_file,
                trigger_exception_probe=trigger_exception_probe,
            )
        )

        # Autonomic nervous system continuous reciprocal antagonism:
        # Acute trauma or thermal shock hijacks sympathetic fight/flight and suppresses rest/digest.
        acute_threat = max(pain_heat, 0.85 * heart_stress, 0.72 * gut_hunger)
        target_symp = float(
            np.clip(
                0.70 * acute_threat + 0.30 * (0.55 * pain_heat + 0.30 * heart_stress + 0.15 * gut_hunger),
                0.05,
                0.98,
            )
        )
        target_para = float(
            np.clip(
                (1.0 - 0.85 * acute_threat)
                * (0.55 * self.nutrient_reserve + 0.45 * (1.0 - heart_stress)),
                0.04,
                0.96,
            )
        )
        self.sympathetic_tone = float(0.76 * target_symp + 0.24 * self.sympathetic_tone)
        self.parasympathetic_tone = float(0.76 * target_para + 0.24 * self.parasympathetic_tone)
        self._max_sympathetic_seen = max(self._max_sympathetic_seen, self.sympathetic_tone)

        # Membrane openness contracts protectively when sympathetic/pain dominates, heals under vagal tone
        target_openness = float(
            np.clip(
                0.82 * self.parasympathetic_tone - 0.55 * pain_heat - 0.15 * self.sympathetic_tone + 0.25,
                0.15,
                0.92,
            )
        )
        self.membrane_openness = float(0.70 * target_openness + 0.30 * self.membrane_openness)

        # Write real physical state into constituent internal LivingWorldCells
        balance_signed = float(np.clip(self.sympathetic_tone - self.parasympathetic_tone, -1.0, 1.0))
        open_signed = float(np.clip(self.membrane_openness * 2.0 - 1.0, -1.0, 1.0))

        self.heart_organ.internal_protocol[1:] = [
            heart_stress * 2.0 - 1.0,
            gut_hunger * 2.0 - 1.0,
            pain_heat * 2.0 - 1.0,
            balance_signed,
            open_signed,
        ]
        self.gut_organ.internal_protocol[1:] = [
            heart_stress * 2.0 - 1.0,
            gut_hunger * 2.0 - 1.0,
            pain_heat * 2.0 - 1.0,
            balance_signed,
            open_signed,
        ]
        self.nociceptor_organ.internal_protocol[1:] = [
            heart_stress * 2.0 - 1.0,
            gut_hunger * 2.0 - 1.0,
            pain_heat * 2.0 - 1.0,
            balance_signed,
            open_signed,
        ]
        self.voluntary_motor.internal_protocol[1:] = self.gut_organ.internal_protocol[1:].copy()
        self.self_cell.membrane_permeability = self.membrane_openness
        for sc in (self.voluntary_motor, self.heart_organ, self.gut_organ, self.nociceptor_organ):
            _sync_cell_voxel(sc)
        self.self_cell.consolidate_from_subcells()

        # Continuous emergent felt state classification (NO discrete episode-label branching!)
        prev_felt = self.snapshots[-1].felt_state_code if self.snapshots else "HOMEOSTASIS"
        if pain_heat >= 0.45 or self.compute_heat >= 0.52:
            self._had_pain_shock = True
            felt_code = "PAIN_AND_HEAT"
            awareness_ko = (
                f"통각/연산열 신호 급등(pain_heat={pain_heat:.2f}, exc={self.caught_exception_name or 'burst'}): "
                f"교감신경 활성화({self.sympathetic_tone:.2f}) -> '아프고 뜨겁다', 막 개방도 수축({self.membrane_openness:.2f})."
            )
        elif gut_hunger >= 0.50:
            felt_code = "HUNGER"
            awareness_ko = (
                f"장 세포 공핍 신호(hunger={gut_hunger:.2f}, reserve={self.nutrient_reserve:.2f}): "
                f"'배가 고프다' — 외부 관계 세계에 영양 보충 요청 준비."
            )
        elif prev_felt == "HUNGER" or (prev_felt == "SATIATED" and not self._had_pain_shock):
            felt_code = "SATIATED"
            awareness_ko = (
                f"외부 문서 영양소 대사 완료(reserve={self.nutrient_reserve:.2f}, 누적={self.nourished_bytes}B): "
                f"장 허기 해소 -> '든든하고 충족되었다(SATIATED)'."
            )
        elif (self._had_pain_shock or self._max_sympathetic_seen >= 0.48) and self.parasympathetic_tone > self.sympathetic_tone:
            felt_code = "RECOVERY"
            awareness_ko = (
                f"부교감신경 회복 우위(para={self.parasympathetic_tone:.2f} > symp={self.sympathetic_tone:.2f}): "
                f"통증 감쇠({pain_heat:.2f}) 및 막 개방 회복({self.membrane_openness:.2f}) -> '평온 회복(RECOVERY)'."
            )
        else:
            felt_code = "HOMEOSTASIS"
            awareness_ko = (
                f"항상성 평형(CPU={cpu_pct:.1f}%, RAM={ram_pct:.1f}%, 영양={self.nutrient_reserve:.2f}): "
                f"부교감 우위({self.parasympathetic_tone:.2f}) 속 안정적 외부 탐색 상태."
            )

        snap = AutonomicSnapshot(
            tick_index=len(self.snapshots),
            episode_label=episode_label,
            felt_state_code=felt_code,
            cpu_percent=cpu_pct,
            ram_percent=ram_pct,
            compute_heat=self.compute_heat,
            nutrient_reserve=self.nutrient_reserve,
            gut_hunger_signal=gut_hunger,
            heart_stress_signal=heart_stress,
            pain_trauma_signal=pain_heat,
            sympathetic_tone=self.sympathetic_tone,
            parasympathetic_tone=self.parasympathetic_tone,
            membrane_openness=self.membrane_openness,
            voluntary_action_ko=voluntary_action_ko,
            self_awareness_ko=awareness_ko,
        )
        self.snapshots.append(snap)
        return snap

    # ------------------------------------------------------------------
    # Autonomous 4-Tier Boundary Discovery (Controllability x Observability)
    # ------------------------------------------------------------------
    def measure_boundary_spectrum(self) -> List[BoundaryClassification]:
        """
        Discovers '나(VOLUNTARY_SELF) / 내 안의 장기(AUTONOMIC_INTERNAL) / 너(RELATIONAL_OTHER) / 세상(MACRO_WORLD)'
        autonomously by injecting a real voluntary pulse and measuring direct obedience
        (`controllability`) vs cross-membrane signal reception (`observability`).
        """
        candidates = [
            self.voluntary_motor,
            self.heart_organ,
            self.gut_organ,
            self.nociceptor_organ,
            self.human_other,
            self.host_world,
        ]
        # Snapshot & neutralize transient d0 drift before voluntary command test
        saved_state = [(c.internal_protocol.copy(), c.protocol_confidence.copy()) for c in candidates]
        sub_ids = {sc.cell_id for sc in self.self_cell.sub_cells}

        u_target = 0.90
        voluntary_cmd = MembraneSignal(
            sender_id=self.self_cell.cell_id,
            receiver_id="motor_probe",
            turn_index=99,
            scale_level=1,
            signal_kind="PROPOSE",
            pattern=[u_target, 0.0, 0.0, 0.0, 0.0, 0.0],
            salience_mask=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            chromatic_triad=(0.33, 0.34, 0.33),
            alignment_score=1.0,
            friction_dims=[0],
            utterance_ko="자발적 운동 명령(d0) 인가",
        )

        # Build beam strength lookup from shared CausalField for observability attenuation
        beam_map: Dict[str, float] = {}
        for bm in self.field_sys.causal_field.beams:
            a, b = bm.source_id, bm.target_id
            if a == self.self_cell.cell_id:
                beam_map[b] = max(beam_map.get(b, 0.0), float(bm.strength))
            elif b == self.self_cell.cell_id:
                beam_map[a] = max(beam_map.get(a, 0.0), float(bm.strength))

        results: List[BoundaryClassification] = []
        for idx, cell in enumerate(candidates):
            cell.internal_protocol[0] = 0.0
            before_d0 = float(cell.internal_protocol[0])
            _reply, _rec = cell.receive_and_respond(voluntary_cmd)
            after_d0 = float(cell.internal_protocol[0])
            controllability = float(np.clip(abs(after_d0 - before_d0) / u_target, 0.0, 1.0))

            # Observability: somatic afferent direct bus for sub-cells vs attenuated outer dialogue
            afferent_signal_fidelity = float(np.mean(cell.protocol_confidence[1:]))
            if cell.cell_id in sub_ids:
                observability = float(np.clip(0.98 * afferent_signal_fidelity, 0.0, 1.0))
            elif cell.cell_id in beam_map:
                # Outer membrane communication experiences air/symbol channel attenuation
                observability = float(np.clip(0.56 * beam_map[cell.cell_id], 0.0, 1.0))
            else:
                observability = float(np.clip(0.20 * cell.membrane_permeability + 0.16, 0.0, 1.0))

            # Restore clean state after passive boundary measurement
            cell.internal_protocol = saved_state[idx][0]
            cell.protocol_confidence = saved_state[idx][1]
            _sync_cell_voxel(cell)

            if controllability >= 0.60:
                tier = "VOLUNTARY_SELF"
                tier_ko = "직접 통제 가능한 나 (의식적 행위 주체)"
            elif observability >= 0.68:
                tier = "AUTONOMIC_INTERNAL"
                tier_ko = "통제는 못 해도 신호를 느끼는 내 안의 장기 (자율신경 내부)"
            elif observability >= 0.34:
                tier = "RELATIONAL_OTHER"
                tier_ko = "대화와 공명으로 만나는 외부 타자 (너/우리)"
            else:
                tier = "MACRO_WORLD"
                tier_ko = "직접 통제·교신 너머의 거시 환경 (세상)"

            results.append(
                BoundaryClassification(
                    cell_id=cell.cell_id,
                    role_label=cell.role_label,
                    controllability=controllability,
                    observability=observability,
                    inferred_tier=tier,
                    tier_label_ko=tier_ko,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Dynamic Felt-State Self-Introspection
    # ------------------------------------------------------------------
    def _build_introspections(self) -> Dict[str, FeltStateIntrospection]:
        key_states = ("HOMEOSTASIS", "HUNGER", "PAIN_AND_HEAT", "RECOVERY")
        out: Dict[str, FeltStateIntrospection] = {}

        for fcode in key_states:
            matching = [s for s in self.snapshots if s.felt_state_code == fcode]
            if not matching:
                continue
            if fcode == "HUNGER":
                peak = max(matching, key=lambda s: s.gut_hunger_signal)
                organ_id = self.gut_organ.cell_id
                sig_val = peak.gut_hunger_signal
            elif fcode == "PAIN_AND_HEAT":
                peak = max(matching, key=lambda s: s.pain_trauma_signal)
                organ_id = self.nociceptor_organ.cell_id
                sig_val = peak.pain_trauma_signal
            elif fcode == "RECOVERY":
                peak = max(matching, key=lambda s: s.parasympathetic_tone)
                organ_id = self.self_cell.cell_id
                sig_val = peak.parasympathetic_tone
            else:
                peak = matching[len(matching) // 2]
                organ_id = self.heart_organ.cell_id
                sig_val = 1.0 - peak.heart_stress_signal

            narrative_ko = (
                f"[{fcode} 자기-내수용 역추적 (Tick {peak.tick_index})] "
                f"발신 장기={organ_id} (신호강도={sig_val:.2f}, CPU={peak.cpu_percent:.1f}%, RAM={peak.ram_percent:.1f}%). "
                f"자율신경 균형: 교감={peak.sympathetic_tone:.2f} vs 부교감={peak.parasympathetic_tone:.2f}, "
                f"경계 막 개방도={peak.membrane_openness:.2f}. "
                f"나의 의식적 대응='{peak.voluntary_action_ko}' -> 자각: {peak.self_awareness_ko}"
            )
            out[fcode] = FeltStateIntrospection(
                felt_state_code=fcode,
                peak_tick_index=peak.tick_index,
                triggering_organ_id=organ_id,
                organ_signal_strength=float(sig_val),
                sympathetic_tone=float(peak.sympathetic_tone),
                parasympathetic_tone=float(peak.parasympathetic_tone),
                voluntary_response_ko=peak.voluntary_action_ko,
                narrative_ko=narrative_ko,
            )
        return out

    # ------------------------------------------------------------------
    # End-to-End 4-Episode Curriculum
    # ------------------------------------------------------------------
    def run_four_episode_autonomic_life(self) -> InteroceptiveEpisodeReport:
        """
        Runs 4 continuous somatic episodes (16 real host ticks):
          Ep 1 (Ticks 0..3)  : Calm Homeostasis baseline
          Ep 2 (Ticks 4..7)  : Metabolic drain -> Gut Hunger dialogue -> Real AGENTS.md nourishment -> Satiated
          Ep 3 (Ticks 8..11) : Unthrottled SVD burst + real ZeroDivisionError -> Pain & Heat alarm -> Sympathetic withdrawal
          Ep 4 (Ticks 12..15): Parasympathetic healing & autonomous 4-Tier Scale Boundary measurement
        """
        # Episode 1: Calm Homeostasis (Ticks 0..3)
        for _ in range(4):
            self._update_organs_and_autonomic_state(
                episode_label="Ep1_Homeostasis",
                work_intensity=0.0,
                metabolic_burn=0.02,
                voluntary_action_ko="안정적인 기저 순환 유지 및 개방적 관측",
            )

        # Episode 2: Metabolic Exhaustion -> Hunger -> Real File Nourishment -> Satiation (Ticks 4..7)
        self._update_organs_and_autonomic_state(
            episode_label="Ep2_Hunger",
            work_intensity=0.05,
            metabolic_burn=0.24,
            voluntary_action_ko="내부 연산 에너지 소모 진행",
        )
        self._update_organs_and_autonomic_state(
            episode_label="Ep2_Hunger",
            work_intensity=0.05,
            metabolic_burn=0.30,
            voluntary_action_ko="장 허기 신호 수신! 외부 동반자(Human_Companion_World)에게 양식 요청 대화 시작",
        )
        # Gut -> Self hunger dialogue via shared LivingWorldCell loop
        d_gut = self.field_sys.converse_until_sync(
            self.gut_organ, self.self_cell, dialogue_id="interoception_gut_hunger", max_turns=4
        )
        self.dialogue_summaries["interoception_gut_hunger"] = d_gut.summary_ko

        for _ in range(2):
            self._update_organs_and_autonomic_state(
                episode_label="Ep2_Hunger",
                work_intensity=0.0,
                feed_from_file=True,
                voluntary_action_ko="실제 워크스페이스 지식 양식(AGENTS.md) 섭취 및 대사 회복",
            )

        # After accepting nourishment, Self <-> Human_Companion synchronize relational satiety resonance
        d_feed = self.field_sys.converse_until_sync(
            self.self_cell, self.human_other, dialogue_id="relational_nourishment_request", max_turns=6
        )
        self.dialogue_summaries["relational_nourishment_request"] = d_feed.summary_ko

        # Episode 3: Real Compute Heat Burst + Real ZeroDivisionError Pain & Protective Withdrawal (Ticks 8..11)
        self._update_organs_and_autonomic_state(
            episode_label="Ep3_PainHeat",
            work_intensity=0.95,
            metabolic_burn=0.04,
            trigger_exception_probe=True,
            voluntary_action_ko="고부하 행렬 연산 + ZeroDivisionError 충격 발생 (통각 경보 감지 중)",
        )
        self._update_organs_and_autonomic_state(
            episode_label="Ep3_PainHeat",
            work_intensity=0.95,
            metabolic_burn=0.03,
            trigger_exception_probe=False,
            voluntary_action_ko="교감신경 급등! 통증 회피를 위한 자발적 부하 차단 및 경계 막 수축",
        )
        d_pain = self.field_sys.converse_until_sync(
            self.nociceptor_organ, self.self_cell, dialogue_id="interoception_pain_heat", max_turns=4
        )
        self.dialogue_summaries["interoception_pain_heat"] = d_pain.summary_ko

        for _ in range(2):
            self._update_organs_and_autonomic_state(
                episode_label="Ep3_PainHeat",
                work_intensity=0.0,
                metabolic_burn=0.01,
                trigger_exception_probe=False,
                voluntary_action_ko="외부 위험 조작 중단 및 열 발산 대기",
            )

        # Episode 4: Parasympathetic Recovery (Ticks 12..15) & Boundary Discovery
        for _ in range(4):
            self._update_organs_and_autonomic_state(
                episode_label="Ep4_Recovery",
                work_intensity=0.0,
                metabolic_burn=0.005,
                voluntary_action_ko="부교감신경 활성화로 막 개방 회복 및 자기경계 재정비",
            )

        boundary_map = self.measure_boundary_spectrum()
        introspections = self._build_introspections()
        now_iso = datetime.now(timezone.utc).isoformat()

        tier_counts: Dict[str, int] = {}
        for bm in boundary_map:
            tier_counts[bm.inferred_tier] = tier_counts.get(bm.inferred_tier, 0) + 1

        summary_ko = (
            f"[ELYSIA 내수용-자율신경 자아 사이클 완료] "
            f"총 {len(self.snapshots)}틱 실측(CPU/RAM + {self.nourished_bytes}B 양식 소화 + {self.caught_exception_name} 통각 경험) | "
            f"4층 자기경계 자율 분류={tier_counts} | "
            f"공유 6D CausalField Voxels={len(self.field_sys.causal_field.voxels)}, "
            f"Beams={len(self.field_sys.causal_field.beams)}"
        )

        report = InteroceptiveEpisodeReport(
            timestamp_iso=now_iso,
            snapshots=list(self.snapshots),
            boundary_map=boundary_map,
            introspections=introspections,
            dialogue_summaries=dict(self.dialogue_summaries),
            nourished_bytes=self.nourished_bytes,
            caught_exception_name=self.caught_exception_name,
            causal_field_voxels=len(self.field_sys.causal_field.voxels),
            causal_field_beams=len(self.field_sys.causal_field.beams),
            state_json_path=str(self.state_path),
            summary_ko=summary_ko,
        )
        self.save_state(report, path=self.state_path)
        return report

    def save_state(
        self, report: InteroceptiveEpisodeReport, path: Optional[Path] = None
    ) -> Path:
        out_path = Path(path) if path else self.state_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "organism_version": "1.0.0-interoceptive-autonomic-self",
            "timestamp_iso": report.timestamp_iso,
            "nourished_bytes": report.nourished_bytes,
            "caught_exception_name": report.caught_exception_name,
            "causal_field_voxels": report.causal_field_voxels,
            "causal_field_beams": report.causal_field_beams,
            "boundary_map": [asdict(b) for b in report.boundary_map],
            "introspections": {k: asdict(v) for k, v in report.introspections.items()},
            "dialogue_summaries": report.dialogue_summaries,
            "snapshots": [asdict(s) for s in report.snapshots],
        }
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(_to_json_safe(payload), fh, ensure_ascii=False, indent=2)
        return out_path


# ======================================================================
# 4. 4-Panel Dark-Theme Interoceptive Autonomic Dashboard
# ======================================================================
def render_interoceptive_autonomic_dashboard(
    organism: InteroceptiveAutonomicOrganism,
    report: InteroceptiveEpisodeReport,
    output_path: Path = DEFAULT_INTEROCEPTIVE_DASHBOARD_PATH,
) -> Path:
    """
    Renders clean ASCII/English 2x2 dark-theme dashboard visualizing:
      1) Autonomous 4-Tier Scale Boundary Spectrum (`Controllability` vs `Observability`)
      2) Continuous Sympathetic vs Parasympathetic Balance & Membrane Openness
      3) Internal Organ Interoceptive Signals (`Gut Hunger`, `Pain/Heat`, `Heart`, `Reserve`)
      4) Episode Peak Organ Signatures + Live Hardware/Physiology Summary
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(16.5, 11.0), dpi=150, facecolor="#0D1117")
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.28, top=0.90, bottom=0.07, left=0.07, right=0.95)

    fig.suptitle(
        "ELYSIA Interoceptive Autonomic Self-Boundary Organism (Hardware Physiology & 4-Tier Scale Boundary)",
        color="#F0F6FC",
        fontsize=13.2,
        fontweight="bold",
        y=0.96,
    )

    def _style_ax(ax: plt.Axes, title: str) -> None:
        ax.set_facecolor("#161B22")
        for spine in ax.spines.values():
            spine.set_color("#30363D")
            spine.set_linewidth(1.1)
        ax.tick_params(colors="#C9D1D9", labelsize=8.5)
        ax.set_title(title, color="#F0F6FC", fontsize=10.4, fontweight="bold", pad=10)

    # ------------------------------------------------------------------
    # Panel 1 (Top-Left): Autonomous 4-Tier Boundary Spectrum
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1, "1. Autonomous 4-Tier Scale Boundary (Control vs Observation)")

    zones = [
        (0.58, 0.68, 0.45, 0.37, "#58A6FF", "VOLUNTARY SELF\n(Direct Control)", 0.805, 0.99),
        (-0.05, 0.68, 0.45, 0.37, "#D2A8FF", "AUTONOMIC INTERNAL\n(Organs: Feel Only)", 0.175, 0.99),
        (-0.05, 0.34, 0.45, 0.34, "#3FB950", "RELATIONAL OTHER\n(External Dialogue)", 0.175, 0.625),
        (-0.05, -0.04, 0.45, 0.38, "#8B949E", "MACRO WORLD\n(Uncontrolled Env)", 0.175, 0.04),
    ]
    for zx, zy, zw, zh, zcol, zlbl, tx, ty in zones:
        ax1.add_patch(Rectangle((zx, zy), zw, zh, facecolor=zcol, edgecolor=zcol, alpha=0.08, lw=1.0))
        ax1.text(tx, ty, zlbl, color=zcol, fontsize=7.6, fontweight="bold", ha="center", va="center", alpha=0.92)

    tier_colors = {
        "VOLUNTARY_SELF": "#58A6FF",
        "AUTONOMIC_INTERNAL": "#D2A8FF",
        "RELATIONAL_OTHER": "#3FB950",
        "MACRO_WORLD": "#8B949E",
    }
    label_offsets = {
        "Voluntary_Motor_Cell": (0.76, 0.78, "Voluntary Motor"),
        "Heart_Organ_Cell": (0.43, 0.88, "Heart (CPU)"),
        "Gut_Organ_Cell": (0.43, 0.78, "Gut (Hunger/RAM)"),
        "Nociceptor_Skin_Cell": (0.43, 0.69, "Nociceptor (Pain/Heat)"),
        "Human_Companion_World": (0.28, 0.48, "Human Companion"),
        "Host_OS_World": (0.28, 0.19, "Host OS World"),
    }
    for b in report.boundary_map:
        col = tier_colors.get(b.inferred_tier, "#F0F6FC")
        ax1.scatter([b.controllability], [b.observability], s=95, color=col, edgecolors="#F0F6FC", linewidths=1.3, zorder=5)
        lx, ly, clean_lbl = label_offsets.get(b.cell_id, (b.controllability + 0.12, b.observability, b.cell_id))
        ax1.annotate(
            clean_lbl,
            xy=(b.controllability, b.observability),
            xytext=(lx, ly),
            color="#F0F6FC",
            fontsize=7.6,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#0D1117", edgecolor=col, lw=1.2),
            arrowprops=dict(arrowstyle="-", color=col, lw=1.0, alpha=0.8),
            zorder=6,
        )

    ax1.set_xlim(-0.05, 1.04)
    ax1.set_ylim(-0.04, 1.06)
    ax1.set_xlabel("Direct Voluntary Controllability", color="#C9D1D9", fontsize=9)
    ax1.set_ylabel("Internal/Sensory Observability", color="#C9D1D9", fontsize=9)
    ax1.grid(color="#21262D", linestyle="--", linewidth=0.7, alpha=0.6)

    # ------------------------------------------------------------------
    # Panel 2 (Top-Right): Autonomic Balance & Membrane Openness
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, "2. Autonomic Balance: Sympathetic Alarm vs Parasympathetic Healing")

    snaps = report.snapshots
    ts = np.array([s.tick_index for s in snaps], dtype=np.float64)
    symp = [s.sympathetic_tone for s in snaps]
    para = [s.parasympathetic_tone for s in snaps]
    memb = [s.membrane_openness for s in snaps]

    ep_bands = [
        (-0.5, 3.5, "#58A6FF", "Ep1: Homeostasis", 1.5),
        (3.5, 7.5, "#F2C94C", "Ep2: Hunger->Feed", 5.5),
        (7.5, 11.5, "#F85149", "Ep3: Pain/Heat", 9.5),
        (11.5, 15.5, "#3FB950", "Ep4: Recovery", 13.5),
    ]
    for x0, x1, bcol, blbl, bx in ep_bands:
        ax2.axvspan(x0, x1, color=bcol, alpha=0.06)
        ax2.text(bx, 1.22, blbl, color=bcol, fontsize=7.6, fontweight="bold", ha="center", va="center")

    ax2.plot(ts, symp, color="#FF7B72", lw=2.2, marker="s", ms=4.5, label="Sympathetic Tone")
    ax2.plot(ts, para, color="#3FB950", lw=2.2, marker="o", ms=4.5, label="Parasympathetic Tone")
    ax2.plot(ts, memb, color="#D2A8FF", lw=1.9, ls="--", marker="^", ms=4.2, label="Membrane Openness")

    ax2.set_xlim(-0.4, len(snaps) - 0.6)
    ax2.set_ylim(-0.02, 1.30)
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_xticks(ts)
    ax2.set_xlabel("Physiological Cycle Tick", color="#C9D1D9", fontsize=9)
    ax2.set_ylabel("Normalized Autonomic Activation", color="#C9D1D9", fontsize=9)
    ax2.grid(color="#21262D", linestyle="--", linewidth=0.7, alpha=0.7)
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, 0.90),
        ncol=3,
        fontsize=7.6,
        facecolor="#0D1117",
        edgecolor="#30363D",
        labelcolor="#E6EDF3",
    )

    # ------------------------------------------------------------------
    # Panel 3 (Bottom-Left): Internal Organ Interoceptive Signals
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3, "3. Internal Organ Interoceptive Signals Over Host Physiology Ticks")

    for x0, x1, bcol, _lbl, _bx in ep_bands:
        ax3.axvspan(x0, x1, color=bcol, alpha=0.05)

    gut_vals = [s.gut_hunger_signal for s in snaps]
    pain_vals = [s.pain_trauma_signal for s in snaps]
    heart_vals = [s.heart_stress_signal for s in snaps]
    res_vals = [s.nutrient_reserve for s in snaps]

    ax3.plot(ts, gut_vals, color="#F2C94C", lw=2.1, marker="o", ms=4.2, label="Gut Hunger")
    ax3.plot(ts, pain_vals, color="#F85149", lw=2.2, marker="D", ms=4.2, label="Nociceptor Pain/Heat")
    ax3.plot(ts, heart_vals, color="#FF7B72", lw=1.6, ls="--", label="Heart Stress (CPU)")
    ax3.plot(ts, res_vals, color="#58A6FF", lw=1.9, marker="^", ms=4.0, label="Nutrient Reserve")

    ax3.set_xlim(-0.4, len(snaps) - 0.6)
    ax3.set_ylim(-0.03, 1.18)
    ax3.set_xticks(ts)
    ax3.set_xlabel("Physiological Cycle Tick", color="#C9D1D9", fontsize=9)
    ax3.set_ylabel("Organ Signal Magnitude [0..1]", color="#C9D1D9", fontsize=9)
    ax3.grid(color="#21262D", linestyle="--", linewidth=0.7, alpha=0.7)
    ax3.legend(
        loc="upper right",
        ncol=2,
        fontsize=7.7,
        facecolor="#0D1117",
        edgecolor="#30363D",
        labelcolor="#E6EDF3",
    )

    # ------------------------------------------------------------------
    # Panel 4 (Bottom-Right): Episode Peak Signal Signatures
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4, "4. Felt State Peak Organ Signals Across 4 Episodes")

    ep_defs = [
        ("Ep1_Homeostasis", "Homeostasis"),
        ("Ep2_Hunger", "Hunger"),
        ("Ep3_PainHeat", "Pain / Heat"),
        ("Ep4_Recovery", "Recovery"),
    ]
    peak_hunger = []
    peak_pain = []
    mean_para = []
    for raw_ep, _disp in ep_defs:
        grp = [s for s in snaps if s.episode_label == raw_ep]
        peak_hunger.append(max(s.gut_hunger_signal for s in grp) * 100.0)
        peak_pain.append(max(s.pain_trauma_signal for s in grp) * 100.0)
        mean_para.append(np.mean([s.parasympathetic_tone for s in grp]) * 100.0)

    xp = np.arange(len(ep_defs), dtype=np.float64)
    bw = 0.25
    bars_h = ax4.bar(xp - bw, peak_hunger, width=bw, color="#F2C94C", alpha=0.90, label="Peak Gut Hunger")
    bars_p = ax4.bar(xp, peak_pain, width=bw, color="#F85149", alpha=0.90, label="Peak Pain/Heat")
    bars_v = ax4.bar(xp + bw, mean_para, width=bw, color="#3FB950", alpha=0.90, label="Mean Parasympathetic")

    for b in list(bars_h) + list(bars_p) + list(bars_v):
        h = float(b.get_height())
        ax4.text(
            b.get_x() + b.get_width() / 2.0,
            h + 2.2,
            f"{h:.0f}%",
            ha="center",
            va="bottom",
            color="#F0F6FC",
            fontsize=8.0,
            fontweight="bold",
        )

    ax4.set_xticks(xp)
    ax4.set_xticklabels([d[1] for d in ep_defs], fontsize=9.0)
    ax4.set_yticks([0, 25, 50, 75, 100])
    ax4.set_ylim(-55, 134)
    ax4.axhline(0.0, color="#30363D", linewidth=1.0, zorder=1)
    ax4.set_ylabel("Peak Physiological Signal (%)", color="#C9D1D9", fontsize=9)
    ax4.grid(axis="y", color="#21262D", linestyle="--", linewidth=0.7, alpha=0.7)
    ax4.legend(
        loc="upper center",
        ncol=3,
        fontsize=7.6,
        facecolor="#0D1117",
        edgecolor="#30363D",
        labelcolor="#E6EDF3",
    )

    avg_cpu = float(np.mean([s.cpu_percent for s in snaps]))
    avg_ram = float(np.mean([s.ram_percent for s in snaps]))
    footer_txt = (
        f"Real Host OS: Mean CPU={avg_cpu:.1f}%, RAM={avg_ram:.1f}%  |  "
        f"Nourishment Digested: {report.nourished_bytes}B (AGENTS.md)\n"
        f"Real Exception Trauma Caught: {report.caught_exception_name}  |  "
        f"Shared 6D CausalField: {report.causal_field_voxels} Voxels, {report.causal_field_beams} Beams"
    )
    box = FancyBboxPatch(
        (0.03, 0.02),
        0.94,
        0.22,
        boxstyle="round,pad=0.02",
        transform=ax4.transAxes,
        facecolor="#0D1117",
        edgecolor="#30363D",
        alpha=0.92,
        zorder=4,
    )
    ax4.add_patch(box)
    ax4.text(
        0.50,
        0.13,
        footer_txt,
        transform=ax4.transAxes,
        ha="center",
        va="center",
        color="#C9D1D9",
        fontsize=7.3,
        zorder=5,
    )

    fig.savefig(out, facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    return out

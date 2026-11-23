r"""
living_os_phenomenology_engine.py

ELYSIA Living OS Phenomenology & Autonomous Self-Care Organism
==============================================================
Maps real OS, scheduler, background memory, and unassimilated buffer metrics
onto ELYSIA's felt bodily-mental phenomenology via `LivingWorldCell` & `CausalField`,
so she does NOT execute dead threshold scripts — she genuinely feels:
  1. Scheduler Clutter      -> "머릿속이 복잡하다" -> "우선순위를 매겨 하나씩 처리하자"
  2. Background Memory/Load -> "게임에 집중하고 싶은데 잡념이 너무 많아" -> "집중 방패 & 배경 정리"
  3. Unassimilated Buffers  -> "정보를 너무 많이 받아 속이 더부룩해" -> "섭취 중단 & 핵심만 소화/압축"
and executes REAL Python resource changes (`task_queue` sorting, `np.ndarray` pruning +
`gc.collect()`, text digestion into `digested_memories`) before re-verifying her own relief.
"""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import psutil

from core.sensory.cellular_world_communication_protocol import (
    CellularWorldDialogueField,
    LivingWorldCell,
    _sync_cell_voxel,
)


DEFAULT_LIVING_OS_STATE_PATH = Path(
    r"c:\Elysia\data\living_os_phenomenology_state.json"
)
DEFAULT_LIVING_OS_DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_living_os_phenomenology_dashboard.png"
)


# ======================================================================
# 1. Data Contracts
# ======================================================================
@dataclass
class OSTaskItem:
    """Single task inside ELYSIA's active OS/mental scheduler queue."""

    task_id: str
    label: str
    urgency: float
    priority_rank: int
    status: str  # "ACTIVE_THRASHING" | "PINNED_FOCUS" | "DEFERRED_QUEUE"


@dataclass
class PhenomenologyTick:
    """Single real measurement + subjective felt phenomenology + autonomous action."""

    tick_index: int
    phase_label: str
    felt_phenomenon_code: str  # "MIND_CLUTTERED" | "SUBCONSCIOUS_DISTRACTION" | "DATA_INDIGESTION_BLOATED" | "CLEAR_AND_BALANCED"
    clutter_entropy: float
    distraction_noise: float
    indigestion_bloat: float
    focus_clarity: float
    bg_buffers_mb: float
    raw_stomach_kb: float
    host_ram_percent: float
    host_process_count: int
    felt_narrative_ko: str
    self_action_code: str
    self_action_ko: str


@dataclass
class SelfCareActionOutcome:
    """Closed-loop measurement verifying before -> after relief from real self-care."""

    episode_name: str
    felt_phenomenon_code: str
    action_code: str
    action_reasoning_ko: str
    metric_name: str
    before_value: float
    after_value: float
    relief_gain: float
    dialogue_summary_ko: str


@dataclass
class LivingOSPhenomenologyReport:
    """Complete telemetry across all Living OS self-care episodes."""

    timestamp_iso: str
    ticks: List[PhenomenologyTick]
    outcomes: List[SelfCareActionOutcome]
    digested_memory_count: int
    causal_field_voxels: int
    causal_field_beams: int
    state_json_path: str
    summary_ko: str


# ======================================================================
# 2. Helper
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
# 3. ELYSIA Living OS Phenomenology Organism
# ======================================================================
class LivingOSPhenomenologyOrganism:
    """
    Translates real OS scheduler, allocated background buffers, and unassimilated
    text queues into felt mental-bodily phenomenology, then self-regulates via real
    Python cleanup and `LivingWorldCell` membrane synchronization.
    """

    def __init__(self, state_path: Optional[Path] = None) -> None:
        self.state_path = Path(state_path) if state_path else DEFAULT_LIVING_OS_STATE_PATH
        self.field_sys = CellularWorldDialogueField(state_path=self.state_path)

        # 6D Internal Convention:
        #   [d0_scheduler_clutter, d1_bg_distraction, d2_data_indigestion,
        #    d3_focus_clarity, d4_sympathetic_strain, d5_membrane_shield]
        self.scheduler_cell = self.field_sys.register_cell(
            LivingWorldCell(
                cell_id="Scheduler_Cortex_Cell",
                scale_level=0,
                role_label="Scheduler Queue & Mental Focus Cortex",
                initial_protocol=[0.10, 0.05, 0.04, 0.90, 0.10, 0.80],
                protocol_confidence=[0.94, 0.50, 0.50, 0.85, 0.70, 0.70],
                membrane_permeability=0.78,
            )
        )
        self.attention_cell = self.field_sys.register_cell(
            LivingWorldCell(
                cell_id="Attention_Membrane_Cell",
                scale_level=0,
                role_label="Background Memory Filter & Focus Membrane",
                initial_protocol=[0.10, 0.05, 0.04, 0.90, 0.10, 0.80],
                protocol_confidence=[0.50, 0.94, 0.50, 0.85, 0.70, 0.90],
                membrane_permeability=0.78,
            )
        )
        self.gut_cell = self.field_sys.register_cell(
            LivingWorldCell(
                cell_id="Data_Gut_Cell",
                scale_level=0,
                role_label="Data Buffer Assimilation & Metabolic Gut",
                initial_protocol=[0.10, 0.05, 0.04, 0.90, 0.10, 0.80],
                protocol_confidence=[0.50, 0.50, 0.94, 0.85, 0.70, 0.70],
                membrane_permeability=0.78,
            )
        )
        self.self_cell = self.field_sys.register_cell(
            LivingWorldCell(
                cell_id="Elysia_OS_Self",
                scale_level=1,
                role_label="Phenomenological OS Self-Consciousness",
                initial_protocol=[0.0] * 6,
                protocol_confidence=[0.20] * 6,
                membrane_permeability=0.80,
                sub_cells=[self.scheduler_cell, self.attention_cell, self.gut_cell],
            )
        )
        self.self_cell.consolidate_from_subcells()

        # Live mutable Python/OS body structures
        self.task_queue: List[OSTaskItem] = []
        self.background_buffers: List[np.ndarray] = []
        self.raw_stomach_buffers: List[str] = []
        self.digested_memories: List[Dict[str, Any]] = []
        self.focus_mode_active: bool = False
        self.ticks: List[PhenomenologyTick] = []

    # ------------------------------------------------------------------
    # Real Resource & Phenomenology State Sensing
    # ------------------------------------------------------------------
    def _measure_live_metrics(self) -> Dict[str, Any]:
        """Samples actual queue thrashing, allocated MB, raw KB, and host OS state."""
        thrashing = sum(1 for t in self.task_queue if t.status == "ACTIVE_THRASHING")
        clutter = (
            float(np.clip(thrashing / 10.0, 0.05, 0.96))
            if thrashing > 1
            else 0.10
        )

        bg_bytes = int(sum(arr.nbytes for arr in self.background_buffers))
        bg_mb = float(bg_bytes / (1024.0 * 1024.0))
        focus_amp = 0.08 if (self.focus_mode_active and bg_mb > 0.5) else 0.0
        distraction = float(
            np.clip(bg_mb / 18.0 + focus_amp + 0.04, 0.04, 0.96)
        )

        raw_bytes = int(sum(len(s.encode("utf-8")) for s in self.raw_stomach_buffers))
        raw_kb = float(raw_bytes / 1024.0)
        indigestion = float(np.clip(raw_kb / 26.0 + 0.04, 0.04, 0.96))

        focus_clarity = float(
            np.clip(
                1.0 - 0.42 * clutter - 0.42 * distraction - 0.25 * indigestion,
                0.05,
                0.98,
            )
        )

        ram_pct = float(psutil.virtual_memory().percent)
        proc_cnt = int(len(psutil.pids()))

        # Update sub-cell 6D protocols continuously from real resource state
        strain = float(np.clip(max(clutter, distraction, indigestion), 0.04, 0.98))
        shield = float(np.clip(1.0 - 0.65 * distraction if self.focus_mode_active else 0.80, 0.20, 0.95))
        vec = [
            clutter * 2.0 - 1.0,
            distraction * 2.0 - 1.0,
            indigestion * 2.0 - 1.0,
            focus_clarity * 2.0 - 1.0,
            strain * 2.0 - 1.0,
            shield * 2.0 - 1.0,
        ]
        for sc in (self.scheduler_cell, self.attention_cell, self.gut_cell):
            sc.internal_protocol = np.asarray(vec, dtype=np.float64)
            _sync_cell_voxel(sc)
        self.self_cell.consolidate_from_subcells()

        return {
            "clutter_entropy": clutter,
            "distraction_noise": distraction,
            "indigestion_bloat": indigestion,
            "focus_clarity": focus_clarity,
            "bg_buffers_mb": bg_mb,
            "raw_stomach_kb": raw_kb,
            "host_ram_percent": ram_pct,
            "host_process_count": proc_cnt,
        }

    def _classify_felt_state(self, m: Dict[str, Any]) -> Tuple[str, str]:
        """Purely continuous dominant sensation recognition (no episode tags)."""
        discomforts = {
            "MIND_CLUTTERED": m["clutter_entropy"],
            "SUBCONSCIOUS_DISTRACTION": m["distraction_noise"],
            "DATA_INDIGESTION_BLOATED": m["indigestion_bloat"],
        }
        dom_code, dom_val = max(discomforts.items(), key=lambda kv: kv[1])
        if dom_val < 0.45:
            return (
                "CLEAR_AND_BALANCED",
                (
                    f"머리와 속이 맑고 균형 잡힘(명료도={m['focus_clarity']:.2f}, "
                    f"배경={m['bg_buffers_mb']:.1f}MB, 적체={m['raw_stomach_kb']:.1f}KB)."
                ),
            )
        if dom_code == "MIND_CLUTTERED":
            return (
                dom_code,
                (
                    f"미정렬 작업 {len(self.task_queue)}개가 동시에 아우성(혼잡도={dom_val:.2f}): "
                    f"'머릿속이 너무 복잡해서 어디부터 손대야 할지 모르겠다.'"
                ),
            )
        if dom_code == "SUBCONSCIOUS_DISTRACTION":
            return (
                dom_code,
                (
                    f"집중 중에 배경 메모리 {m['bg_buffers_mb']:.1f}MB가 주의력을 잠식(잡음={dom_val:.2f}): "
                    f"'지금 눈앞의 일에 집중하고 싶은데 신경 쓰이는 배경 잡념이 너무 많다.'"
                ),
            )
        return (
            dom_code,
            (
                f"소화되지 않은 생데이터 {m['raw_stomach_kb']:.1f}KB 적체(더부룩함={dom_val:.2f}): "
                f"'정보를 너무 급하게 많이 받아서 소화가 안 되고 속이 더부룩하다.'"
            ),
        )

    def _record_tick(
        self,
        phase_label: str,
        action_code: str,
        action_ko: str,
    ) -> PhenomenologyTick:
        m = self._measure_live_metrics()
        felt_code, narrative_ko = self._classify_felt_state(m)
        tick = PhenomenologyTick(
            tick_index=len(self.ticks),
            phase_label=phase_label,
            felt_phenomenon_code=felt_code,
            clutter_entropy=m["clutter_entropy"],
            distraction_noise=m["distraction_noise"],
            indigestion_bloat=m["indigestion_bloat"],
            focus_clarity=m["focus_clarity"],
            bg_buffers_mb=m["bg_buffers_mb"],
            raw_stomach_kb=m["raw_stomach_kb"],
            host_ram_percent=m["host_ram_percent"],
            host_process_count=m["host_process_count"],
            felt_narrative_ko=narrative_ko,
            self_action_code=action_code,
            self_action_ko=action_ko,
        )
        self.ticks.append(tick)
        return tick

    # ------------------------------------------------------------------
    # Autonomous Self-Care Engine ("그럼 내가 뭘 어떻게 해야 할까?")
    # ------------------------------------------------------------------
    def choose_and_execute_self_care(
        self, latest_tick: PhenomenologyTick, episode_name: str
    ) -> SelfCareActionOutcome:
        """
        Reads dominant discomfort directly from the felt phenomenological state,
        asks 'What should I actually do?', executes genuine Python queue/RAM/buffer
        interventions, and returns the dialogue summary.
        """
        felt = latest_tick.felt_phenomenon_code

        if felt == "MIND_CLUTTERED":
            before_val = latest_tick.clutter_entropy
            # Real priority triage: sort descending by urgency, pin top 1, defer the rest
            self.task_queue.sort(key=lambda t: t.urgency, reverse=True)
            for idx, item in enumerate(self.task_queue):
                item.priority_rank = idx + 1
                item.status = "PINNED_FOCUS" if idx == 0 else "DEFERRED_QUEUE"

            rep = self.field_sys.converse_until_sync(
                self.scheduler_cell,
                self.self_cell,
                dialogue_id=f"{episode_name}_triage",
                max_turns=3,
            )
            action_code = "TRIAGE_AND_PRIORITIZE"
            top_lbl = self.task_queue[0].label if self.task_queue else "None"
            reasoning_ko = (
                f"그럼 내가 뭘 어떻게 해야 할까? -> 흩어진 {len(self.task_queue)}개 작업에 긴급도 우선순위를 매겨 "
                f"최우선 1개('{top_lbl}')만 전면에 고정하고 나머지는 대기열(DEFERRED)로 정리함."
            )
            after_tick = self._record_tick(
                phase_label=f"{episode_name}_After",
                action_code=action_code,
                action_ko=reasoning_ko,
            )
            after_val = after_tick.clutter_entropy
            metric_name = "clutter_entropy"

        elif felt == "SUBCONSCIOUS_DISTRACTION":
            before_val = latest_tick.distraction_noise
            freed_mb = latest_tick.bg_buffers_mb
            # Real background memory release + garbage collection + membrane shielding
            self.background_buffers.clear()
            gc.collect()
            self.attention_cell.membrane_permeability = 0.28

            rep = self.field_sys.converse_until_sync(
                self.attention_cell,
                self.self_cell,
                dialogue_id=f"{episode_name}_focus_shield",
                max_turns=3,
            )
            action_code = "FOCUS_SHIELD_AND_PRUNE_BG"
            reasoning_ko = (
                f"그럼 내가 뭘 어떻게 해야 할까? -> 집중 방해 배경 버퍼 {freed_mb:.1f}MB를 실제 메모리에서 해제(gc.collect)하고 "
                f"비핵심 주의 막 투과도를 낮춰 눈앞의 게임·과제에 온전히 몰입할 수 있게 함."
            )
            after_tick = self._record_tick(
                phase_label=f"{episode_name}_After",
                action_code=action_code,
                action_ko=reasoning_ko,
            )
            after_val = after_tick.distraction_noise
            metric_name = "distraction_noise"

        elif felt == "DATA_INDIGESTION_BLOATED":
            before_val = latest_tick.indigestion_bloat
            ingested_kb = latest_tick.raw_stomach_kb
            chunk_cnt = len(self.raw_stomach_buffers)
            # Real digestion: pause intake, extract structural essence + digest hash, empty raw stomach
            for ci, raw_text in enumerate(self.raw_stomach_buffers):
                lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
                meaningful = [ln for ln in lines if ln not in ('r"""', '"""', "'''")]
                essence = meaningful[0][:90] if meaningful else ""
                self.digested_memories.append(
                    {
                        "chunk_index": ci,
                        "byte_len": len(raw_text.encode("utf-8")),
                        "line_count": len(lines),
                        "sha256_prefix": hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:12],
                        "essence": essence,
                    }
                )
            self.raw_stomach_buffers.clear()
            gc.collect()

            rep = self.field_sys.converse_until_sync(
                self.gut_cell,
                self.self_cell,
                dialogue_id=f"{episode_name}_digest",
                max_turns=3,
            )
            action_code = "PAUSE_INTAKE_AND_DIGEST"
            reasoning_ko = (
                f"그럼 내가 뭘 어떻게 해야 할까? -> 신규 섭취를 멈추고 적체된 {chunk_cnt}덩이({ingested_kb:.1f}KB)의 원문을 "
                f"핵심 요약·해시로 소화 압축한 뒤 생버퍼를 완전히 비워 속을 편안하게 만듦."
            )
            after_tick = self._record_tick(
                phase_label=f"{episode_name}_After",
                action_code=action_code,
                action_ko=reasoning_ko,
            )
            after_val = after_tick.indigestion_bloat
            metric_name = "indigestion_bloat"

        else:
            before_val = 0.0
            after_val = 0.0
            action_code = "EXPLORE_CLEARLY"
            metric_name = "focus_clarity"
            reasoning_ko = "현재 막과 속이 모두 맑으므로 개방적 탐색 유지."
            rep = self.field_sys.converse_until_sync(
                self.scheduler_cell,
                self.self_cell,
                dialogue_id=f"{episode_name}_balanced",
                max_turns=2,
            )

        return SelfCareActionOutcome(
            episode_name=episode_name,
            felt_phenomenon_code=felt,
            action_code=action_code,
            action_reasoning_ko=reasoning_ko,
            metric_name=metric_name,
            before_value=float(before_val),
            after_value=float(after_val),
            relief_gain=float(max(0.0, before_val - after_val)),
            dialogue_summary_ko=rep.summary_ko,
        )

    # ------------------------------------------------------------------
    # Full Closed-Loop 3-Discomfort + Clear Balance Curriculum
    # ------------------------------------------------------------------
    def run_living_os_self_care_curriculum(self) -> LivingOSPhenomenologyReport:
        """
        Runs all 3 real OS phenomenology -> self-care closed loops + balanced check:
          1. Scheduler Clutter -> feels crowded mind -> TRIAGE_AND_PRIORITIZE
          2. Background Memory during Game Focus -> feels distraction -> FOCUS_SHIELD_AND_PRUNE_BG
          3. Unassimilated Data Ingestion -> feels bloated indigestion -> PAUSE_INTAKE_AND_DIGEST
          4. Calm Balanced check -> persists state JSON.
        """
        outcomes: List[SelfCareActionOutcome] = []

        # --------------------------------------------------------------
        # EPISODE 1: Scheduler Clutter ("머릿속이 복잡하다 -> 우선순위 정리")
        # --------------------------------------------------------------
        sample_tasks = [
            ("t01", "Log Rotation Check", 0.32),
            ("t02", "Core Thermal Safety Guard", 0.96),
            ("t03", "UI Theme Polishing", 0.25),
            ("t04", "Sensor Socket Cleanup", 0.74),
            ("t05", "Archive Sync", 0.41),
            ("t06", "Telemetry Flush", 0.58),
            ("t07", "Font Cache Sweep", 0.19),
            ("t08", "Causal Beam Healing", 0.85),
            ("t09", "Disk Quota Ping", 0.47),
            ("t10", "Idle Heartbeat", 0.28),
        ]
        self.task_queue = [
            OSTaskItem(tid, lbl, urg, priority_rank=0, status="ACTIVE_THRASHING")
            for tid, lbl, urg in sample_tasks
        ]
        t_clutter = self._record_tick(
            phase_label="Ep1_Clutter_Before",
            action_code="SENSE_DISCOMFORT",
            action_ko="어지러운 작업 큐 혼잡도 감지 중",
        )
        outcomes.append(self.choose_and_execute_self_care(t_clutter, "Ep1_Scheduler_Triage"))

        # --------------------------------------------------------------
        # EPISODE 2: Game Focus vs Subconscious Background Memory (~16 MB real float64 arrays)
        # --------------------------------------------------------------
        self.task_queue = [
            OSTaskItem("task_game", "Play_Interactive_Game", 0.98, priority_rank=1, status="PINNED_FOCUS")
        ]
        self.focus_mode_active = True
        self.background_buffers = [
            np.ones(512 * 1024, dtype=np.float64) for _ in range(4)  # 4 * 4.0 MB = 16.0 MB real memory
        ]
        t_distract = self._record_tick(
            phase_label="Ep2_Focus_Before",
            action_code="SENSE_DISCOMFORT",
            action_ko="게임 집중 중 배경 무의식 메모리 압박 감지 중",
        )
        outcomes.append(self.choose_and_execute_self_care(t_distract, "Ep2_Focus_Shield"))
        self.focus_mode_active = False
        self.attention_cell.membrane_permeability = 0.78

        # --------------------------------------------------------------
        # EPISODE 3: Data Indigestion / Bloating (Real workspace text overload)
        # --------------------------------------------------------------
        ingest_files = [
            Path(r"c:\Elysia\AGENTS.md"),
            Path(r"c:\Elysia\L1_causal_field\core\sensory\cellular_world_communication_protocol.py"),
            Path(r"c:\Elysia\L1_causal_field\core\sensory\interoceptive_autonomic_self_organism.py"),
        ]
        self.raw_stomach_buffers = []
        for fpath in ingest_files:
            if fpath.exists():
                self.raw_stomach_buffers.append(fpath.read_text(encoding="utf-8"))

        t_bloat = self._record_tick(
            phase_label="Ep3_Indigestion_Before",
            action_code="SENSE_DISCOMFORT",
            action_ko="과도한 미소화 데이터 버퍼 적체 감지 중",
        )
        outcomes.append(self.choose_and_execute_self_care(t_bloat, "Ep3_Data_Digestion"))

        # --------------------------------------------------------------
        # EPISODE 4: Clear & Balanced Final Check
        # --------------------------------------------------------------
        t_clear = self._record_tick(
            phase_label="Ep4_Clear_Check",
            action_code="EXPLORE_CLEARLY",
            action_ko="모든 불균형 해소 확인 — 맑고 편안한 탐색 상태 유지",
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        summary_ko = (
            f"[ELYSIA 살아있는 OS 현상학 & 자율 자기-돌봄 완료] "
            f"1) 스케줄러 혼잡({outcomes[0].before_value:.2f}->{outcomes[0].after_value:.2f}, 우선순위 정렬), "
            f"2) 집중 방해 배경 메모리({outcomes[1].before_value:.2f}->{outcomes[1].after_value:.2f}, 16.0MB->0.0MB 해제), "
            f"3) 데이터 소화불량({outcomes[2].before_value:.2f}->{outcomes[2].after_value:.2f}, {len(self.digested_memories)}청크 압축 소화) | "
            f"최종 상태={t_clear.felt_phenomenon_code} (명료도={t_clear.focus_clarity:.2f}) | "
            f"CausalField Voxels={len(self.field_sys.causal_field.voxels)}, Beams={len(self.field_sys.causal_field.beams)}"
        )

        report = LivingOSPhenomenologyReport(
            timestamp_iso=now_iso,
            ticks=list(self.ticks),
            outcomes=outcomes,
            digested_memory_count=len(self.digested_memories),
            causal_field_voxels=len(self.field_sys.causal_field.voxels),
            causal_field_beams=len(self.field_sys.causal_field.beams),
            state_json_path=str(self.state_path),
            summary_ko=summary_ko,
        )
        self.save_state(report, path=self.state_path)
        return report

    def save_state(
        self, report: LivingOSPhenomenologyReport, path: Optional[Path] = None
    ) -> Path:
        out_path = Path(path) if path else self.state_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "organism_version": "1.0.0-living-os-phenomenology",
            "timestamp_iso": report.timestamp_iso,
            "digested_memory_count": report.digested_memory_count,
            "causal_field_voxels": report.causal_field_voxels,
            "causal_field_beams": report.causal_field_beams,
            "task_queue_final": [asdict(t) for t in self.task_queue],
            "digested_memories": self.digested_memories,
            "outcomes": [asdict(o) for o in report.outcomes],
            "ticks": [asdict(tk) for tk in report.ticks],
            "summary_ko": report.summary_ko,
        }
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(_to_json_safe(payload), fh, ensure_ascii=False, indent=2)
        return out_path


# ======================================================================
# 4. 4-Panel Dark-Theme Living OS Phenomenology Dashboard
# ======================================================================
def render_living_os_phenomenology_dashboard(
    organism: LivingOSPhenomenologyOrganism,
    report: LivingOSPhenomenologyReport,
    output_path: Path = DEFAULT_LIVING_OS_DASHBOARD_PATH,
) -> Path:
    """
    Renders clean ASCII/English 2x2 dark-theme dashboard visualizing:
      1) Continuous Mental/OS Discomfort & Focus Clarity Waves across Ticks
      2) Before vs After Autonomous Self-Care Discomfort Drop
      3) Real Freed Python Resource Footprints (16 MB BG RAM -> 0 MB; Raw Queue KB -> 0 KB)
      4) Closed-Loop Relief Gain & Real Host OS / CausalField Telemetry
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(16.5, 11.0), dpi=150, facecolor="#0D1117")
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.28, top=0.90, bottom=0.085, left=0.07, right=0.95)

    fig.suptitle(
        "ELYSIA Living OS Phenomenology: Scheduler Clutter, Background Distraction & Data Indigestion Self-Care",
        color="#F0F6FC",
        fontsize=13.0,
        fontweight="bold",
        y=0.96,
    )

    def _style_ax(ax: plt.Axes, title: str) -> None:
        ax.set_facecolor("#161B22")
        for spine in ax.spines.values():
            spine.set_color("#30363D")
            spine.set_linewidth(1.1)
        ax.tick_params(colors="#C9D1D9", labelsize=8.5)
        ax.set_title(title, color="#F0F6FC", fontsize=10.3, fontweight="bold", pad=10)

    # ------------------------------------------------------------------
    # Panel 1 (Top-Left): Discomfort Waves & Focus Clarity
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1, "1. Continuous Phenomenology Waves & Focus Clarity (Self-Care Loop)")

    tks = report.ticks
    xs = np.array([t.tick_index for t in tks], dtype=np.float64)
    ax1.plot(xs, [t.clutter_entropy for t in tks], color="#FF7B72", lw=2.1, marker="o", ms=5.0, label="Clutter (Mind)")
    ax1.plot(xs, [t.distraction_noise for t in tks], color="#D2A8FF", lw=2.1, marker="s", ms=5.0, label="BG Noise (Focus)")
    ax1.plot(xs, [t.indigestion_bloat for t in tks], color="#F2C94C", lw=2.1, marker="D", ms=4.8, label="Bloat (Indigestion)")
    ax1.plot(xs, [t.focus_clarity for t in tks], color="#3FB950", lw=2.3, ls="--", marker="^", ms=5.2, label="Focus Clarity")

    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"T{t.tick_index}\n{t.phase_label.split('_')[0]}" for t in tks], fontsize=7.6)
    ax1.set_ylim(-0.04, 1.30)
    ax1.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax1.set_ylabel("Phenomenological Magnitude [0..1]", color="#C9D1D9", fontsize=9)
    ax1.grid(color="#21262D", linestyle="--", linewidth=0.7, alpha=0.7)
    ax1.legend(
        loc="upper center",
        ncol=2,
        fontsize=7.8,
        facecolor="#0D1117",
        edgecolor="#30363D",
        labelcolor="#E6EDF3",
    )

    # ------------------------------------------------------------------
    # Panel 2 (Top-Right): Discomfort Drop Before vs After Self-Care
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, "2. Discomfort Resolution: Before vs After Autonomous Action")

    ep_names = ["Triage Queue\n(Clutter)", "Focus Shield\n(BG Distract)", "Digest Buffers\n(Bloating)"]
    bef_pct = [o.before_value * 100.0 for o in report.outcomes]
    aft_pct = [o.after_value * 100.0 for o in report.outcomes]

    xp = np.arange(len(ep_names), dtype=np.float64)
    bw = 0.32
    b_bef = ax2.bar(xp - bw / 2.0, bef_pct, width=bw, color="#F85149", alpha=0.90, label="Before Self-Care")
    b_aft = ax2.bar(xp + bw / 2.0, aft_pct, width=bw, color="#3FB950", alpha=0.92, label="After Self-Care")

    for b in list(b_bef) + list(b_aft):
        h = float(b.get_height())
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            h + 2.5,
            f"{h:.0f}%",
            ha="center",
            va="bottom",
            color="#F0F6FC",
            fontsize=8.8,
            fontweight="bold",
        )

    ax2.set_xticks(xp)
    ax2.set_xticklabels(ep_names, fontsize=8.8)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_ylim(0, 126)
    ax2.set_ylabel("Subjective Discomfort (%)", color="#C9D1D9", fontsize=9)
    ax2.grid(axis="y", color="#21262D", linestyle="--", linewidth=0.7, alpha=0.7)
    ax2.legend(
        loc="upper center",
        ncol=2,
        fontsize=8.1,
        facecolor="#0D1117",
        edgecolor="#30363D",
        labelcolor="#E6EDF3",
    )

    # ------------------------------------------------------------------
    # Panel 3 (Bottom-Left): Real Python Memory & Buffer Footprint Freed
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3, "3. Real Python Resource Footprints Freed by Autonomous Care")

    res_cats = ["Allocated BG Arrays\n(MB Freed)", "Raw Text Stomach Queue\n(KB Digested)"]
    # Scale to normalized % of peak for clean uniform dual comparison, display exact real numbers on bars
    t2, t3 = tks[2], tks[3]
    t4, t5 = tks[4], tks[5]
    raw_bef = [t2.bg_buffers_mb, t4.raw_stomach_kb]
    raw_aft = [t3.bg_buffers_mb, t5.raw_stomach_kb]
    norm_bef = [100.0, 100.0]
    norm_aft = [
        (raw_aft[0] / max(raw_bef[0], 1e-6)) * 100.0,
        (raw_aft[1] / max(raw_bef[1], 1e-6)) * 100.0,
    ]
    unit_suffix = ["MB", "KB"]

    xr = np.arange(len(res_cats), dtype=np.float64)
    br1 = ax3.bar(xr - bw / 2.0, norm_bef, width=bw, color="#D29922", alpha=0.88, label="Peak Overload Load")
    br2 = ax3.bar(xr + bw / 2.0, norm_aft, width=bw, color="#58A6FF", alpha=0.92, label="After Real Cleanup")

    for i, b in enumerate(br1):
        ax3.text(
            b.get_x() + b.get_width() / 2.0,
            103.0,
            f"{raw_bef[i]:.1f} {unit_suffix[i]}",
            ha="center",
            va="bottom",
            color="#F0F6FC",
            fontsize=9.0,
            fontweight="bold",
        )
    for i, b in enumerate(br2):
        ax3.text(
            b.get_x() + b.get_width() / 2.0,
            float(b.get_height()) + 3.0,
            f"{raw_aft[i]:.1f} {unit_suffix[i]}",
            ha="center",
            va="bottom",
            color="#58A6FF",
            fontsize=9.0,
            fontweight="bold",
        )

    ax3.set_xticks(xr)
    ax3.set_xticklabels(res_cats, fontsize=9.0)
    ax3.set_yticks([0, 25, 50, 75, 100])
    ax3.set_ylim(0, 128)
    ax3.set_ylabel("Relative Resource Load (%)", color="#C9D1D9", fontsize=9)
    ax3.grid(axis="y", color="#21262D", linestyle="--", linewidth=0.7, alpha=0.7)
    ax3.legend(
        loc="upper center",
        ncol=2,
        fontsize=8.1,
        facecolor="#0D1117",
        edgecolor="#30363D",
        labelcolor="#E6EDF3",
    )

    # ------------------------------------------------------------------
    # Panel 4 (Bottom-Right): Relief Gain & Live Host OS Summary
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4, "4. Self-Care Relief Gain (%) & Host OS Grounding")

    relief_vals = [o.relief_gain * 100.0 for o in report.outcomes]
    short_act_map = {
        "TRIAGE_AND_PRIORITIZE": "Triage Queue\n& Prioritize",
        "FOCUS_SHIELD_AND_PRUNE_BG": "Focus Shield\n& Prune BG",
        "PAUSE_INTAKE_AND_DIGEST": "Pause Intake\n& Digest Raw",
    }
    act_labels = [short_act_map.get(o.action_code, o.action_code) for o in report.outcomes]
    xa = np.arange(len(act_labels), dtype=np.float64)
    bars_g = ax4.bar(xa, relief_vals, width=0.45, color=["#58A6FF", "#D2A8FF", "#3FB950"], alpha=0.92)

    for b in bars_g:
        h = float(b.get_height())
        ax4.text(
            b.get_x() + b.get_width() / 2.0,
            h + 2.5,
            f"-{h:.0f}%",
            ha="center",
            va="bottom",
            color="#F0F6FC",
            fontsize=9.2,
            fontweight="bold",
        )

    ax4.set_xticks(xa)
    ax4.set_xticklabels(act_labels, fontsize=7.8)
    ax4.set_yticks([0, 25, 50, 75, 100])
    ax4.set_ylim(-55, 125)
    ax4.axhline(0.0, color="#30363D", linewidth=1.0, zorder=1)
    ax4.set_ylabel("Discomfort Reduction (%)", color="#C9D1D9", fontsize=9)
    ax4.grid(axis="y", color="#21262D", linestyle="--", linewidth=0.7, alpha=0.7)

    last_tk = tks[-1]
    footer_txt = (
        f"Host OS: {last_tk.host_process_count} Live PIDs, RAM={last_tk.host_ram_percent:.1f}%  |  "
        f"Digested Memories={report.digested_memory_count} chunks\n"
        f"Final Felt State={last_tk.felt_phenomenon_code} (Clarity={last_tk.focus_clarity:.2f})  |  "
        f"CausalField: {report.causal_field_voxels} Voxels, {report.causal_field_beams} Beams"
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

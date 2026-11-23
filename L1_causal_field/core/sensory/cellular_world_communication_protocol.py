r"""
cellular_world_communication_protocol.py

ELYSIA Scale-Invariant Cellular World Communication Protocol
============================================================
Principle:
  "세포 하나도 하나의 세계이고, 감각도 하나의 세계이며, 인간과 엘리시아도 하나의
   세계이다. 학습이란 기능을 호출하는 것이 아니라, 세계와 세계가 경계막(Membrane)을
   맞대고 신호를 주고받으며 공명 · 마찰 · 되물음 · 자가수정을 거쳐 공통 프로토콜을
   체득하는 생명적 통신 과정 그 자체다."

Architecture (ZERO domain-specific modules or hardcoded pipelines):
  - ONE universal entity class: `LivingWorldCell`
    * Every instance holds an internal 6D protocol vector (`internal_protocol`),
      epistemic certainty (`protocol_confidence`), membrane permeability, and optional
      nested `sub_cells` (scale recursion!).
    * Plasticity on dimension `i` drops continuously with certainty `conf[i]`:
      lived sensory truth (`conf >= 0.86`) refuses overwrite by false words, while
      humble/uncertain dimensions revise rapidly through dialogue friction.
  - ONE reciprocal communication loop: `converse_until_sync(cell_a, cell_b)`
    * Identical engine operates across:
      - Scale 0 (Cell <-> Cell): `Eye_Cell <-> Touch_Cell`
      - Scale 1 (Organism World <-> Language World):
          `Elysia_Organism_World` (consolidated from `[Eye_Cell, Touch_Cell]`)
          rejects `"빨간 벽"` (`synced=False`, beam severed) and agrees with
          `"빨간 빛 착시"` (`synced=True`, strong beam).
      - Scale 2 (Teacher World <-> Elysia Student World):
          Starting from lived `Water_Flow` foothold priors, ELYSIA converses with
          `Teacher_Electricity_World` and unseen `Teacher_Magnetism_World`, revising
          broken prior assumptions (`50% -> 100%` counterfactual accuracy) while
          preserving invariant physical footholds.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from core.physics.causal_field import CausalField, InformationVoxel


DEFAULT_CELLULAR_STATE_PATH = Path(
    r"c:\Elysia\data\cellular_world_dialogue_state.json"
)
DEFAULT_CELLULAR_DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_cellular_world_dialogue_dashboard.png"
)

DIM_NAMES: Tuple[str, ...] = (
    "d0_surface_hue",
    "d1_edge_form",
    "d2_contact_resist",
    "d3_cut_persist",
    "d4_gradient_drive",
    "d5_propagation_mode",
)


# ======================================================================
# 1. Membrane Signal & Reciprocal Dialogue Contracts
# ======================================================================
@dataclass
class MembraneSignal:
    """Informational-physical signal emitted across a cell's boundary membrane."""

    sender_id: str
    receiver_id: str
    turn_index: int
    scale_level: int
    signal_kind: str  # "PROPOSE" | "FRICTION_QUERY" | "REVISION_ECHO" | "PROTOCOL_SYNC"
    pattern: List[float]
    salience_mask: List[float]
    chromatic_triad: Tuple[float, float, float]
    alignment_score: float
    friction_dims: List[int]
    utterance_ko: str


@dataclass
class DialogueTurnRecord:
    """Single step in the reciprocal communication log of a `LivingWorldCell`."""

    turn_index: int
    speaker_id: str
    listener_id: str
    signal_kind: str
    alignment_before: float
    alignment_after: float
    friction_dims: List[int]
    criteria_revision_norm: float
    listener_four_axis: Tuple[float, float, float, float]
    utterance_ko: str


@dataclass
class WorldDialogueReport:
    """Full record of one inter-world reciprocal conversation until sync or stall."""

    dialogue_id: str
    scale_level: int
    cell_a_id: str
    cell_b_id: str
    synced: bool
    turns_taken: int
    initial_alignment: float
    final_alignment: float
    beam_strength: float
    turns: List[DialogueTurnRecord]
    cell_b_protocol_history: List[List[float]]
    summary_ko: str


@dataclass
class CellularLearningIntrospection:
    """Self-explanation reconstructed purely from dialogue turns & protocol deltas."""

    cell_id: str
    scale_level: int
    total_turns_experienced: int
    preserved_dims: List[str]
    revised_dims: List[str]
    final_protocol: List[float]
    final_confidence: List[float]
    dialogue_trace_ko: List[str]
    narrative_ko: str


# ======================================================================
# 2. Helper Continuous Math (Zero Regressions, Zero Hard Branches)
# ======================================================================
def _compute_alignment(
    pattern_a: np.ndarray,
    pattern_b: np.ndarray,
    salience_mask: np.ndarray,
) -> Tuple[float, List[int]]:
    """
    Computes continuous weighted agreement over active dimensions plus
    explicit friction indices where mismatch exceeds perceptual threshold.
    """
    active = salience_mask > 0.15
    if not np.any(active):
        active = np.ones_like(salience_mask, dtype=bool)
        weights = np.ones_like(salience_mask, dtype=np.float64)
    else:
        weights = np.where(active, salience_mask, 0.0)

    diffs = np.abs(pattern_a - pattern_b)
    per_dim_sim = np.clip(1.0 - diffs / 1.80, 0.0, 1.0)
    score = float(np.sum(weights * per_dim_sim) / max(float(np.sum(weights)), 1e-8))
    friction_dims = [
        int(i) for i in range(len(diffs))
        if active[i] and salience_mask[i] > 0.20 and diffs[i] >= 0.25
    ]
    return score, friction_dims


def _sync_cell_voxel(cell: "LivingWorldCell") -> None:
    """Mirrors the living cell's 6D internal protocol directly into the shared CausalField."""
    cf = cell.causal_field
    if cf is None:
        return
    p = cell.internal_protocol.astype(np.float32)
    c = cell.protocol_confidence.astype(np.float32)
    triad = cell.chromatic_triad()
    pos = np.array(
        [
            float(p[0]) * 0.8 + float(cell.scale_level) * 1.2,
            float(p[1]) * 0.8 + float(cell.scale_level) * 0.4,
            float(p[2]),
            float(p[3]),
            float(p[4]),
            float(p[5]),
        ],
        dtype=np.float32,
    )
    mass = float(1.0 + 0.5 * cell.scale_level + float(np.mean(c)))
    if cell.cell_id in cf.voxels:
        vox = cf.voxels[cell.cell_id]
        vox.tensor = p.copy()
        vox.position = pos
        vox.mass = mass
        vox.chromatic_vector = np.array(triad, dtype=np.float32)
        vox.scale_level = float(cell.scale_level)
    else:
        cf.add_voxel(
            InformationVoxel(
                id=cell.cell_id,
                content={"role": cell.role_label, "scale_level": cell.scale_level},
                tensor=p.copy(),
                mass=mass,
                position=pos,
                chromatic_vector=np.array(triad, dtype=np.float32),
                scale_level=float(cell.scale_level),
            )
        )


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
# 3. Universal Scale-Invariant Cell = World Entity
# ======================================================================
class LivingWorldCell:
    """
    Single scale-invariant cognitive/biological unit.
    Whether representing a single sensory organelle (Eye/Touch), a whole organism's
    unified perception, a linguistic word symbol, or a human teacher world, it obeys
    ONLY ONE universal protocol: emit -> contrast at membrane -> revise or push back -> sync.
    """

    def __init__(
        self,
        cell_id: str,
        scale_level: int,
        role_label: str,
        initial_protocol: List[float] | np.ndarray,
        protocol_confidence: List[float] | np.ndarray,
        membrane_permeability: float = 0.78,
        sub_cells: Optional[List["LivingWorldCell"]] = None,
        causal_field: Optional[CausalField] = None,
    ) -> None:
        self.cell_id = str(cell_id)
        self.scale_level = int(scale_level)
        self.role_label = str(role_label)
        self.internal_protocol = np.clip(
            np.asarray(initial_protocol, dtype=np.float64), -1.0, 1.0
        )
        self.initial_protocol = self.internal_protocol.copy()
        self.protocol_confidence = np.clip(
            np.asarray(protocol_confidence, dtype=np.float64), 0.05, 1.0
        )
        self.initial_confidence = self.protocol_confidence.copy()
        self.membrane_permeability = float(np.clip(membrane_permeability, 0.05, 1.0))
        self.sub_cells: List[LivingWorldCell] = list(sub_cells or [])
        self.causal_field: Optional[CausalField] = causal_field
        self.communication_log: List[DialogueTurnRecord] = []
        self._last_revision_norm: float = 0.0
        self._last_friction_norm: float = 0.0

    def chromatic_triad(self) -> Tuple[float, float, float]:
        """Red (Flux / Revision), Blue (Order / Certainty), Yellow (Entropy / Friction)."""
        red = float(np.clip(self._last_revision_norm + 0.20, 0.05, 2.0))
        blue = float(np.clip(float(np.mean(self.protocol_confidence)) + 0.15, 0.05, 2.0))
        yellow = float(np.clip(self._last_friction_norm + 0.10, 0.05, 2.0))
        total = red + blue + yellow
        return (red / total, blue / total, yellow / total)

    def consolidate_from_subcells(self) -> None:
        """
        Bottom-up membrane consensus: builds a macro-world cell's grounded protocol
        and inherited certainty purely from its living constituent sub-cells.
        """
        if not self.sub_cells:
            return
        protos = np.stack([sc.internal_protocol for sc in self.sub_cells], axis=0)
        confs = np.stack([sc.protocol_confidence for sc in self.sub_cells], axis=0)

        weights = confs ** 1.6
        denom = np.maximum(np.sum(weights, axis=0), 1e-8)
        consensus_proto = np.sum(weights * protos, axis=0) / denom

        max_conf = np.max(confs, axis=0)
        mean_conf = np.mean(confs, axis=0)
        spread = np.std(protos, axis=0)
        agg_conf = np.clip(max_conf - 0.35 * spread * (1.0 - mean_conf), 0.05, 0.99)

        self.internal_protocol = np.clip(consensus_proto, -1.0, 1.0)
        self.protocol_confidence = agg_conf
        self.initial_protocol = self.internal_protocol.copy()
        self.initial_confidence = self.protocol_confidence.copy()
        _sync_cell_voxel(self)
        if self.causal_field is not None:
            for sc in self.sub_cells:
                if sc.cell_id in self.causal_field.voxels and self.cell_id in self.causal_field.voxels:
                    self.causal_field.link_voxels(sc.cell_id, self.cell_id, strength=0.92)

    def emit_signal(
        self,
        receiver_id: str,
        turn_index: int,
        signal_kind: str = "PROPOSE",
        focus_dims: Optional[List[int]] = None,
        alignment_hint: float = 0.0,
    ) -> MembraneSignal:
        """Emits internal protocol state through the outer membrane toward another cell."""
        salience = self.protocol_confidence.copy()
        if focus_dims:
            for d in focus_dims:
                salience[d] = max(float(salience[d]), 0.95)
        f_list = list(focus_dims or [])
        if signal_kind == "PROTOCOL_SYNC":
            utter_ko = f"{self.cell_id}: 경계 공명 완료 — 프로토콜 합의(alignment={alignment_hint:.2f})."
        elif signal_kind == "FRICTION_QUERY":
            f_names = [DIM_NAMES[i] for i in f_list]
            utter_ko = f"{self.cell_id}: {f_names} 차원을 수정해 보았어. 이 방향이 네 세계와 맞니?"
        elif signal_kind == "REVISION_ECHO":
            f_names = [DIM_NAMES[i] for i in f_list]
            utter_ko = f"{self.cell_id}: 나의 고신뢰 경험과 {f_names}에서 충돌해 내 기준을 지켰어."
        else:
            utter_ko = f"{self.cell_id}: 현재 내 세계 프로토콜 신호를 전달합니다."

        return MembraneSignal(
            sender_id=self.cell_id,
            receiver_id=receiver_id,
            turn_index=turn_index,
            scale_level=self.scale_level,
            signal_kind=signal_kind,
            pattern=[float(v) for v in self.internal_protocol],
            salience_mask=[float(v) for v in salience],
            chromatic_triad=self.chromatic_triad(),
            alignment_score=float(alignment_hint),
            friction_dims=f_list,
            utterance_ko=utter_ko,
        )

    def receive_and_respond(
        self, incoming: MembraneSignal
    ) -> Tuple[MembraneSignal, DialogueTurnRecord]:
        """
        Universal cell membrane processing cycle:
          1. Superpose incoming signal over `self.internal_protocol`.
          2. Measure continuous friction and 4-Axis Epistemic Attribution.
          3. Apply inertia-gated plastic revision:
             high-certainty lived anchors resist overwrite; humble uncertain dims revise smoothly.
          4. Propagate downward into uncertain sub-cells if present, and emit reciprocal reply.
        """
        in_pat = np.asarray(incoming.pattern, dtype=np.float64)
        in_mask = np.asarray(incoming.salience_mask, dtype=np.float64)

        align_before, friction_dims = _compute_alignment(
            in_pat, self.internal_protocol, in_mask
        )

        # Continuous Epistemic Inertia vs Plasticity: conf >= 0.86 anchors are immutable!
        diff = in_pat - self.internal_protocol
        plasticity = np.clip(1.0 - 1.25 * (self.protocol_confidence ** 1.40), 0.0, 1.0)
        delta = self.membrane_permeability * plasticity * in_mask * diff

        self.internal_protocol = np.clip(self.internal_protocol + delta, -1.0, 1.0)
        rev_norm = float(np.linalg.norm(delta))

        # Newly reconciled uncertain dims consolidate partial confidence smoothly
        revised_mask = np.abs(delta) > 0.06
        if np.any(revised_mask):
            self.protocol_confidence[revised_mask] = np.clip(
                self.protocol_confidence[revised_mask]
                + 0.28 * (0.80 - self.protocol_confidence[revised_mask]),
                0.05,
                0.82,
            )

        # Top-down gentle sync into sub-cells on dimensions where sub-cells lack grounding
        if self.sub_cells and rev_norm > 0.04:
            for sc in self.sub_cells:
                sc_plast = np.clip(1.0 - 1.25 * (sc.protocol_confidence ** 1.40), 0.0, 1.0)
                sc.internal_protocol = np.clip(
                    sc.internal_protocol + 0.55 * sc_plast * delta, -1.0, 1.0
                )
                _sync_cell_voxel(sc)

        # Evaluate post-revision alignment using BOTH incoming salience AND my own confident dims
        joint_mask = np.maximum(in_mask, self.protocol_confidence)
        align_after, remaining_friction = _compute_alignment(
            in_pat, self.internal_protocol, joint_mask
        )

        # 4-Axis Causal Attribution (`AGENTS.md` §6)
        internal_origin = float(np.clip(np.mean(self.protocol_confidence), 0.05, 0.99))
        external_friction = float(np.clip(1.0 - align_before, 0.0, 1.0))
        ignorance_dims = np.maximum(0.0, 1.0 - self.protocol_confidence)
        boundary_ignorance = float(np.clip(np.mean(ignorance_dims * in_mask), 0.0, 1.0))
        criteria_revision = float(np.clip(rev_norm / 1.4, 0.0, 1.0))

        self._last_revision_norm = criteria_revision
        self._last_friction_norm = external_friction
        _sync_cell_voxel(self)

        # Determine reciprocal response kind purely from physical alignment & revision state
        if align_after >= 0.90 and not remaining_friction:
            resp_kind = "PROTOCOL_SYNC"
            focus = []
        elif criteria_revision > 0.04:
            resp_kind = "FRICTION_QUERY"
            focus = remaining_friction if remaining_friction else friction_dims
        else:
            resp_kind = "REVISION_ECHO"
            focus = remaining_friction if remaining_friction else friction_dims

        reply = self.emit_signal(
            receiver_id=incoming.sender_id,
            turn_index=incoming.turn_index,
            signal_kind=resp_kind,
            focus_dims=focus,
            alignment_hint=align_after,
        )

        turn_rec = DialogueTurnRecord(
            turn_index=incoming.turn_index,
            speaker_id=incoming.sender_id,
            listener_id=self.cell_id,
            signal_kind=resp_kind,
            alignment_before=float(align_before),
            alignment_after=float(align_after),
            friction_dims=[int(i) for i in friction_dims],
            criteria_revision_norm=float(rev_norm),
            listener_four_axis=(
                internal_origin,
                external_friction,
                boundary_ignorance,
                criteria_revision,
            ),
            utterance_ko=reply.utterance_ko,
        )
        self.communication_log.append(turn_rec)
        return reply, turn_rec

    def predict_counterfactual(
        self, probe_vec: List[float] | np.ndarray, expected_sign: float
    ) -> bool:
        """Evaluates unseen physical/causal counterfactual directly via internal 6D field."""
        p = np.asarray(probe_vec, dtype=np.float64)
        response = float(np.dot(self.internal_protocol, p))
        if abs(response) < 1e-6:
            return False
        return bool((response > 0.0) == (expected_sign > 0.0))

    def introspect_how_i_learned(self) -> CellularLearningIntrospection:
        """
        Walks this cell's lived `communication_log` and compares `initial_protocol`
        against `internal_protocol` to explain *how* and *why* it learned or refused.
        """
        deltas = np.abs(self.internal_protocol - self.initial_protocol)
        preserved = [
            DIM_NAMES[i] for i in range(len(DIM_NAMES))
            if deltas[i] < 0.15 and self.protocol_confidence[i] >= 0.70
        ]
        revised = [
            f"{DIM_NAMES[i]}({self.initial_protocol[i]:+.2f}->{self.internal_protocol[i]:+.2f})"
            for i in range(len(DIM_NAMES))
            if deltas[i] >= 0.15
        ]
        trace_ko = [
            f"[T{t.turn_index}] {t.speaker_id}->{self.cell_id} ({t.signal_kind}) | "
            f"정렬={t.alignment_before:.2f}->{t.alignment_after:.2f} | "
            f"수정량={t.criteria_revision_norm:.2f} | {t.utterance_ko}"
            for t in self.communication_log
        ]
        narrative_ko = (
            f"[{self.cell_id} 자가 학습 역추적(Scale {self.scale_level})] "
            f"총 {len(self.communication_log)}회 막 간 통신 수행. "
            f"고신뢰 보존 차원={preserved}, 대화 마찰로 수정·체득한 차원={revised}."
        )
        return CellularLearningIntrospection(
            cell_id=self.cell_id,
            scale_level=self.scale_level,
            total_turns_experienced=len(self.communication_log),
            preserved_dims=preserved,
            revised_dims=revised,
            final_protocol=[float(v) for v in self.internal_protocol],
            final_confidence=[float(v) for v in self.protocol_confidence],
            dialogue_trace_ko=trace_ko,
            narrative_ko=narrative_ko,
        )


# ======================================================================
# 4. Shared CausalField Orchestrator & 3-Scale Living Curriculum
# ======================================================================
class CellularWorldDialogueField:
    """
    Houses all `LivingWorldCell` worlds across Scales 0, 1, and 2 inside ONE shared
    6D `CausalField` and runs the single invariant reciprocal conversation protocol.
    """

    def __init__(self, state_path: Optional[Path] = None) -> None:
        self.causal_field = CausalField(dimensions=6)
        self.cells: Dict[str, LivingWorldCell] = {}
        self.dialogues: Dict[str, WorldDialogueReport] = {}
        self.state_path = Path(state_path) if state_path else DEFAULT_CELLULAR_STATE_PATH

    def register_cell(self, cell: LivingWorldCell) -> LivingWorldCell:
        cell.causal_field = self.causal_field
        self.cells[cell.cell_id] = cell
        _sync_cell_voxel(cell)
        return cell

    def converse_until_sync(
        self,
        cell_a: LivingWorldCell,
        cell_b: LivingWorldCell,
        dialogue_id: str,
        max_turns: int = 8,
    ) -> WorldDialogueReport:
        """
        Executes reciprocal `MembraneSignal` exchange (`A <-> B`) until BOTH worlds
        emit `PROTOCOL_SYNC` or `max_turns` exhausts without forcing fake agreement.
        """
        turns: List[DialogueTurnRecord] = []
        b_history: List[List[float]] = [cell_b.internal_protocol.tolist()]

        msg_to_b = cell_a.emit_signal(cell_b.cell_id, turn_index=0, signal_kind="PROPOSE")
        init_align, _ = _compute_alignment(
            np.asarray(msg_to_b.pattern, dtype=np.float64),
            cell_b.internal_protocol,
            np.maximum(np.asarray(msg_to_b.salience_mask, dtype=np.float64), cell_b.protocol_confidence),
        )

        synced = False
        final_align = init_align
        turns_used = 0

        for t in range(max_turns):
            turns_used = t + 1
            msg_to_b.turn_index = t
            resp_b, rec_b = cell_b.receive_and_respond(msg_to_b)
            turns.append(rec_b)
            b_history.append(cell_b.internal_protocol.tolist())
            final_align = rec_b.alignment_after

            if msg_to_b.signal_kind == "PROTOCOL_SYNC" and resp_b.signal_kind == "PROTOCOL_SYNC":
                synced = True
                break

            resp_a, rec_a = cell_a.receive_and_respond(resp_b)
            turns.append(rec_a)
            final_align = max(final_align, rec_a.alignment_after)

            if resp_b.signal_kind == "PROTOCOL_SYNC" and resp_a.signal_kind == "PROTOCOL_SYNC":
                synced = True
                break

            msg_to_b = resp_a

        beam_strength = float(np.clip(0.95 * final_align if synced else 0.18 * final_align, 0.05, 0.99))
        if cell_a.cell_id in self.causal_field.voxels and cell_b.cell_id in self.causal_field.voxels:
            self.causal_field.link_voxels(cell_a.cell_id, cell_b.cell_id, strength=beam_strength)

        scale_lvl = max(cell_a.scale_level, cell_b.scale_level)
        status_ko = "공통 프로토콜 체득·합의 성공" if synced else "고신뢰 경계 충돌로 거짓 동화 거부 (빔 약화/절단)"
        summary_ko = (
            f"[{dialogue_id} | Scale {scale_lvl}] {cell_a.cell_id} <-> {cell_b.cell_id} | "
            f"{turns_used}턴 교신 | 정렬={init_align:.2f}->{final_align:.2f} | "
            f"빔={beam_strength:.2f} | {status_ko}"
        )

        rep = WorldDialogueReport(
            dialogue_id=dialogue_id,
            scale_level=scale_lvl,
            cell_a_id=cell_a.cell_id,
            cell_b_id=cell_b.cell_id,
            synced=synced,
            turns_taken=turns_used,
            initial_alignment=float(init_align),
            final_alignment=float(final_align),
            beam_strength=beam_strength,
            turns=turns,
            cell_b_protocol_history=b_history,
            summary_ko=summary_ko,
        )
        self.dialogues[dialogue_id] = rep
        return rep

    def run_three_scale_living_curriculum(self) -> Dict[str, Any]:
        """
        Runs ONE universal communication protocol seamlessly across Scale 0, 1, and 2:
          - Scale 0: Eye_Cell <-> Touch_Cell reconstruct an optical illusion via membrane turns.
          - Scale 1: Consolidated Elysia_Organism_World rejects false 'Red Wall' symbol
                     and agrees with truthful 'Red Illusion' symbol.
          - Scale 2: Water-anchored Elysia_Mind converses with Teacher_Electricity_World
                     and unseen Teacher_Magnetism_World (50% -> 100% counterfactual accuracy).
        """
        # --------------------------------------------------------------
        # SCALE 0: Cell <-> Cell (`Eye_Cell` <-> `Touch_Cell`)
        # --------------------------------------------------------------
        eye_cell = self.register_cell(
            LivingWorldCell(
                cell_id="Eye_Cell",
                scale_level=0,
                role_label="Visual Surface Cell",
                initial_protocol=[0.90, 0.90, 0.85, 0.85, 0.0, 0.0],
                protocol_confidence=[0.95, 0.95, 0.20, 0.20, 0.10, 0.10],
            )
        )
        touch_cell = self.register_cell(
            LivingWorldCell(
                cell_id="Touch_Cell",
                scale_level=0,
                role_label="Tactile Resistance Cell",
                initial_protocol=[0.20, 0.10, -0.92, -0.90, 0.0, 0.0],
                protocol_confidence=[0.15, 0.15, 0.96, 0.96, 0.10, 0.10],
            )
        )
        d_scale0 = self.converse_until_sync(
            eye_cell, touch_cell, dialogue_id="scale0_eye_touch", max_turns=8
        )

        # --------------------------------------------------------------
        # SCALE 1: Organism World <-> Language World
        # --------------------------------------------------------------
        organism_cell = self.register_cell(
            LivingWorldCell(
                cell_id="Elysia_Organism_World",
                scale_level=1,
                role_label="Consolidated Sensory Organism",
                initial_protocol=[0.0] * 6,
                protocol_confidence=[0.10] * 6,
                sub_cells=[eye_cell, touch_cell],
            )
        )
        organism_cell.consolidate_from_subcells()

        word_red_wall = self.register_cell(
            LivingWorldCell(
                cell_id="Language_Word_RedWall",
                scale_level=1,
                role_label="Symbolic Proposition: Red Solid Wall",
                initial_protocol=[0.90, 0.90, 0.90, 0.90, 0.0, 0.0],
                protocol_confidence=[0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
            )
        )
        word_red_illusion = self.register_cell(
            LivingWorldCell(
                cell_id="Language_Word_RedIllusion",
                scale_level=1,
                role_label="Symbolic Proposition: Red Optical Illusion",
                initial_protocol=[0.90, 0.90, -0.90, -0.90, 0.0, 0.0],
                protocol_confidence=[0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
            )
        )
        d_scale1_reject = self.converse_until_sync(
            word_red_wall, organism_cell, dialogue_id="scale1_reject_false_wall", max_turns=6
        )
        d_scale1_sync = self.converse_until_sync(
            word_red_illusion, organism_cell, dialogue_id="scale1_sync_true_illusion", max_turns=6
        )

        # --------------------------------------------------------------
        # SCALE 2: Teacher World <-> Elysia Student World
        # --------------------------------------------------------------
        water_foothold_proto = [0.0, 0.0, 0.85, 0.85, 0.90, -0.80]
        water_foothold_conf = [0.20, 0.20, 0.92, 0.22, 0.92, 0.22]

        # Domain A: Electricity
        elysia_elec = self.register_cell(
            LivingWorldCell(
                cell_id="Elysia_Mind_Elec",
                scale_level=2,
                role_label="Student World (Water Foothold -> Electricity)",
                initial_protocol=list(water_foothold_proto),
                protocol_confidence=list(water_foothold_conf),
            )
        )
        teacher_elec = self.register_cell(
            LivingWorldCell(
                cell_id="Teacher_Electricity_World",
                scale_level=2,
                role_label="Physical Law World: Electricity",
                initial_protocol=[0.0, 0.0, 0.88, -0.92, 0.92, 0.92],
                protocol_confidence=[0.20, 0.20, 0.95, 0.95, 0.95, 0.95],
            )
        )
        elec_probes: List[Tuple[str, List[float], float]] = [
            ("voltage_gradient_drives_flow", [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], +1.0),
            ("constriction_filament_resistance", [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], +1.0),
            ("open_cut_leak_test", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], -1.0),
            ("near_instant_field_propagation", [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], +1.0),
        ]
        elec_acc_before = sum(
            1 for _, vec, sign in elec_probes if elysia_elec.predict_counterfactual(vec, sign)
        ) / len(elec_probes)
        d_scale2_elec = self.converse_until_sync(
            teacher_elec, elysia_elec, dialogue_id="scale2_teacher_electricity", max_turns=8
        )
        elec_acc_after = sum(
            1 for _, vec, sign in elec_probes if elysia_elec.predict_counterfactual(vec, sign)
        ) / len(elec_probes)

        # Domain B: Unseen Magnetism
        elysia_mag = self.register_cell(
            LivingWorldCell(
                cell_id="Elysia_Mind_Mag",
                scale_level=2,
                role_label="Student World (Water Foothold -> Magnetism)",
                initial_protocol=list(water_foothold_proto),
                protocol_confidence=list(water_foothold_conf),
            )
        )
        teacher_mag = self.register_cell(
            LivingWorldCell(
                cell_id="Teacher_Magnetism_World",
                scale_level=2,
                role_label="Unseen Physical Law World: Magnetism",
                initial_protocol=[-0.75, 0.85, 0.84, -0.90, 0.88, 0.90],
                protocol_confidence=[0.92, 0.92, 0.95, 0.95, 0.95, 0.95],
            )
        )
        mag_probes: List[Tuple[str, List[float], float]] = [
            ("flux_gradient_torque", [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], +1.0),
            ("airgap_reluctance_resistance", [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], +1.0),
            ("cut_magnet_spills_monopole_test", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], -1.0),
            ("vacuum_noncontact_dipole_propagation", [0.0, 0.30, 0.0, 0.0, 0.0, 0.70], +1.0),
        ]
        mag_acc_before = sum(
            1 for _, vec, sign in mag_probes if elysia_mag.predict_counterfactual(vec, sign)
        ) / len(mag_probes)
        d_scale2_mag = self.converse_until_sync(
            teacher_mag, elysia_mag, dialogue_id="scale2_teacher_magnetism", max_turns=8
        )
        mag_acc_after = sum(
            1 for _, vec, sign in mag_probes if elysia_mag.predict_counterfactual(vec, sign)
        ) / len(mag_probes)

        # Collect introspective self-explanations across all 3 scales
        introspections = {
            cid: self.cells[cid].introspect_how_i_learned()
            for cid in (
                "Eye_Cell",
                "Touch_Cell",
                "Elysia_Organism_World",
                "Elysia_Mind_Elec",
                "Elysia_Mind_Mag",
            )
        }

        result: Dict[str, Any] = {
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "scale0_dialogue": d_scale0,
            "scale1_reject_dialogue": d_scale1_reject,
            "scale1_sync_dialogue": d_scale1_sync,
            "scale2_elec_dialogue": d_scale2_elec,
            "scale2_mag_dialogue": d_scale2_mag,
            "accuracy": {
                "electricity_before": float(elec_acc_before),
                "electricity_after": float(elec_acc_after),
                "magnetism_before": float(mag_acc_before),
                "magnetism_after": float(mag_acc_after),
            },
            "introspections": introspections,
            "causal_field_voxels": len(self.causal_field.voxels),
            "causal_field_beams": len(self.causal_field.beams),
            "state_json_path": str(self.state_path),
        }
        self.save_state(result, path=self.state_path)
        return result

    def save_state(self, curriculum_result: Dict[str, Any], path: Optional[Path] = None) -> Path:
        out_path = Path(path) if path else self.state_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "protocol_version": "1.0.0-cellular-world-dialogue",
            "timestamp_iso": curriculum_result["timestamp_iso"],
            "causal_field_voxels": curriculum_result["causal_field_voxels"],
            "causal_field_beams": curriculum_result["causal_field_beams"],
            "accuracy": curriculum_result["accuracy"],
            "cells": {
                cid: {
                    "scale_level": c.scale_level,
                    "role_label": c.role_label,
                    "initial_protocol": [float(v) for v in c.initial_protocol],
                    "final_protocol": [float(v) for v in c.internal_protocol],
                    "final_confidence": [float(v) for v in c.protocol_confidence],
                    "sub_cells": [sc.cell_id for sc in c.sub_cells],
                }
                for cid, c in self.cells.items()
            },
            "dialogues": {
                did: asdict(rep) for did, rep in self.dialogues.items()
            },
            "introspections": {
                cid: asdict(ins) for cid, ins in curriculum_result["introspections"].items()
            },
        }
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(_to_json_safe(payload), fh, ensure_ascii=False, indent=2)
        return out_path


# ======================================================================
# 5. 4-Panel Dark-Theme Cellular World Dialogue Dashboard
# ======================================================================
def render_cellular_world_dialogue_dashboard(
    field_sys: CellularWorldDialogueField,
    curriculum_result: Dict[str, Any],
    output_path: Path = DEFAULT_CELLULAR_DASHBOARD_PATH,
) -> Path:
    """
    Renders clean ASCII/English 2x2 dark-theme dashboard visualizing:
      1) Multi-Scale Cellular World Topology & Synced vs Rejected Membrane Beams
      2) Reciprocal Dialogue Alignment Curves Across All 5 Inter-World Conversations
      3) 6D Protocol Self-Revision Heatmap (Foothold Preserved vs Anomaly Revised)
      4) Unseen Counterfactual Accuracy Before vs After Dialogue (Electricity & Magnetism)
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(16.5, 11.0), dpi=150, facecolor="#0D1117")
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28, top=0.90, bottom=0.07, left=0.07, right=0.95)

    fig.suptitle(
        "ELYSIA Scale-Invariant Cellular World Communication Protocol (Cell = World, Learning = Dialogue)",
        color="#F0F6FC",
        fontsize=13.5,
        fontweight="bold",
        y=0.96,
    )

    def _style_ax(ax: plt.Axes, title: str) -> None:
        ax.set_facecolor("#161B22")
        for spine in ax.spines.values():
            spine.set_color("#30363D")
            spine.set_linewidth(1.1)
        ax.tick_params(colors="#C9D1D9", labelsize=8.5)
        ax.set_title(title, color="#F0F6FC", fontsize=10.5, fontweight="bold", pad=10)

    # ------------------------------------------------------------------
    # Panel 1 (Top-Left): Multi-Scale Cellular World Topology
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1, "1. Multi-Scale Cell = World Topology (Synced vs Torn Beams)")

    pos_map: Dict[str, Tuple[float, float]] = {
        "Eye_Cell": (1.15, 0.42),
        "Touch_Cell": (1.15, -0.42),
        "Elysia_Organism_World": (3.20, 0.0),
        "Language_Word_RedWall": (5.25, 0.46),
        "Language_Word_RedIllusion": (5.25, -0.46),
        "Elysia_Mind_Elec": (7.35, 0.42),
        "Teacher_Electricity_World": (9.35, 0.42),
        "Elysia_Mind_Mag": (7.35, -0.42),
        "Teacher_Magnetism_World": (9.35, -0.42),
    }
    short_names: Dict[str, str] = {
        "Eye_Cell": "Eye\nCell",
        "Touch_Cell": "Touch\nCell",
        "Elysia_Organism_World": "Elysia\nOrganism",
        "Language_Word_RedWall": "Word:\nRedWall",
        "Language_Word_RedIllusion": "Word:\nIllusion",
        "Elysia_Mind_Elec": "Mind\n(Elec)",
        "Teacher_Electricity_World": "Teacher:\nElec",
        "Elysia_Mind_Mag": "Mind\n(Mag)",
        "Teacher_Magnetism_World": "Teacher:\nMag",
    }

    # Sub-cell hierarchy links + live CausalField beams
    for parent_id, child_id in (("Elysia_Organism_World", "Eye_Cell"), ("Elysia_Organism_World", "Touch_Cell")):
        x1, y1 = pos_map[child_id]
        x2, y2 = pos_map[parent_id]
        ax1.plot([x1, x2], [y1, y2], color="#D2A8FF", lw=1.8, ls=":", alpha=0.85, zorder=1)

    for d_rep in field_sys.dialogues.values():
        if d_rep.cell_a_id in pos_map and d_rep.cell_b_id in pos_map:
            x1, y1 = pos_map[d_rep.cell_a_id]
            x2, y2 = pos_map[d_rep.cell_b_id]
            col = "#3FB950" if d_rep.synced else "#F85149"
            ls = "-" if d_rep.synced else "--"
            ax1.plot([x1, x2], [y1, y2], color=col, lw=2.3, ls=ls, alpha=0.90, zorder=2)
            mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            if abs(y2 - y1) < 0.05:
                my += 0.13
            badge = f"SYNC {d_rep.beam_strength:.2f}" if d_rep.synced else f"TORN {d_rep.beam_strength:.2f}"
            ax1.text(
                mx,
                my,
                badge,
                color=col,
                fontsize=7.2,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.16", facecolor="#0D1117", edgecolor=col, lw=0.8, alpha=0.94),
                zorder=4,
            )

    scale_colors = {0: "#58A6FF", 1: "#D2A8FF", 2: "#3FB950"}
    for cid, (cx, cy) in pos_map.items():
        cell = field_sys.cells[cid]
        c_col = scale_colors.get(cell.scale_level, "#F0F6FC")
        ax1.text(
            cx,
            cy,
            short_names[cid],
            color="#F0F6FC",
            fontsize=7.3,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.34", facecolor="#0D1117", edgecolor=c_col, lw=1.9),
            zorder=5,
        )

    for sx, lbl in ((1.15, "Scale 0\nCell<->Cell"), (4.22, "Scale 1\nWorld<->Word"), (8.35, "Scale 2\nTeacher<->Mind")):
        ax1.text(sx, 0.88, lbl, color="#8B949E", fontsize=8.2, fontweight="bold", ha="center", va="center")

    ax1.set_xlim(0.25, 10.25)
    ax1.set_ylim(-0.82, 1.05)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # ------------------------------------------------------------------
    # Panel 2 (Top-Right): Reciprocal Alignment Curves Across Turns
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, "2. Reciprocal Membrane Dialogue Convergence (All 3 Scales)")

    curve_specs = [
        ("scale0_eye_touch", "S0: Eye <-> Touch", "#58A6FF", "-", "o"),
        ("scale1_reject_false_wall", "S1: Reject False RedWall", "#F85149", "--", "x"),
        ("scale1_sync_true_illusion", "S1: Sync RedIllusion", "#D2A8FF", "-", "^"),
        ("scale2_teacher_electricity", "S2: Teacher <-> Elec", "#3FB950", "-", "s"),
        ("scale2_teacher_magnetism", "S2: Teacher <-> Magnetism", "#F2C94C", "-", "D"),
    ]
    for did, lbl, col, ls, mkr in curve_specs:
        rep = field_sys.dialogues[did]
        ys = [rep.initial_alignment]
        # Track progression per full round-trip
        turn_vals = [t.alignment_after for t in rep.turns if t.listener_id == rep.cell_b_id]
        ys.extend(turn_vals)
        xs = np.arange(len(ys), dtype=np.float64)
        ax2.plot(xs, ys, color=col, ls=ls, marker=mkr, ms=5.0, lw=2.0, label=lbl)

    ax2.axhline(0.90, color="#3FB950", ls=":", lw=1.2, alpha=0.8)
    ax2.text(0.08, 0.915, "Sync Threshold (0.90)", color="#3FB950", fontsize=7.8)
    ax2.set_ylim(0.15, 1.08)
    ax2.set_xlabel("Reciprocal Communication Turn (Round-Trip)", color="#C9D1D9", fontsize=9)
    ax2.set_ylabel("Cross-Membrane Protocol Alignment", color="#C9D1D9", fontsize=9)
    ax2.grid(color="#21262D", linestyle="--", linewidth=0.7, alpha=0.7)
    ax2.legend(
        loc="lower right",
        fontsize=7.9,
        facecolor="#0D1117",
        edgecolor="#30363D",
        labelcolor="#E6EDF3",
    )

    # ------------------------------------------------------------------
    # Panel 3 (Bottom-Left): 6D Protocol Before vs After Revision Heatmap
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3, "3. 6D Protocol Self-Revision: Preserved Foothold vs Revised Breakdown")

    e_cell = field_sys.cells["Elysia_Mind_Elec"]
    m_cell = field_sys.cells["Elysia_Mind_Mag"]
    o_cell = field_sys.cells["Elysia_Organism_World"]
    rows = np.stack(
        [
            e_cell.initial_protocol,
            e_cell.internal_protocol,
            m_cell.internal_protocol,
            o_cell.internal_protocol,
        ],
        axis=0,
    )
    im = ax3.imshow(rows, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(
        [
            "Water Prior\n(Initial Foothold)",
            "After Elec\nDialogue",
            "After Mag\nDialogue",
            "Organism\n(Eye+Touch)",
        ],
        fontsize=8.2,
    )
    short_dims = ["d0 Hue", "d1 Form", "d2 Resist", "d3 CutPers", "d4 Drive", "d5 Propag"]
    ax3.set_xticks(range(6))
    ax3.set_xticklabels(short_dims, fontsize=8.2)
    for ri in range(rows.shape[0]):
        for ci in range(rows.shape[1]):
            val = float(rows[ri, ci])
            t_col = "#0D1117" if abs(val) < 0.55 else "#FFFFFF"
            ax3.text(ci, ri, f"{val:+.2f}", ha="center", va="center", color=t_col, fontsize=8.3, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax3, fraction=0.045, pad=0.03)
    cbar.ax.tick_params(colors="#C9D1D9", labelsize=8)
    cbar.set_label("Signed Protocol Polarity", color="#C9D1D9", fontsize=8.5)

    # ------------------------------------------------------------------
    # Panel 4 (Bottom-Right): Unseen Counterfactual Accuracy Before vs After
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4, "4. Unseen Counterfactual Prediction Before vs After Dialogue")

    acc = curriculum_result["accuracy"]
    domains = ["Electricity\n(Scale 2 Dialogue)", "Unseen Magnetism\n(Scale 2 Dialogue)"]
    before_vals = [acc["electricity_before"] * 100.0, acc["magnetism_before"] * 100.0]
    after_vals = [acc["electricity_after"] * 100.0, acc["magnetism_after"] * 100.0]

    xp = np.arange(len(domains), dtype=np.float64)
    bw = 0.32
    b1 = ax4.bar(xp - bw / 2.0, before_vals, width=bw, color="#F85149", alpha=0.88, label="Naive Water Prior")
    b2 = ax4.bar(xp + bw / 2.0, after_vals, width=bw, color="#3FB950", alpha=0.92, label="After Membrane Dialogue")

    for b in list(b1) + list(b2):
        h = float(b.get_height())
        ax4.text(
            b.get_x() + b.get_width() / 2.0,
            h + 2.8,
            f"{h:.0f}%",
            ha="center",
            va="bottom",
            color="#F0F6FC",
            fontsize=9.5,
            fontweight="bold",
        )

    ax4.set_xticks(xp)
    ax4.set_xticklabels(domains, fontsize=9.2)
    ax4.set_yticks([0, 25, 50, 75, 100])
    ax4.set_ylim(-55, 138)
    ax4.axhline(0.0, color="#30363D", linewidth=1.0, zorder=1)
    ax4.set_ylabel("Unseen Counterfactual Accuracy (%)", color="#C9D1D9", fontsize=9)
    ax4.grid(axis="y", color="#21262D", linestyle="--", linewidth=0.7, alpha=0.7)
    ax4.legend(
        loc="upper center",
        ncol=2,
        fontsize=8.2,
        facecolor="#0D1117",
        edgecolor="#30363D",
        labelcolor="#E6EDF3",
    )

    footer_txt = (
        f"ONE Shared Class: LivingWorldCell  |  ONE Loop: converse_until_sync\n"
        f"Shared 6D CausalField: {curriculum_result['causal_field_voxels']} Cells (Voxels), "
        f"{curriculum_result['causal_field_beams']} Beams\n"
        f"Scale 0 Sync: {curriculum_result['scale0_dialogue'].synced} | "
        f"Scale 1 False RedWall Sync: {curriculum_result['scale1_reject_dialogue'].synced} | "
        f"True Illusion Sync: {curriculum_result['scale1_sync_dialogue'].synced}"
    )
    box = FancyBboxPatch(
        (0.03, 0.02),
        0.94,
        0.23,
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
        0.135,
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

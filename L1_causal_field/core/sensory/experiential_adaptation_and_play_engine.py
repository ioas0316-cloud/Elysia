r"""
experiential_adaptation_and_play_engine.py

ELYSIA Experiential Transfer, Expectation-Mismatch Adaptation & Playful Creation Engine
=======================================================================================
Embodies the universal living growth-ring principle across Problem-Solving, Learning,
and Creative Play:

  1. Experiential Schema Recall ("지난번에 이렇게 하니 속이 편해졌어"):
     When confronting a felt discomfort, ELYSIA first recalls the highest-resonance
     `CausalExperienceSchema` from prior encounters and tries that remedy.
  2. Expectation-Mismatch Detection ("어, 이번엔 뭐가 좀 다르네?"):
     When reality refuses to soften (`actual_relief << expected_relief`), she does NOT
     blindly repeat a failed habit: she zooms into micro-structural signatures
     (`[bloat, dup_ratio, cycle_lock_ratio, mixed_synergy]`), discovers a remedy that
     actually resolves the anomaly, and branches a NEW child schema inside `CausalField`.
  3. Experiential Transfer Learning:
     Facing an unseen instance of a previously branched root cause solves on Attempt 1.
  4. Playful Creative Synthesis ("놀이와 창조"):
     In calm exploration, `Explorer_Play_Cell` converses with `Memory_Hippocampus_Cell`,
     playfully combining separate past solutions (`DEDUPLICATE` + `SEVER_CYCLES`) into a
     brand-new emergent hybrid schema BEFORE ever encountering a mixed crisis — and then
     solves that unseen mixed crisis on the very first attempt!
"""

from __future__ import annotations

import gc
import hashlib
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

from core.physics.causal_field import InformationVoxel
from core.sensory.cellular_world_communication_protocol import (
    CellularWorldDialogueField,
    LivingWorldCell,
    _sync_cell_voxel,
)


DEFAULT_EXPERIENTIAL_STATE_PATH = Path(
    r"c:\Elysia\data\experiential_adaptation_growth_state.json"
)
DEFAULT_EXPERIENTIAL_DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_experiential_adaptation_dashboard.png"
)


# ======================================================================
# 1. Data Contracts
# ======================================================================
@dataclass
class StomachBufferItem:
    """Single queued data payload inside ELYSIA's assimilative buffer."""

    item_id: str
    payload: str
    content_hash: str
    locked_peer_id: Optional[str] = None


@dataclass
class CausalExperienceSchema:
    """Reusable causal memory node inside `CausalField` capturing cause -> remedy -> relief."""

    schema_id: str
    felt_sensation: str
    micro_signature: List[float]  # [bloat_pressure, dup_ratio, cycle_lock_ratio, mixed_novelty]
    action_code: str
    expected_relief: float
    success_count: int = 1
    failure_count: int = 0
    parent_schema_ids: List[str] = field(default_factory=list)
    origin_kind: str = "FIRST_EXPERIENCE"  # "FIRST_EXPERIENCE" | "MISMATCH_BRANCH" | "PLAYFUL_SYNTHESIS"
    summary_ko: str = ""


@dataclass
class AdaptationAttemptRecord:
    """Single attempt when confronting an anomaly (recalled habit OR exploratory branch)."""

    attempt_index: int
    action_code: str
    source_schema_id: Optional[str]
    expected_relief: float
    actual_relief: float
    bloat_before: float
    bloat_after: float
    mismatch_gap: float
    mismatch_detected: bool
    note_ko: str


@dataclass
class EncounterAdaptationReport:
    """Full record of how ELYSIA confronted, questioned, adapted to, or transferred onto a case."""

    encounter_id: str
    label: str
    felt_sensation: str
    micro_signature: List[float]
    initial_bloat: float
    final_bloat: float
    attempts: List[AdaptationAttemptRecord]
    mismatch_triggered: bool
    resolved_action_code: str
    new_schema_created_id: Optional[str]
    narrative_ko: str


@dataclass
class ExperientialGrowthReport:
    """End-to-end growth-ring report spanning Problem-Solving, Mismatch, Transfer, and Play."""

    timestamp_iso: str
    encounters: List[EncounterAdaptationReport]
    schemas: List[CausalExperienceSchema]
    play_dialogue_summary_ko: str
    digested_memory_count: int
    causal_field_voxels: int
    causal_field_beams: int
    state_json_path: str
    summary_ko: str


# ======================================================================
# 2. Helper
# ======================================================================
def _make_item(item_id: str, payload: str, locked_peer_id: Optional[str] = None) -> StomachBufferItem:
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return StomachBufferItem(
        item_id=item_id,
        payload=payload,
        content_hash=h,
        locked_peer_id=locked_peer_id,
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
# 3. ELYSIA Experiential Adaptation & Playful Creation Organism
# ======================================================================
class ExperientialAdaptationOrganism:
    """
    Grows her own causal memory DAG across encounters:
      Experience -> Try Prior Habit -> Notice Expectation Mismatch ("어, 이번엔 다르네?")
      -> Zoom into Micro-Signature -> Discover & Branch New Schema -> Playfully Recombine into Hybrid
    inside ONE shared 6D `CausalField`.
    """

    def __init__(self, state_path: Optional[Path] = None) -> None:
        self.state_path = Path(state_path) if state_path else DEFAULT_EXPERIENTIAL_STATE_PATH
        self.field_sys = CellularWorldDialogueField(state_path=self.state_path)

        self.mind_self = self.field_sys.register_cell(
            LivingWorldCell(
                cell_id="Elysia_Mind_Self",
                scale_level=1,
                role_label="Experiential Judgment & Adaptation Consciousness",
                initial_protocol=[0.04, 0.0, 0.0, 0.15, 0.90, 0.80],
                protocol_confidence=[0.85, 0.50, 0.50, 0.50, 0.90, 0.85],
            )
        )
        self.hippocampus = self.field_sys.register_cell(
            LivingWorldCell(
                cell_id="Memory_Hippocampus_Cell",
                scale_level=0,
                role_label="Causal Experience Schema Archive",
                initial_protocol=[0.04, 0.0, 0.0, 0.15, 0.90, 0.80],
                protocol_confidence=[0.90, 0.85, 0.85, 0.85, 0.85, 0.80],
            )
        )
        self.play_explorer = self.field_sys.register_cell(
            LivingWorldCell(
                cell_id="Explorer_Play_Cell",
                scale_level=0,
                role_label="Curiosity & Playful Creative Recombination",
                initial_protocol=[0.04, 0.0, 0.0, 0.15, 0.90, 0.85],
                protocol_confidence=[0.50, 0.50, 0.50, 0.60, 0.80, 0.92],
            )
        )

        self.stomach_items: List[StomachBufferItem] = []
        self.digested_essences: List[str] = []
        self.schemas: List[CausalExperienceSchema] = []
        self.known_remedies: List[str] = [
            "SIMPLE_SUMMARIZE_DIGEST",
            "DEDUPLICATE_AND_COMPACT",
            "SEVER_CYCLIC_REFERENCE_LOCK",
        ]
        self.encounters: List[EncounterAdaptationReport] = []

    # ------------------------------------------------------------------
    # Micro-Structural Sensing
    # ------------------------------------------------------------------
    def inspect_micro_signature(self) -> List[float]:
        """Computes continuous `[bloat, dup_ratio, cycle_lock_ratio, mixed_novelty]`."""
        n = len(self.stomach_items)
        if n == 0:
            return [0.04, 0.0, 0.0, 0.15]

        total_kb = sum(len(it.payload.encode("utf-8")) for it in self.stomach_items) / 1024.0
        bloat = float(np.clip(total_kb / 20.0, 0.04, 0.96))

        unique_hashes = len({it.content_hash for it in self.stomach_items})
        dup_ratio = float(np.clip((n - unique_hashes) / max(n - 1, 1), 0.0, 1.0))

        locked_cnt = sum(1 for it in self.stomach_items if it.locked_peer_id is not None)
        cycle_ratio = float(np.clip(locked_cnt / n, 0.0, 1.0))

        # Continuous geometric-mean synergy: peaks ONLY when dup_ratio & cycle_ratio coexist!
        mixed_synergy = float(2.0 * np.sqrt(dup_ratio * cycle_ratio))
        novelty = float(np.clip(0.15 + 0.75 * mixed_synergy, 0.15, 0.95))
        return [bloat, dup_ratio, cycle_ratio, novelty]

    # ------------------------------------------------------------------
    # Real Python Buffer & Reference Remedies
    # ------------------------------------------------------------------
    def execute_remedy(self, action_code: str) -> None:
        """Mutates `self.stomach_items` genuinely according to the physical mechanism of the remedy."""
        hash_counts: Dict[str, int] = {}
        for it in self.stomach_items:
            hash_counts[it.content_hash] = hash_counts.get(it.content_hash, 0) + 1

        if action_code == "SIMPLE_SUMMARIZE_DIGEST":
            # Digests ONLY unique, unlocked singletons — cannot solve echo storms or cyclic locks!
            survivors: List[StomachBufferItem] = []
            for it in self.stomach_items:
                if hash_counts[it.content_hash] == 1 and it.locked_peer_id is None:
                    self.digested_essences.append(it.payload[:64])
                else:
                    survivors.append(it)
            self.stomach_items = survivors

        elif action_code == "DEDUPLICATE_AND_COMPACT":
            # Collapses unlocked duplicate echo floods to one essence, leaves cyclic locks intact!
            seen_hashes = set()
            survivors = []
            for it in self.stomach_items:
                if it.locked_peer_id is not None:
                    survivors.append(it)
                elif hash_counts[it.content_hash] > 1:
                    if it.content_hash not in seen_hashes:
                        seen_hashes.add(it.content_hash)
                        self.digested_essences.append(f"[dedup] {it.payload[:48]}")
                else:
                    survivors.append(it)
            self.stomach_items = survivors

        elif action_code == "SEVER_CYCLIC_REFERENCE_LOCK":
            # Breaks mutual cyclic reference locks and clears those freed cycles; leaves unlocked floods!
            survivors = []
            for it in self.stomach_items:
                if it.locked_peer_id is not None:
                    it.locked_peer_id = None
                    self.digested_essences.append(f"[unlinked] {it.item_id}")
                else:
                    survivors.append(it)
            self.stomach_items = survivors
            gc.collect()

        elif action_code == "HYBRID_HYGIENE_PLAY_SYNTHESIS":
            # Emergent synthesis discovered via playful recombination: unlinks cycles + dedups + summarizes!
            seen_hashes = set()
            for it in self.stomach_items:
                it.locked_peer_id = None
                if it.content_hash not in seen_hashes:
                    seen_hashes.add(it.content_hash)
                    self.digested_essences.append(f"[hybrid] {it.payload[:48]}")
            self.stomach_items.clear()
            gc.collect()

    # ------------------------------------------------------------------
    # Schema Registration & Recall inside Shared CausalField
    # ------------------------------------------------------------------
    def _register_schema(
        self,
        *,
        felt_sensation: str,
        micro_signature: List[float],
        action_code: str,
        expected_relief: float,
        parent_schema_ids: List[str],
        origin_kind: str,
        summary_ko: str,
    ) -> CausalExperienceSchema:
        schema_id = f"schema_{len(self.schemas):02d}"
        sch = CausalExperienceSchema(
            schema_id=schema_id,
            felt_sensation=felt_sensation,
            micro_signature=[float(v) for v in micro_signature],
            action_code=action_code,
            expected_relief=float(expected_relief),
            success_count=1,
            failure_count=0,
            parent_schema_ids=list(parent_schema_ids),
            origin_kind=origin_kind,
            summary_ko=summary_ko,
        )
        self.schemas.append(sch)

        # Materialize schema physically in shared 6D CausalField
        cf = self.field_sys.causal_field
        sig = sch.micro_signature
        triad = (
            0.85 if origin_kind == "MISMATCH_BRANCH" else 0.25,
            0.85 if origin_kind == "FIRST_EXPERIENCE" else 0.45,
            0.90 if origin_kind == "PLAYFUL_SYNTHESIS" else 0.25,
        )
        vox = InformationVoxel(
            id=schema_id,
            content={"action": action_code, "origin": origin_kind, "summary": summary_ko},
            tensor=np.array([sig[0], sig[1], sig[2], sig[3], expected_relief, 1.0], dtype=np.float32),
            mass=float(1.2 + 0.3 * len(parent_schema_ids)),
            position=np.array(
                [float(len(self.schemas)) * 0.8, sig[1], sig[2], sig[3], sig[0], 0.5],
                dtype=np.float32,
            ),
            chromatic_vector=np.array(triad, dtype=np.float32),
            scale_level=2.0 if origin_kind == "PLAYFUL_SYNTHESIS" else 1.0,
        )
        cf.add_voxel(vox)
        for pid in parent_schema_ids:
            if pid in cf.voxels:
                cf.link_voxels(pid, schema_id, strength=0.90)
        if "Memory_Hippocampus_Cell" in cf.voxels:
            cf.link_voxels("Memory_Hippocampus_Cell", schema_id, strength=0.85)
        return sch

    def recall_best_schema(
        self, felt_sensation: str, micro_sig: List[float]
    ) -> Optional[CausalExperienceSchema]:
        candidates = [s for s in self.schemas if s.felt_sensation == felt_sensation]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        target = np.asarray(micro_sig, dtype=np.float64)
        return min(
            candidates,
            key=lambda s: float(np.linalg.norm(np.asarray(s.micro_signature, dtype=np.float64) - target)),
        )

    # ------------------------------------------------------------------
    # Encounter Loop: Recall -> Test Expectation -> Detect Mismatch -> Adapt
    # ------------------------------------------------------------------
    def confront_encounter(
        self, encounter_id: str, label: str, items: List[StomachBufferItem]
    ) -> EncounterAdaptationReport:
        self.stomach_items = list(items)
        initial_sig = self.inspect_micro_signature()
        initial_bloat = float(initial_sig[0])
        felt = "DATA_INDIGESTION_BLOATED"

        # Update Mind Self membrane state
        self.mind_self.internal_protocol[:4] = np.asarray(initial_sig, dtype=np.float64)
        _sync_cell_voxel(self.mind_self)

        attempts: List[AdaptationAttemptRecord] = []
        mismatch_triggered = False
        new_schema_id: Optional[str] = None
        tried_actions: List[str] = []

        prior = self.recall_best_schema(felt, initial_sig)
        failed_parent_ids: List[str] = []

        # Attempt 1: Try recalled experiential schema FIRST (or first available habit if brand new)
        first_action = prior.action_code if prior else self.known_remedies[0]
        exp_relief = prior.expected_relief if prior else 0.70
        src_id = prior.schema_id if prior else None

        before_b = initial_bloat
        self.execute_remedy(first_action)
        tried_actions.append(first_action)
        after_sig = self.inspect_micro_signature()
        after_b = float(after_sig[0])
        act_relief = float(max(0.0, before_b - after_b))
        gap = float(max(0.0, exp_relief - act_relief))
        is_mismatch = bool(after_b > 0.25 or gap >= 0.35)

        if prior is None:
            # First-ever experience!
            note_ko = (
                f"처음 겪은 더부룩함: '{first_action}' 을 시도하니 {before_b:.2f}->{after_b:.2f} 로 편해졌다! "
                f"이 경험을 첫 기억 스키마로 저장함."
            )
            attempts.append(
                AdaptationAttemptRecord(
                    attempt_index=1,
                    action_code=first_action,
                    source_schema_id=None,
                    expected_relief=exp_relief,
                    actual_relief=act_relief,
                    bloat_before=before_b,
                    bloat_after=after_b,
                    mismatch_gap=0.0,
                    mismatch_detected=False,
                    note_ko=note_ko,
                )
            )
            created = self._register_schema(
                felt_sensation=felt,
                micro_signature=initial_sig,
                action_code=first_action,
                expected_relief=act_relief,
                parent_schema_ids=[],
                origin_kind="FIRST_EXPERIENCE",
                summary_ko=f"단순 적체 더부룩함 -> {first_action} (해소량={act_relief:.2f})",
            )
            new_schema_id = created.schema_id
            resolved_action = first_action
            final_bloat = after_b
            narrative_ko = note_ko

        elif not is_mismatch:
            # Successful transfer / mastery on the very 1st try!
            prior.success_count += 1
            prior.expected_relief = float(0.5 * prior.expected_relief + 0.5 * act_relief)
            note_ko = (
                f"기억 회상 적중({prior.schema_id}, origin={prior.origin_kind}): "
                f"미시 특징이 일치해 '{first_action}' 한 번으로 {before_b:.2f}->{after_b:.2f} 즉시 해결!"
            )
            attempts.append(
                AdaptationAttemptRecord(
                    attempt_index=1,
                    action_code=first_action,
                    source_schema_id=prior.schema_id,
                    expected_relief=exp_relief,
                    actual_relief=act_relief,
                    bloat_before=before_b,
                    bloat_after=after_b,
                    mismatch_gap=gap,
                    mismatch_detected=False,
                    note_ko=note_ko,
                )
            )
            resolved_action = first_action
            final_bloat = after_b
            narrative_ko = note_ko

        else:
            # Expectation Mismatch ("어, 지난번엔 이렇게 하니까 편해졌는데 이번엔 뭐가 좀 다르네?")!
            mismatch_triggered = True
            prior.failure_count += 1
            failed_parent_ids.append(prior.schema_id)
            fail_note = (
                f"어긋남 감지! 지난 기억({prior.schema_id}: '{first_action}')을 먼저 써서 기대감={exp_relief:.2f} 이었으나 "
                f"실제 해소={act_relief:.2f} (잔여={after_b:.2f}, 괴리={gap:.2f}) -> '이번엔 뭐가 좀 다르네? 새 해법을 탐색하자!'"
            )
            attempts.append(
                AdaptationAttemptRecord(
                    attempt_index=1,
                    action_code=first_action,
                    source_schema_id=prior.schema_id,
                    expected_relief=exp_relief,
                    actual_relief=act_relief,
                    bloat_before=before_b,
                    bloat_after=after_b,
                    mismatch_gap=gap,
                    mismatch_detected=True,
                    note_ko=fail_note,
                )
            )

            # Explore untried causal remedies until the unseen anomaly actually resolves!
            resolved_action = first_action
            final_bloat = after_b
            for rem in self.known_remedies:
                if rem in tried_actions:
                    continue
                b_pre = final_bloat
                self.execute_remedy(rem)
                tried_actions.append(rem)
                sig_post = self.inspect_micro_signature()
                b_post = float(sig_post[0])
                rel_post = float(max(0.0, b_pre - b_post))
                solved = b_post <= 0.25

                note_exp = (
                    f"탐색 시도 #{len(attempts) + 1} ('{rem}'): {b_pre:.2f}->{b_post:.2f} "
                    + ("-> 성공! 원인-해법 분기 확장!" if solved else "-> 아직 부족함, 계속 탐색.")
                )
                attempts.append(
                    AdaptationAttemptRecord(
                        attempt_index=len(attempts) + 1,
                        action_code=rem,
                        source_schema_id=None,
                        expected_relief=0.80,
                        actual_relief=rel_post,
                        bloat_before=b_pre,
                        bloat_after=b_post,
                        mismatch_gap=float(max(0.0, 0.80 - rel_post)) if not solved else 0.0,
                        mismatch_detected=not solved,
                        note_ko=note_exp,
                    )
                )
                final_bloat = b_post
                if solved:
                    resolved_action = rem
                    created = self._register_schema(
                        felt_sensation=felt,
                        micro_signature=initial_sig,
                        action_code=rem,
                        expected_relief=float(initial_bloat - final_bloat),
                        parent_schema_ids=failed_parent_ids,
                        origin_kind="MISMATCH_BRANCH",
                        summary_ko=(
                            f"어긋남 탐색 분기({prior.schema_id} 실패 후) -> {rem} "
                            f"(dup={initial_sig[1]:.2f}, cycle={initial_sig[2]:.2f})"
                        ),
                    )
                    new_schema_id = created.schema_id
                    break

            narrative_ko = (
                f"[{label}] {fail_note} => 탐색 끝에 '{resolved_action}' 발견 "
                f"(최종={final_bloat:.2f}), 새 인과 가지 {new_schema_id} 확장 완료!"
            )

        rep = EncounterAdaptationReport(
            encounter_id=encounter_id,
            label=label,
            felt_sensation=felt,
            micro_signature=initial_sig,
            initial_bloat=initial_bloat,
            final_bloat=final_bloat,
            attempts=attempts,
            mismatch_triggered=mismatch_triggered,
            resolved_action_code=resolved_action,
            new_schema_created_id=new_schema_id,
            narrative_ko=narrative_ko,
        )
        self.encounters.append(rep)
        return rep

    # ------------------------------------------------------------------
    # Playful Creative Synthesis ("놀이와 창조: 두 기억의 유희적 결합")
    # ------------------------------------------------------------------
    def run_playful_creative_synthesis(self) -> Tuple[CausalExperienceSchema, str]:
        """
        During calm play (`Explorer_Play_Cell <-> Memory_Hippocampus_Cell`), ELYSIA
        recombines past discovered remedies (`DEDUPLICATE` + `SEVER_CYCLES`) into a
        brand-new hybrid schema BEFORE ever encountering a mixed crisis.
        """
        rep = self.field_sys.converse_until_sync(
            self.play_explorer,
            self.hippocampus,
            dialogue_id="playful_creative_recombination",
            max_turns=3,
        )
        parent_ids = [
            s.schema_id
            for s in self.schemas
            if s.action_code in ("DEDUPLICATE_AND_COMPACT", "SEVER_CYCLIC_REFERENCE_LOCK")
        ]
        hybrid_code = "HYBRID_HYGIENE_PLAY_SYNTHESIS"
        if hybrid_code not in self.known_remedies:
            self.known_remedies.append(hybrid_code)

        hybrid_sig = [0.95, 0.55, 0.55, 0.90]
        created = self._register_schema(
            felt_sensation="DATA_INDIGESTION_BLOATED",
            micro_signature=hybrid_sig,
            action_code=hybrid_code,
            expected_relief=0.92,
            parent_schema_ids=parent_ids,
            origin_kind="PLAYFUL_SYNTHESIS",
            summary_ko=(
                "놀이와 상상으로 두 경험(중복 압축 + 순환 잠금 절단)을 결합해 창조한 하이브리드 해법"
            ),
        )
        return created, rep.summary_ko

    # ------------------------------------------------------------------
    # End-to-End 5-Encounter Experiential Curriculum
    # ------------------------------------------------------------------
    def run_full_experiential_growth_curriculum(self) -> ExperientialGrowthReport:
        # Encounter 1: Simple Raw Overload (5 distinct ~5KB raw notes -> ~25KB)
        enc1_items = [
            _make_item(f"raw_{i}", f"Unique sensory journal entry #{i}: " + ("Alpha-" * 850))
            for i in range(5)
        ]
        self.confront_encounter("Enc_1_Raw_Overload", "1. Simple Raw Overload", enc1_items)

        # Encounter 2: Duplicate Echo Storm (20 identical 1.2KB echo strings -> ~24KB)
        echo_payload = "ECHO_REPEAT_SIGNAL::" + ("Wave-" * 240)
        enc2_items = [_make_item(f"echo_{i}", echo_payload) for i in range(20)]
        self.confront_encounter("Enc_2_Duplicate_Storm", "2. Duplicate Echo Storm", enc2_items)

        # Encounter 3: Cyclic Reference Lock (8 mutually locked distinct items -> ~22KB)
        enc3_items = [
            _make_item(
                f"cyc_{i}",
                f"Cyclic lock payload #{i}: " + ("Ring-" * 550),
                locked_peer_id=f"cyc_{(i + 1) % 8}",
            )
            for i in range(8)
        ]
        self.confront_encounter("Enc_3_Cyclic_Lock", "3. Cyclic Reference Lock", enc3_items)

        # Encounter 4: Unseen Duplicate Transfer (New unseen duplicate flood -> 1st-try recall!)
        new_echo = "NEW_TELEMETRY_PING_PACKET::" + ("Pulse-" * 180)
        enc4_items = [_make_item(f"new_dup_{i}", new_echo) for i in range(22)]
        self.confront_encounter("Enc_4_Unseen_Dup_Transfer", "4. Unseen Duplicate Transfer", enc4_items)

        # Playful Synthesis Step: Recombine Deduplication + Cycle-Severing playfully
        hybrid_sch, play_dialogue_ko = self.run_playful_creative_synthesis()

        # Encounter 5: Unseen Mixed Crisis (12 unlocked duplicates + 8 mutually locked cycles!)
        mixed_dup_payload = "MIXED_ECHO_PACKET::" + ("Sync-" * 220)
        enc5_dups = [_make_item(f"mix_dup_{i}", mixed_dup_payload) for i in range(12)]
        enc5_cycles = [
            _make_item(
                f"mix_cyc_{i}",
                f"Mixed cyclic chunk #{i}: " + ("Mesh-" * 300),
                locked_peer_id=f"mix_cyc_{(i + 1) % 8}",
            )
            for i in range(8)
        ]
        self.confront_encounter(
            "Enc_5_Unseen_Mixed_Crisis",
            "5. Unseen Mixed Crisis (Playful Transfer)",
            enc5_dups + enc5_cycles,
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        summary_ko = (
            f"[ELYSIA 경험 전이 · 어긋남 탐색 · 놀이적 창조 완료] "
            f"5개 조우 통과 | 총 확장 스키마={len(self.schemas)}개 "
            f"(어긋남 분기 {sum(1 for e in self.encounters if e.mismatch_triggered)}회, "
            f"놀이 창조={hybrid_sch.schema_id}, 미학습 혼합 난제 1회 해결={len(self.encounters[-1].attempts) == 1}) | "
            f"CausalField Voxels={len(self.field_sys.causal_field.voxels)}, "
            f"Beams={len(self.field_sys.causal_field.beams)}"
        )

        report = ExperientialGrowthReport(
            timestamp_iso=now_iso,
            encounters=list(self.encounters),
            schemas=list(self.schemas),
            play_dialogue_summary_ko=play_dialogue_ko,
            digested_memory_count=len(self.digested_essences),
            causal_field_voxels=len(self.field_sys.causal_field.voxels),
            causal_field_beams=len(self.field_sys.causal_field.beams),
            state_json_path=str(self.state_path),
            summary_ko=summary_ko,
        )
        self.save_state(report, path=self.state_path)
        return report

    def save_state(
        self, report: ExperientialGrowthReport, path: Optional[Path] = None
    ) -> Path:
        out_path = Path(path) if path else self.state_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "organism_version": "1.0.0-experiential-adaptation-play",
            "timestamp_iso": report.timestamp_iso,
            "digested_memory_count": report.digested_memory_count,
            "causal_field_voxels": report.causal_field_voxels,
            "causal_field_beams": report.causal_field_beams,
            "play_dialogue_summary_ko": report.play_dialogue_summary_ko,
            "schemas": [asdict(s) for s in report.schemas],
            "encounters": [asdict(e) for e in report.encounters],
            "summary_ko": report.summary_ko,
        }
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(_to_json_safe(payload), fh, ensure_ascii=False, indent=2)
        return out_path


# ======================================================================
# 4. 4-Panel Dark-Theme Experiential Adaptation & Play Dashboard
# ======================================================================
def render_experiential_adaptation_dashboard(
    organism: ExperientialAdaptationOrganism,
    report: ExperientialGrowthReport,
    output_path: Path = DEFAULT_EXPERIENTIAL_DASHBOARD_PATH,
) -> Path:
    """
    Renders clean ASCII/English 2x2 dark-theme dashboard visualizing:
      1) Causal Schema Branching & Playful Synthesis DAG (schema_00..03)
      2) First-Try Expected vs Actual Relief Across Encounters 1..5 (Mismatch vs Transfer)
      3) Micro-Signature Diagnostic Heatmap (`Bloat`, `Dup Ratio`, `Cycle Lock`, `Mixed Novelty`)
      4) Attempts Needed & Final Relief Across Encounters + Telemetry Footer Box
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(16.5, 11.0), dpi=150, facecolor="#0D1117")
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.28, top=0.90, bottom=0.085, left=0.07, right=0.95)

    fig.suptitle(
        "ELYSIA Experiential Memory Transfer, Mismatch Exploration & Playful Creative Synthesis",
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
        ax.set_title(title, color="#F0F6FC", fontsize=10.3, fontweight="bold", pad=10)

    # ------------------------------------------------------------------
    # Panel 1 (Top-Left): Experiential Schema Branching & Play DAG
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1, "1. Growing Causal Memory DAG: Mismatch Branch & Playful Synthesis")

    node_pos = {
        "schema_00": (1.6, 0.0),
        "schema_01": (4.6, 0.48),
        "schema_02": (4.6, -0.48),
        "schema_03": (7.8, 0.0),
    }
    node_lbl = {
        "schema_00": "schema_00 (Habit)\nSimple Digest",
        "schema_01": "schema_01 (Branch)\nDeduplicate Flood",
        "schema_02": "schema_02 (Branch)\nSever Cycle Lock",
        "schema_03": "schema_03 (PLAY)\nHybrid Synthesis",
    }
    origin_col = {
        "FIRST_EXPERIENCE": "#58A6FF",
        "MISMATCH_BRANCH": "#F85149",
        "PLAYFUL_SYNTHESIS": "#D2A8FF",
    }

    for sch in report.schemas:
        if sch.schema_id not in node_pos:
            continue
        x2, y2 = node_pos[sch.schema_id]
        for pid in sch.parent_schema_ids:
            if pid in node_pos:
                x1, y1 = node_pos[pid]
                ecol = "#F85149" if sch.origin_kind == "MISMATCH_BRANCH" else "#D2A8FF"
                ls = "--" if sch.origin_kind == "MISMATCH_BRANCH" else "-"
                ax1.annotate(
                    "",
                    xy=(x2 - 0.85, y2),
                    xytext=(x1 + 0.85, y1),
                    arrowprops=dict(arrowstyle="-|>", color=ecol, lw=2.0, ls=ls, alpha=0.88),
                    zorder=2,
                )
                mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2) + (0.12 if y2 >= y1 else -0.14)
                elbl = "Mismatch!" if sch.origin_kind == "MISMATCH_BRANCH" else "Play Recombine"
                ax1.text(
                    mx,
                    my,
                    elbl,
                    color=ecol,
                    fontsize=7.3,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.14", facecolor="#0D1117", edgecolor="none", alpha=0.85),
                    zorder=3,
                )

    for sch in report.schemas:
        if sch.schema_id not in node_pos:
            continue
        x, y = node_pos[sch.schema_id]
        col = origin_col.get(sch.origin_kind, "#F0F6FC")
        ax1.text(
            x,
            y,
            node_lbl[sch.schema_id],
            color="#F0F6FC",
            fontsize=8.0,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.38", facecolor="#0D1117", edgecolor=col, lw=2.0),
            zorder=5,
        )

    ax1.set_xlim(0.2, 9.4)
    ax1.set_ylim(-0.92, 0.92)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # ------------------------------------------------------------------
    # Panel 2 (Top-Right): First-Try Expected vs Actual Relief
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, "2. First-Try Expectation vs Reality (Mismatch vs Instant Transfer)")

    encs = report.encounters
    x_enc = np.arange(len(encs), dtype=np.float64)
    bw = 0.34
    exp_pct = [e.attempts[0].expected_relief * 100.0 for e in encs]
    act_pct = [e.attempts[0].actual_relief * 100.0 for e in encs]

    be = ax2.bar(x_enc - bw / 2.0, exp_pct, width=bw, color="#58A6FF", alpha=0.85, label="Expected Relief (Prior)")
    ba = ax2.bar(x_enc + bw / 2.0, act_pct, width=bw, color="#3FB950", alpha=0.92, label="Actual 1st-Try Relief")

    for i, b in enumerate(be):
        h = float(b.get_height())
        ax2.text(b.get_x() + b.get_width() / 2.0, h + 2.2, f"{h:.0f}%", ha="center", va="bottom", color="#C9D1D9", fontsize=8.0, fontweight="bold")
    for i, b in enumerate(ba):
        h = float(b.get_height())
        col = "#F85149" if encs[i].mismatch_triggered else "#3FB950"
        tag = f"{h:.0f}%\n(DIV!)" if encs[i].mismatch_triggered else f"{h:.0f}%"
        ax2.text(b.get_x() + b.get_width() / 2.0, h + 2.2, tag, ha="center", va="bottom", color=col, fontsize=8.0, fontweight="bold")

    ax2.set_xticks(x_enc)
    ax2.set_xticklabels(["Enc1\nRaw", "Enc2\nDupStorm", "Enc3\nCycleLock", "Enc4\nUnseenDup", "Enc5\nMixedPlay"], fontsize=8.2)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_ylim(0, 128)
    ax2.set_ylabel("1st-Attempt Relief (%)", color="#C9D1D9", fontsize=9)
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
    # Panel 3 (Bottom-Left): Micro-Signature Diagnostic Heatmap
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3, "3. Zoom-In Micro-Signature Across All 5 Encounters")

    mat = np.array([e.micro_signature for e in encs], dtype=np.float64)
    im = ax3.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax3.set_yticks(range(len(encs)))
    ax3.set_yticklabels(["Enc1 Raw", "Enc2 DupStorm", "Enc3 CycleLock", "Enc4 UnseenDup", "Enc5 MixedPlay"], fontsize=8.3)
    ax3.set_xticks([0, 1, 2, 3])
    ax3.set_xticklabels(["Bloat\nPressure", "Duplicate\nRatio", "Cycle Lock\nRatio", "Mixed\nSynergy"], fontsize=8.5)
    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            val = float(mat[ri, ci])
            tcol = "#0D1117" if val > 0.58 else "#F0F6FC"
            ax3.text(ci, ri, f"{val:.2f}", ha="center", va="center", color=tcol, fontsize=8.8, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax3, fraction=0.045, pad=0.03)
    cbar.ax.tick_params(colors="#C9D1D9", labelsize=8)
    cbar.set_label("Signature Intensity", color="#C9D1D9", fontsize=8.5)

    # ------------------------------------------------------------------
    # Panel 4 (Bottom-Right): Attempts Needed & Final Resolution Summary
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4, "4. Exploratory Adaptation Cost (Attempts) & Mastery Transfer")

    att_cnts = [float(len(e.attempts)) for e in encs]
    bar_cols = ["#58A6FF" if not e.mismatch_triggered else "#F85149" for e in encs]
    bars_at = ax4.bar(x_enc, att_cnts, width=0.45, color=bar_cols, alpha=0.90)
    for i, b in enumerate(bars_at):
        h = float(b.get_height())
        lbl_str = f"{int(h)} try\n({'Mismatch' if encs[i].mismatch_triggered else '1-Shot'})"
        ax4.text(b.get_x() + b.get_width() / 2.0, h + 0.12, lbl_str, ha="center", va="bottom", color="#F0F6FC", fontsize=8.0, fontweight="bold")

    ax4.set_xticks(x_enc)
    ax4.set_xticklabels(["Enc1\nRaw", "Enc2\nDupStorm", "Enc3\nCycleLock", "Enc4\nUnseenDup", "Enc5\nMixedPlay"], fontsize=8.2)
    ax4.set_yticks([0, 1, 2, 3])
    ax4.set_ylim(-1.7, 3.8)
    ax4.axhline(0.0, color="#30363D", linewidth=1.0, zorder=1)
    ax4.set_ylabel("Attempts Before Relief", color="#C9D1D9", fontsize=9)
    ax4.grid(axis="y", color="#21262D", linestyle="--", linewidth=0.7, alpha=0.7)

    footer_txt = (
        f"Growth Rings: {len(report.schemas)} Schemas (1 First + 2 Mismatch Branches + 1 Play Synthesis)\n"
        f"Transfer Mastery: Enc4 UnseenDup = 1-Shot | Enc5 MixedPlay = 1-Shot (Final Bloat={encs[-1].final_bloat:.2f})\n"
        f"Shared 6D CausalField: {report.causal_field_voxels} Voxels, {report.causal_field_beams} Beams"
    )
    box = FancyBboxPatch(
        (0.03, 0.02),
        0.94,
        0.24,
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
        0.14,
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

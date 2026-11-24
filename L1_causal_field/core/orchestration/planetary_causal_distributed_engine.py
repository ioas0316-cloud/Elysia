"""
ELYSIA Planetary Scale-Rotor Distributed Causal Engine (Phase 7 / v2.3+)
========================================================================

Implements the planetary relational distributed causal architecture unifying:
- Dual-Layer Causality:
  * Hard Layer: Exact integer orbital slots (`ORBIT_MOD = 3600`), integer energy
    & enhancement levels, and SHA-256 Provenance hash chains (zero float drift).
  * Soft Layer: Continuous complex waveforms & 3-tier Earth/Water/Sky spectral
    rotors absorbing network phase jitter via Kuramoto phase-locking.
- Admissible Free-Will Causal Cone:
  User free-will choices select from causally reachable futures; non-causal
  state mutations (cheats) fall outside the cone and are excluded as non-existent.
- VirtualRelationalMemory:
  O(1) perspective axis switching across `vision`, `language`, `code`, and
  `mechanics` with zero recomputation.
- 4-Node Planetary Mutual Spherical Mirror Quorum (3f+1 Byzantine consensus):
  Peer cross-validation, automatic T_anchor pruning rollback, signed HMAC-SHA256
  governance patches, and dropped-packet self-healing via cone + mirror curvature.
- Multi-Frequency Ecosystem Re-Causalization & Parallel Universe Projection.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.topology.causal_scale_rotor import (
    CausalScaleRotorField,
    RotorNode,
    ScaleRotor,
)

if TYPE_CHECKING:
    from core.physics.causal_field import CausalField


ORBIT_MOD: int = 3600  # 0.1-degree discrete planetary angular lattice


@dataclass(frozen=True)
class ProvenanceRecord:
    """Immutable cryptographic provenance link for every pure state transition."""

    tick: int
    prev_hash: str
    action: str
    param: int
    steps: int
    rules_version: int
    orbit_slot: int
    energy: int
    enhancement_level: int
    state_hash: str
    is_fast_forward: bool = False
    witness_tag: str = ""


@dataclass
class HardCausalState:
    """Exact deterministic integer state underlying distributed consensus."""

    tick: int = 0
    orbit_slot: int = 0
    energy: int = 0
    enhancement_level: int = 0
    rules_version: int = 1
    state_hash: str = ""

    def copy(self) -> "HardCausalState":
        return HardCausalState(
            tick=self.tick,
            orbit_slot=self.orbit_slot,
            energy=self.energy,
            enhancement_level=self.enhancement_level,
            rules_version=self.rules_version,
            state_hash=self.state_hash,
        )


@dataclass
class SoftSpectralState:
    """Continuous complex waveform state (Earth, Water, Sky) absorbing jitter."""

    complex_wave: complex = 1.0 + 0.0j
    phase_error_rad: float = 0.0
    layer_waves: Tuple[complex, complex, complex] = (1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j)

    def copy(self) -> "SoftSpectralState":
        return SoftSpectralState(
            complex_wave=complex(self.complex_wave),
            phase_error_rad=float(self.phase_error_rad),
            layer_waves=(
                complex(self.layer_waves[0]),
                complex(self.layer_waves[1]),
                complex(self.layer_waves[2]),
            ),
        )


class VirtualRelationalMemory:
    """
    Virtual Relational Memory sitting above physical linear storage.
    Maintains modality-preserving perspective axis bindings (`vision`,
    `language`, `code`, `mechanics`) with O(1) lookup and zero recomputation.
    """

    VALID_AXES = ("vision", "language", "code", "mechanics")

    def __init__(self) -> None:
        self._backing_store: Dict[str, Dict[str, Any]] = {}
        self.recompute_count: int = 0
        self.axis_switch_count: int = 0

    def bind_entity(
        self,
        entity_id: str,
        hard_state: HardCausalState,
        soft_state: SoftSpectralState,
        provenance: Sequence[ProvenanceRecord],
    ) -> None:
        angle_deg = (hard_state.orbit_slot / float(ORBIT_MOD)) * 360.0
        phase_rad = cmath.phase(soft_state.complex_wave)
        tip_hash = provenance[-1].state_hash if provenance else hard_state.state_hash
        last_action = provenance[-1].action if provenance else "INIT"

        views = {
            "vision": {
                "axis": "vision",
                "orbital_angle_deg": round(angle_deg, 2),
                "spectral_luminance": round(abs(soft_state.layer_waves[2]), 4),
                "refraction_phase_rad": round(phase_rad, 4),
            },
            "language": {
                "axis": "language",
                "causal_verb": last_action,
                "narrative_summary": (
                    f"Entity '{entity_id}' stands at {angle_deg:.1f}deg after {last_action} "
                    f"(energy={hard_state.energy}, tier={hard_state.enhancement_level})."
                ),
            },
            "code": {
                "axis": "code",
                "tick": hard_state.tick,
                "rules_version": hard_state.rules_version,
                "provenance_length": len(provenance),
                "tip_hash": tip_hash,
            },
            "mechanics": {
                "axis": "mechanics",
                "orbit_slot": hard_state.orbit_slot,
                "energy": hard_state.energy,
                "enhancement_level": hard_state.enhancement_level,
                "earth_water_sky_phases": [round(cmath.phase(w), 4) for w in soft_state.layer_waves],
            },
        }
        self._backing_store[entity_id] = {
            "hard": hard_state.copy(),
            "soft": soft_state.copy(),
            "provenance_len": len(provenance),
            "views": views,
        }

    def switch_axis(self, entity_id: str, axis: str) -> Dict[str, Any]:
        if axis not in self.VALID_AXES:
            raise ValueError(f"Unknown perspective axis '{axis}', expected one of {self.VALID_AXES}")
        if entity_id not in self._backing_store:
            raise KeyError(f"Entity '{entity_id}' not bound in VirtualRelationalMemory")
        self.axis_switch_count += 1
        return dict(self._backing_store[entity_id]["views"][axis])


class CausalSkeleton:
    """
    Shared immutable invariant causal rules & SHA-256 provenance verifier.
    Updates only through authenticated HMAC-SHA256 signed governance patches.
    """

    def __init__(
        self,
        rules_version: int = 1,
        max_move_step: int = 60,
        max_accumulate_step: int = 25,
        enhance_cost: int = 20,
        phase_jitter_tolerance: float = 0.45,
        anchor_interval: int = 10,
    ) -> None:
        self.rules_version = int(rules_version)
        self.max_move_step = int(max_move_step)
        self.max_accumulate_step = int(max_accumulate_step)
        self.enhance_cost = int(enhance_cost)
        self.phase_jitter_tolerance = float(phase_jitter_tolerance)
        self.anchor_interval = max(1, int(anchor_interval))

    @staticmethod
    def compute_state_hash(
        prev_hash: str,
        tick: int,
        action: str,
        param: int,
        steps: int,
        rules_version: int,
        orbit_slot: int,
        energy: int,
        enhancement_level: int,
    ) -> str:
        canonical = (
            f"{prev_hash}|{tick}|{action}|{param}|{steps}|"
            f"{rules_version}|{orbit_slot}|{energy}|{enhancement_level}"
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]

    def genesis_state(self) -> HardCausalState:
        h0 = self.compute_state_hash(
            "GENESIS", 0, "INIT", 0, 0, self.rules_version, 0, 0, 0
        )
        return HardCausalState(
            tick=0,
            orbit_slot=0,
            energy=0,
            enhancement_level=0,
            rules_version=self.rules_version,
            state_hash=h0,
        )

    def admissible_free_will_cone(self, state: HardCausalState) -> Dict[str, range]:
        """Returns the exact admissible free-will action ranges reachable in 1 tick."""
        max_enh = state.energy // max(1, self.enhance_cost)
        return {
            "MOVE": range(-self.max_move_step, self.max_move_step + 1),
            "ACCUMULATE": range(1, self.max_accumulate_step + 1),
            "ENHANCE": range(1, max_enh + 1) if max_enh >= 1 else range(0, 0),
        }

    def apply_pure_transition(
        self,
        hard: HardCausalState,
        soft: SoftSpectralState,
        action: str,
        param: int,
        witness_tag: str = "",
    ) -> Tuple[HardCausalState, SoftSpectralState, ProvenanceRecord]:
        """Pure causal transition within the Admissible Free-Will Cone."""
        p = int(param)
        cone = self.admissible_free_will_cone(hard)
        if action not in cone:
            raise ValueError(f"NON_CAUSAL_TRANSITION: unknown action '{action}'")
        if p not in cone[action]:
            raise ValueError(
                f"NON_CAUSAL_TRANSITION: {action}({p}) lies outside admissible cone "
                f"(energy={hard.energy}, max_move={self.max_move_step}, "
                f"max_acc={self.max_accumulate_step}, enhance_cost={self.enhance_cost})"
            )

        new_orbit = hard.orbit_slot
        new_energy = hard.energy
        new_enh = hard.enhancement_level

        if action == "MOVE":
            new_orbit = (hard.orbit_slot + p) % ORBIT_MOD
            d_theta = (2.0 * math.pi * p) / float(ORBIT_MOD)
        elif action == "ACCUMULATE":
            new_energy = hard.energy + p
            d_theta = (2.0 * math.pi * p) / 360.0
        else:  # ENHANCE
            new_energy = hard.energy - (p * self.enhance_cost)
            new_enh = hard.enhancement_level + p
            d_theta = 2.0 * math.pi * (0.125 * p)

        new_tick = hard.tick + 1
        prev_h = hard.state_hash or "GENESIS"
        new_h = self.compute_state_hash(
            prev_h,
            new_tick,
            action,
            p,
            1,
            self.rules_version,
            new_orbit,
            new_energy,
            new_enh,
        )
        next_hard = HardCausalState(
            tick=new_tick,
            orbit_slot=new_orbit,
            energy=new_energy,
            enhancement_level=new_enh,
            rules_version=self.rules_version,
            state_hash=new_h,
        )
        next_soft = SoftSpectralState(
            complex_wave=soft.complex_wave * cmath.exp(1j * d_theta),
            phase_error_rad=soft.phase_error_rad,
            layer_waves=(
                soft.layer_waves[0] * cmath.exp(1j * 0.25 * d_theta),
                soft.layer_waves[1] * cmath.exp(1j * 1.00 * d_theta),
                soft.layer_waves[2] * cmath.exp(1j * 3.00 * d_theta),
            ),
        )
        rec = ProvenanceRecord(
            tick=new_tick,
            prev_hash=prev_h,
            action=action,
            param=p,
            steps=1,
            rules_version=self.rules_version,
            orbit_slot=new_orbit,
            energy=new_energy,
            enhancement_level=new_enh,
            state_hash=new_h,
            is_fast_forward=False,
            witness_tag=witness_tag,
        )
        return next_hard, next_soft, rec

    def fast_forward(
        self,
        hard: HardCausalState,
        soft: SoftSpectralState,
        steps: int,
        action: str,
        param: int,
        anomaly_check: Optional[Callable[[HardCausalState, int, str, int], bool]] = None,
    ) -> Tuple[HardCausalState, SoftSpectralState, ProvenanceRecord]:
        """
        Flow-direct O(1) transition along a steady validated orbit (`MOVE` or
        `ACCUMULATE`) preserving exact Provenance traceability.
        """
        n_steps = int(steps)
        p = int(param)
        if n_steps < 1:
            raise ValueError("NON_CAUSAL_ANOMALY_BRANCH: fast_forward steps must be >= 1")
        if action not in ("MOVE", "ACCUMULATE"):
            raise ValueError(
                f"NON_CAUSAL_ANOMALY_BRANCH: fast_forward supports only MOVE/ACCUMULATE, got {action}"
            )
        cone = self.admissible_free_will_cone(hard)
        if p not in cone[action]:
            raise ValueError(
                f"NON_CAUSAL_ANOMALY_BRANCH: {action}({p}) exceeds per-tick admissible cone"
            )
        if anomaly_check is not None and not anomaly_check(hard, n_steps, action, p):
            raise ValueError("NON_CAUSAL_ANOMALY_BRANCH: anomaly detector rejected branch")

        new_orbit = hard.orbit_slot
        new_energy = hard.energy
        if action == "MOVE":
            new_orbit = (hard.orbit_slot + n_steps * p) % ORBIT_MOD
            d_theta = (2.0 * math.pi * n_steps * p) / float(ORBIT_MOD)
        else:
            new_energy = hard.energy + n_steps * p
            d_theta = (2.0 * math.pi * n_steps * p) / 360.0

        new_tick = hard.tick + n_steps
        prev_h = hard.state_hash or "GENESIS"
        new_h = self.compute_state_hash(
            prev_h,
            new_tick,
            action,
            p,
            n_steps,
            self.rules_version,
            new_orbit,
            new_energy,
            hard.enhancement_level,
        )
        next_hard = HardCausalState(
            tick=new_tick,
            orbit_slot=new_orbit,
            energy=new_energy,
            enhancement_level=hard.enhancement_level,
            rules_version=self.rules_version,
            state_hash=new_h,
        )
        next_soft = SoftSpectralState(
            complex_wave=soft.complex_wave * cmath.exp(1j * d_theta),
            phase_error_rad=soft.phase_error_rad,
            layer_waves=(
                soft.layer_waves[0] * cmath.exp(1j * 0.25 * d_theta),
                soft.layer_waves[1] * cmath.exp(1j * 1.00 * d_theta),
                soft.layer_waves[2] * cmath.exp(1j * 3.00 * d_theta),
            ),
        )
        rec = ProvenanceRecord(
            tick=new_tick,
            prev_hash=prev_h,
            action=action,
            param=p,
            steps=n_steps,
            rules_version=self.rules_version,
            orbit_slot=new_orbit,
            energy=new_energy,
            enhancement_level=hard.enhancement_level,
            state_hash=new_h,
            is_fast_forward=True,
            witness_tag="fast_forward",
        )
        return next_hard, next_soft, rec

    def verify_provenance_chain(
        self,
        genesis_state: HardCausalState,
        chain: Sequence[ProvenanceRecord],
    ) -> Tuple[bool, Optional[int], str]:
        """Independently replays and cryptographically verifies every link in `chain`."""
        curr_hard = genesis_state.copy()
        curr_soft = SoftSpectralState()

        for idx, rec in enumerate(chain):
            if rec.prev_hash != curr_hard.state_hash:
                return False, idx, f"prev_hash mismatch at index {idx}"
            try:
                if rec.is_fast_forward:
                    next_h, next_s, exp_rec = self.fast_forward(
                        curr_hard, curr_soft, rec.steps, rec.action, rec.param
                    )
                else:
                    if rec.steps != 1:
                        return False, idx, f"non-fast-forward steps={rec.steps} != 1"
                    next_h, next_s, exp_rec = self.apply_pure_transition(
                        curr_hard, curr_soft, rec.action, rec.param
                    )
            except ValueError as exc:
                return False, idx, str(exc)

            # Compare against rules_version captured in record if patched or current
            exp_hash = self.compute_state_hash(
                rec.prev_hash,
                rec.tick,
                rec.action,
                rec.param,
                rec.steps,
                rec.rules_version,
                next_h.orbit_slot,
                next_h.energy,
                next_h.enhancement_level,
            )
            if (
                rec.tick != next_h.tick
                or rec.orbit_slot != next_h.orbit_slot
                or rec.energy != next_h.energy
                or rec.enhancement_level != next_h.enhancement_level
                or rec.state_hash != exp_hash
            ):
                return False, idx, f"NON_CAUSAL_FRAUD: state or hash mismatch at tick {rec.tick}"

            curr_hard = HardCausalState(
                tick=rec.tick,
                orbit_slot=rec.orbit_slot,
                energy=rec.energy,
                enhancement_level=rec.enhancement_level,
                rules_version=rec.rules_version,
                state_hash=rec.state_hash,
            )
            curr_soft = next_s

        return True, None, "PROVENANCE_VERIFIED"

    def sign_patch(self, patch_params: Dict[str, Any], authority_secret: str) -> str:
        payload = json.dumps(
            {"rules_version": self.rules_version, "patch": patch_params},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hmac.new(authority_secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()

    def apply_patch(
        self,
        patch_params: Dict[str, Any],
        signature_hex: str,
        authority_secret: str,
    ) -> Tuple[bool, str]:
        """Applies governance rule updates ONLY when HMAC-SHA256 signature verifies."""
        expected = self.sign_patch(patch_params, authority_secret)
        if not hmac.compare_digest(expected, str(signature_hex)):
            return False, "INVALID_SIGNATURE_REJECTED"

        for key in (
            "max_move_step",
            "max_accumulate_step",
            "enhance_cost",
            "anchor_interval",
        ):
            if key in patch_params:
                setattr(self, key, int(patch_params[key]))
        if "phase_jitter_tolerance" in patch_params:
            self.phase_jitter_tolerance = float(patch_params["phase_jitter_tolerance"])

        self.rules_version += 1
        return True, f"PATCH_APPLIED_v{self.rules_version}"


def fold_to_core(
    hard: HardCausalState,
    soft: SoftSpectralState,
    provenance: Sequence[ProvenanceRecord],
) -> Dict[str, Any]:
    """
    Folds multi-layer Earth/Water/Sky waves inward onto the invariant Core (0/1)
    while retaining the full Provenance lineage.
    """
    core_vector = sum(soft.layer_waves) / 3.0
    return {
        "core_vector": core_vector,
        "core_axis_phase": cmath.phase(core_vector),
        "hard_state": hard.copy(),
        "soft_state": soft.copy(),
        "provenance_chain": list(provenance),
        "lineage_length": len(provenance),
        "tip_hash": provenance[-1].state_hash if provenance else hard.state_hash,
    }


def unfold_to_boundary(
    core_packet: Dict[str, Any],
    vmem: VirtualRelationalMemory,
    entity_id: str,
    skeleton: CausalSkeleton,
    genesis_state: HardCausalState,
) -> Dict[str, Any]:
    """
    Unfolds a Core packet outward across all 4 boundary perspective axes while
    cryptographically verifying 100% Provenance preservation from genesis.
    """
    chain: List[ProvenanceRecord] = core_packet["provenance_chain"]
    ok, bad_idx, msg = skeleton.verify_provenance_chain(genesis_state, chain)
    if not ok:
        raise ValueError(f"Provenance corrupted during unfold at {bad_idx}: {msg}")

    hard: HardCausalState = core_packet["hard_state"]
    soft: SoftSpectralState = core_packet["soft_state"]
    vmem.bind_entity(entity_id, hard, soft, chain)
    axes = {ax: vmem.switch_axis(entity_id, ax) for ax in VirtualRelationalMemory.VALID_AXES}
    return {
        "entity_id": entity_id,
        "provenance_intact": ok,
        "verification_msg": msg,
        "lineage_length": len(chain),
        "tip_hash": core_packet["tip_hash"],
        "axes": axes,
    }


class PlanetaryCausalNode:
    """Autonomous local peer node executing pure transitions & maintaining checkpoints."""

    def __init__(self, node_id: str, skeleton: CausalSkeleton) -> None:
        self.node_id = node_id
        self.skeleton = skeleton
        self.genesis_state = skeleton.genesis_state()
        self.hard_state = self.genesis_state.copy()
        self.soft_state = SoftSpectralState()
        self.provenance: List[ProvenanceRecord] = []
        self.valid_checkpoint: Tuple[HardCausalState, SoftSpectralState, int] = (
            self.genesis_state.copy(),
            self.soft_state.copy(),
            0,
        )
        self.vmem = VirtualRelationalMemory()
        self.rollback_count: int = 0

    def execute_action(self, action: str, param: int) -> ProvenanceRecord:
        nh, ns, rec = self.skeleton.apply_pure_transition(
            self.hard_state, self.soft_state, action, param, witness_tag=self.node_id
        )
        self.hard_state = nh
        self.soft_state = ns
        self.provenance.append(rec)
        return rec

    def execute_fast_forward(
        self,
        steps: int,
        action: str,
        param: int,
        anomaly_check: Optional[Callable[[HardCausalState, int, str, int], bool]] = None,
    ) -> ProvenanceRecord:
        nh, ns, rec = self.skeleton.fast_forward(
            self.hard_state,
            self.soft_state,
            steps,
            action,
            param,
            anomaly_check=anomaly_check,
        )
        self.hard_state = nh
        self.soft_state = ns
        self.provenance.append(rec)
        return rec

    def mark_checkpoint(self) -> None:
        self.valid_checkpoint = (
            self.hard_state.copy(),
            self.soft_state.copy(),
            len(self.provenance),
        )

    def inject_cheat(self, cheat_type: str = "enhance_without_accumulate") -> None:
        """Simulates non-causal local memory tampering outside the admissible cone."""
        if cheat_type == "enhance_without_accumulate":
            self.hard_state.enhancement_level += 5
        elif cheat_type == "energy_overflow":
            self.hard_state.energy += 500
        else:
            self.hard_state.orbit_slot = (self.hard_state.orbit_slot + 999) % ORBIT_MOD

    def inject_phase_jitter(self, delta_rad: float) -> None:
        """Injects soft complex phase jitter (simulating network lag) without breaking hard chain."""
        self.soft_state.complex_wave *= cmath.exp(1j * float(delta_rad))
        self.soft_state.phase_error_rad = abs(float(delta_rad))

    def rollback_to_checkpoint(self) -> None:
        """Prunes corrupted branch and restores valid T_anchor checkpoint."""
        ck_hard, ck_soft, ck_len = self.valid_checkpoint
        self.hard_state = ck_hard.copy()
        self.soft_state = ck_soft.copy()
        self.provenance = self.provenance[:ck_len]
        self.rollback_count += 1


class PlanetaryQuorumNetwork:
    """
    4-Node Mutual Spherical Mirror Quorum (3f+1 Byzantine fault tolerance)
    operating without any central authority server.
    """

    def __init__(self, skeleton: Optional[CausalSkeleton] = None) -> None:
        self.skeleton = skeleton if skeleton is not None else CausalSkeleton()
        self.nodes: Dict[str, PlanetaryCausalNode] = {
            nid: PlanetaryCausalNode(nid, self.skeleton)
            for nid in ("Node_A", "Node_B", "Node_C", "Node_D")
        }

    def broadcast_action(self, action: str, param: int) -> Dict[str, ProvenanceRecord]:
        return {nid: node.execute_action(action, param) for nid, node in self.nodes.items()}

    def broadcast_fast_forward(self, steps: int, action: str, param: int) -> Dict[str, ProvenanceRecord]:
        return {
            nid: node.execute_fast_forward(steps, action, param)
            for nid, node in self.nodes.items()
        }

    def sync_complex_phase_lock(self) -> Dict[str, Any]:
        """
        Kuramoto-like mutual spherical mirror phase-locking across peers whose
        phase deviation stays within `phase_jitter_tolerance`.
        """
        phases = [cmath.phase(n.soft_state.complex_wave) for n in self.nodes.values()]
        mean_vec = sum(cmath.exp(1j * p) for p in phases) / float(len(phases))
        target_phase = cmath.phase(mean_vec)

        absorbed_nodes: List[str] = []
        for nid, node in self.nodes.items():
            cur_p = cmath.phase(node.soft_state.complex_wave)
            diff = (target_phase - cur_p + math.pi) % (2.0 * math.pi) - math.pi
            if abs(diff) > 1e-6 and abs(diff) <= self.skeleton.phase_jitter_tolerance:
                node.soft_state.complex_wave *= cmath.exp(1j * diff)
                node.soft_state.phase_error_rad = 0.0
                absorbed_nodes.append(nid)

        return {
            "target_phase_rad": round(target_phase, 4),
            "jitter_absorbed_nodes": absorbed_nodes,
        }

    def validate_quorum(self) -> Dict[str, Any]:
        """
        Cross-validates all peers' Provenance chains and integer state integrity
        via Byzantine quorum (>= 3 of 4 nodes matching). Automatically prunes &
        rolls back any corrupted peer and re-syncs from honest mirrors.
        """
        valid_hashes: Dict[str, str] = {}
        rejected: Dict[str, str] = {}

        for nid, node in self.nodes.items():
            # Check hard state matches provenance tip (catches direct memory tampering!)
            tip_rec = node.provenance[-1] if node.provenance else None
            expected_tip = tip_rec.state_hash if tip_rec else node.genesis_state.state_hash
            expected_tick = tip_rec.tick if tip_rec else 0
            expected_orbit = tip_rec.orbit_slot if tip_rec else 0
            expected_energy = tip_rec.energy if tip_rec else 0
            expected_enh = tip_rec.enhancement_level if tip_rec else 0

            if (
                node.hard_state.state_hash != expected_tip
                or node.hard_state.tick != expected_tick
                or node.hard_state.orbit_slot != expected_orbit
                or node.hard_state.energy != expected_energy
                or node.hard_state.enhancement_level != expected_enh
            ):
                rejected[nid] = "NON_CAUSAL_FRAUD: hard_state diverges from provenance tip"
                continue

            ok, bad_idx, msg = self.skeleton.verify_provenance_chain(
                node.genesis_state, node.provenance
            )
            if not ok:
                rejected[nid] = f"NON_CAUSAL_FRAUD at {bad_idx}: {msg}"
                continue

            valid_hashes[nid] = node.hard_state.state_hash

        # Majority quorum consensus (>= 3 matching nodes)
        counts: Dict[str, int] = {}
        for h in valid_hashes.values():
            counts[h] = counts.get(h, 0) + 1

        consensus_hash: Optional[str] = None
        if counts:
            top_h, top_cnt = max(counts.items(), key=lambda kv: kv[1])
            if top_cnt >= 3:
                consensus_hash = top_h

        if consensus_hash is not None:
            for nid, h in list(valid_hashes.items()):
                if h != consensus_hash:
                    rejected[nid] = "NON_CAUSAL_DESYNC: state hash outside 3/4 quorum"

        # Rollback rejected peers to T_anchor and heal from honest mirror if quorum holds
        honest_donor = next(
            (self.nodes[nid] for nid in valid_hashes if nid not in rejected),
            None,
        )
        for nid in rejected:
            bad_node = self.nodes[nid]
            bad_node.rollback_to_checkpoint()
            if honest_donor is not None and len(honest_donor.provenance) > len(bad_node.provenance):
                for rec in honest_donor.provenance[len(bad_node.provenance) :]:
                    if rec.is_fast_forward:
                        bad_node.execute_fast_forward(rec.steps, rec.action, rec.param)
                    else:
                        bad_node.execute_action(rec.action, rec.param)
                bad_node.soft_state = honest_donor.soft_state.copy()

        # Advance T_anchor checkpoints on full verified consensus
        if not rejected and consensus_hash is not None:
            for node in self.nodes.values():
                node.mark_checkpoint()

        final_hashes = {nid: n.hard_state.state_hash for nid, n in self.nodes.items()}
        return {
            "consensus_reached": len(set(final_hashes.values())) == 1,
            "consensus_hash": consensus_hash,
            "rejected_nodes": rejected,
            "final_hashes": final_hashes,
        }

    def heal_dropped_packet(
        self,
        damaged_node_id: str,
        peer_anchor_record: ProvenanceRecord,
    ) -> Dict[str, Any]:
        """
        Recovers a missing single-tick transition packet on `damaged_node_id`
        using the Admissible Free-Will Cone + mutual spherical mirror anchor hash.
        Rejects impossible cheat states disguised as packet loss.
        """
        node = self.nodes[damaged_node_id]
        cone = self.skeleton.admissible_free_will_cone(node.hard_state)

        for action, param_range in cone.items():
            for p in param_range:
                if action in ("MOVE", "ACCUMULATE") or p >= 1:
                    nh, ns, rec = self.skeleton.apply_pure_transition(
                        node.hard_state,
                        node.soft_state,
                        action,
                        p,
                        witness_tag="mirror_healed",
                    )
                    if rec.state_hash == peer_anchor_record.state_hash:
                        node.hard_state = nh
                        node.soft_state = ns
                        node.provenance.append(rec)
                        return {
                            "healed": True,
                            "fraud_detected": False,
                            "recovered_action": action,
                            "recovered_param": p,
                            "state_hash": rec.state_hash,
                        }

        # No admissible causal path can produce peer_anchor_record -> reject as fraud
        node.rollback_to_checkpoint()
        return {
            "healed": False,
            "fraud_detected": True,
            "reason": "NO_ADMISSIBLE_CAUSAL_PATH_IN_CONE",
        }


class PlanetaryEcosystemLoop:
    """
    Couples a Low-Frequency Macro Environment Rotor (Earth/Water/Sky) with a
    High-Frequency NPC/Entity Causal Rotor, supporting Re-Causalization and
    zero-mutation Parallel Universe counterfactual simulations.
    """

    def __init__(self, skeleton: Optional[CausalSkeleton] = None) -> None:
        self.skeleton = skeleton if skeleton is not None else CausalSkeleton()
        self.env_rotors = CausalScaleRotorField(
            [
                RotorNode("earth_crust", ScaleRotor(axis=(0.0, 0.0, 1.0), angular_rate=0.05, radius=1.0)),
                RotorNode("water_tide", ScaleRotor(axis=(0.0, 0.2, 1.0), angular_rate=0.22, radius=0.5), "earth_crust"),
                RotorNode("sky_climate", ScaleRotor(axis=(0.2, 0.0, 1.0), angular_rate=0.65, radius=0.25), "water_tide"),
            ]
        )
        self.npc_hard = self.skeleton.genesis_state()
        self.npc_soft = SoftSpectralState()
        self.npc_provenance: List[ProvenanceRecord] = []
        self.env_tick: int = 0
        self.interaction_memories: List[Dict[str, Any]] = []
        self.future_orbit_preview: List[int] = []

    def _current_env_phase(self) -> float:
        t = float(self.env_tick)
        slices = self.env_rotors.slice_at(t)
        sky = slices["sky_climate"]
        return float(math.atan2(sky[1], sky[0]))

    def step_ecosystem(
        self,
        dt_ticks: int = 1,
        user_free_will_stimulus: Optional[Tuple[str, int]] = None,
    ) -> Dict[str, Any]:
        env_before = self._current_env_phase()
        self.env_tick += max(1, int(dt_ticks))
        env_after = self._current_env_phase()

        if user_free_will_stimulus is not None:
            action, param = user_free_will_stimulus
            nh, ns, rec = self.skeleton.apply_pure_transition(
                self.npc_hard, self.npc_soft, action, param, witness_tag="user_stimulus"
            )
            self.npc_hard = nh
            self.npc_soft = ns
            self.npc_provenance.append(rec)
            self.interaction_memories.append(
                {
                    "tick": self.npc_hard.tick,
                    "action": action,
                    "param": param,
                    "env_phase_at_event": env_after,
                    "reinterpreted_resonance": 1.0,
                }
            )
        else:
            # Natural low-frequency environmental coupling step
            step_move = int(round(10.0 * math.cos(env_after)))
            nh, ns, rec = self.skeleton.apply_pure_transition(
                self.npc_hard, self.npc_soft, "MOVE", step_move, witness_tag="env_coupling"
            )
            self.npc_hard = nh
            self.npc_soft = ns
            self.npc_provenance.append(rec)

        # Re-Causalization: reinterpret past interactions under shifted macro environment
        for mem in self.interaction_memories:
            delta_phi = env_after - float(mem["env_phase_at_event"])
            mem["reinterpreted_resonance"] = round(float(math.cos(delta_phi)), 4)

        # Predict 4-tick future orbital slots
        preview: List[int] = []
        sim_h = self.npc_hard.copy()
        sim_s = self.npc_soft.copy()
        for k in range(1, 5):
            fut_env = math.atan2(
                self.env_rotors.slice_at(float(self.env_tick + k))["sky_climate"][1],
                self.env_rotors.slice_at(float(self.env_tick + k))["sky_climate"][0],
            )
            fut_move = int(round(10.0 * math.cos(fut_env)))
            sim_h, sim_s, _ = self.skeleton.apply_pure_transition(sim_h, sim_s, "MOVE", fut_move)
            preview.append(sim_h.orbit_slot)
        self.future_orbit_preview = preview

        return {
            "env_tick": self.env_tick,
            "env_phase_shift": round(env_after - env_before, 4),
            "npc_orbit_slot": self.npc_hard.orbit_slot,
            "npc_energy": self.npc_hard.energy,
            "reinterpreted_memories": list(self.interaction_memories),
            "future_orbit_preview": list(preview),
        }

    def project_parallel_universe(
        self,
        counterfactual_stimulus: Tuple[str, int],
        horizon_ticks: int = 4,
    ) -> Dict[str, Any]:
        """
        Clones verified state into an isolated counterfactual universe branch to
        simulate `What-If` future trajectories with zero primary state mutation.
        """
        sim_h = self.npc_hard.copy()
        sim_s = self.npc_soft.copy()
        act, prm = counterfactual_stimulus
        sim_h, sim_s, _ = self.skeleton.apply_pure_transition(
            sim_h, sim_s, act, prm, witness_tag="parallel_whatif"
        )
        parallel_orbits = [sim_h.orbit_slot]
        for k in range(1, int(horizon_ticks)):
            fut_env = math.atan2(
                self.env_rotors.slice_at(float(self.env_tick + k))["sky_climate"][1],
                self.env_rotors.slice_at(float(self.env_tick + k))["sky_climate"][0],
            )
            fut_move = int(round(10.0 * math.cos(fut_env)))
            sim_h, sim_s, _ = self.skeleton.apply_pure_transition(sim_h, sim_s, "MOVE", fut_move)
            parallel_orbits.append(sim_h.orbit_slot)

        primary_orbit = self.future_orbit_preview[-1] if self.future_orbit_preview else self.npc_hard.orbit_slot
        divergence = abs(parallel_orbits[-1] - primary_orbit)
        return {
            "primary_orbit_tail": primary_orbit,
            "parallel_orbits": parallel_orbits,
            "orbital_divergence": divergence,
        }


def sync_to_causal_field(
    network: PlanetaryQuorumNetwork,
    ecosystem: Optional[PlanetaryEcosystemLoop] = None,
    causal_field: Optional["CausalField"] = None,
    prefix: str = "planet",
) -> Dict[str, Any]:
    """
    Projects the 4 planetary quorum mirror nodes and Earth/Water/Sky + NPC
    rotors into `CausalField` with consistent 4D voxel tensors.
    """
    from core.physics.causal_field import CausalField, EngramAttractor, InformationVoxel

    field = causal_field if causal_field is not None else CausalField(dimensions=3)
    node_vids: List[str] = []

    for idx, (nid, node) in enumerate(network.nodes.items()):
        ang = (2.0 * math.pi * node.hard_state.orbit_slot) / float(ORBIT_MOD)
        pos = np.array([math.cos(ang), math.sin(ang), float(idx) * 0.2], dtype=np.float32)
        vid = f"{prefix}_{nid}"
        field.add_voxel(
            InformationVoxel(
                id=vid,
                content={"node": nid, "hash": node.hard_state.state_hash},
                tensor=np.array(
                    [
                        float(node.hard_state.orbit_slot) / float(ORBIT_MOD),
                        float(node.hard_state.energy),
                        float(node.hard_state.enhancement_level),
                        float(cmath.phase(node.soft_state.complex_wave)),
                    ],
                    dtype=np.float32,
                ),
                mass=max(1.0, 1.0 + 0.05 * node.hard_state.energy),
                position=pos[: field.dimensions],
                phase_angle=ang % (2.0 * math.pi),
            )
        )
        field.register_engram(
            EngramAttractor(
                id=f"{prefix}_engram_{nid}",
                name=nid,
                position=pos[: field.dimensions],
                intensity=2.0,
                tier="meso",
                symbolic_intent=f"Mirror {nid} tick={node.hard_state.tick}",
            )
        )
        node_vids.append(vid)

    beams = 0
    for i in range(len(node_vids)):
        for j in range(i + 1, len(node_vids)):
            field.link_voxels(node_vids[i], node_vids[j], strength=0.90)
            beams += 1

    if ecosystem is not None:
        npc_vid = f"{prefix}_npc_rotor"
        ang_npc = (2.0 * math.pi * ecosystem.npc_hard.orbit_slot) / float(ORBIT_MOD)
        pos_npc = np.array([math.cos(ang_npc) * 0.5, math.sin(ang_npc) * 0.5, 1.0], dtype=np.float32)
        field.add_voxel(
            InformationVoxel(
                id=npc_vid,
                content={"npc_orbit": ecosystem.npc_hard.orbit_slot},
                tensor=np.array(
                    [
                        float(ecosystem.npc_hard.orbit_slot) / float(ORBIT_MOD),
                        float(ecosystem.npc_hard.energy),
                        float(ecosystem.env_tick),
                        float(cmath.phase(ecosystem.npc_soft.complex_wave)),
                    ],
                    dtype=np.float32,
                ),
                mass=1.5,
                position=pos_npc[: field.dimensions],
                phase_angle=ang_npc,
            )
        )
        field.link_voxels(node_vids[0], npc_vid, strength=0.80)
        beams += 1

    return {
        "causal_field": field,
        "node_voxels": node_vids,
        "beams_linked": beams,
    }

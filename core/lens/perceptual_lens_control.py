"""
Perceptual Lens Control (지각 제어 메커니즘).

관측 대상의 스케일이 달라져도 하부의 인과적 골격(Structural Invariant)을 손실 없이 보존한 채
관측 대역폭(C_lens)만을 동적으로 재정렬하는 인지 조절기 모듈.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum

from core.physics.causal_field import CausalField, InformationVoxel, ConnectivityBeam
from core.lens.cognitive_lens_engine import CognitiveLensEngine, ContextualDimension


class ScaleLevel(Enum):
    QUANTUM_MICRO = "quantum_micro"
    LOCAL_PARTICLE = "local_particle"
    DROPLET = "droplet"
    FLUID_STREAM = "fluid_stream"
    GALACTIC_MACRO = "galactic_macro"


@dataclass
class CausalInvariant:
    """
    구조 불변량 (I_c): 스케일 변환에도 손실되지 않는 범주론적 인과 골격.
    노드 간 인과 관계성(Morphisms), 위상적 연결성, 에너지/장력 보존 비율을 보존하여 가역적 역추적을 가능케 함.
    """
    invariant_id: str
    morphisms: Dict[Tuple[str, str], float]  # (source_id, target_id) -> normalized coupling/directionality
    topological_dimension: int
    energy_conservation_ratio: float
    causal_signature: np.ndarray  # Fixed structural shape signature
    meta_properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservationBandwidth:
    """
    관측 대역폭 (C_lens): 현재 렌즈가 수용 가능한 관측 스케일 범위 및 장력 한계치.
    """
    scale_min: float = 0.01
    scale_max: float = 100.0
    current_scale: float = 1.0
    max_friction_capacity: float = 10.0
    noise_filter_threshold: float = 0.05


@dataclass
class SealedAttractor:
    """
    SealedAttractor: 임계 장력(V_critical) 초과 시, 전체 인과망 파열을 막기 위해 충돌 노드 및 비동기 파열점을 격리하는 상태 저장소.
    """
    attractor_id: str
    sealed_voxels: Dict[str, InformationVoxel]
    sealed_beams: List[ConnectivityBeam]
    critical_friction_level: float
    isolation_timestamp: float
    reason: str


@dataclass
class ReframingShift:
    """
    자율 리프레이밍 트리거 결과.
    """
    shift_type: str  # "MACRO_SHIFT" or "MICRO_SHIFT" or "STABLE"
    previous_scale: float
    new_scale: float
    accumulated_friction: float
    isolated_attractor_id: Optional[str] = None


class PerceptualLensControl:
    """
    지각 제어 메커니즘 (Perceptual Lens Control Engine).

    1. 동적 스케일 줌 (Scale-Adaptive Refocusing)
    2. 위상적 동형 보존 사상 (Isomorphic Projection Rules)
    3. 자율 렌즈 전환 임계 제어 (Reframing Triggers)
    """

    def __init__(
        self,
        causal_field: Optional[CausalField] = None,
        cognitive_engine: Optional[CognitiveLensEngine] = None,
        bandwidth: Optional[ObservationBandwidth] = None,
        critical_friction_threshold: float = 25.0
    ):
        self.field = causal_field if causal_field is not None else CausalField()
        self.cognitive_engine = cognitive_engine if cognitive_engine is not None else CognitiveLensEngine()
        self.bandwidth = bandwidth if bandwidth is not None else ObservationBandwidth()
        self.critical_friction_threshold = critical_friction_threshold

        # State tracking
        self.sealed_attractors: Dict[str, SealedAttractor] = {}
        self.invariants: Dict[str, CausalInvariant] = {}
        self.accumulated_friction: float = 0.0
        self.scale_history: List[float] = [self.bandwidth.current_scale]

    def compute_friction_tension(self) -> float:
        """
        인과적 마찰 장력 (V_t) 계산.
        Voxels 간 상충, ConnectivityBeam 장력, 장 위상 이탈 감지를 종합하여 계산.
        """
        friction = 0.0

        # 1. Connectivity Beam tension accumulation
        for beam in self.field.beams:
            if not beam.is_broken:
                friction += beam.current_tension
            else:
                friction += beam.break_threshold * 1.5  # Broken beams contribute elevated friction

        # 2. Voxel internal phase / velocity friction & noise
        for voxel in self.field.voxels.values():
            speed = float(np.linalg.norm(voxel.velocity))
            potential_mismatch = abs(voxel.potential)
            entropy = float(voxel.chromatic_vector[2])  # Yellow component
            friction += (speed * 0.2 + potential_mismatch * 0.5 + entropy * 1.0) * self.bandwidth.current_scale

        self.accumulated_friction = float(friction)
        return self.accumulated_friction

    def extract_structural_invariant(self, invariant_id: str = "primary_invariant") -> CausalInvariant:
        """
        위상적 동형 보존 사상 (Isomorphic Projection Rules):
        현재 field의 voxels 및 beams로부터 스케일 독립적 인과 구조 불변량 (I_c)을 추출.
        """
        morphisms: Dict[Tuple[str, str], float] = {}
        total_energy = 0.0

        for beam in self.field.beams:
            if not beam.is_broken:
                # Directional morphic coupling normalized by distance/strength
                v_src = self.field.voxels.get(beam.source_id)
                v_tgt = self.field.voxels.get(beam.target_id)
                if v_src and v_tgt:
                    dist = max(1e-5, float(np.linalg.norm(v_src.position - v_tgt.position)))
                    coupling = beam.strength / dist
                    morphisms[(beam.source_id, beam.target_id)] = float(coupling)

        for voxel in self.field.voxels.values():
            kinetic = 0.5 * voxel.mass * float(np.sum(voxel.velocity ** 2))
            potential = abs(voxel.potential)
            total_energy += kinetic + potential

        # Construct scale-invariant tensor signature (normalized positions/tensors)
        if self.field.voxels:
            tensors = [v.tensor.flatten() for v in self.field.voxels.values()]
            max_len = max(len(t) for t in tensors)
            padded = [np.pad(t, (0, max_len - len(t))) for t in tensors]
            avg_tensor = np.mean(padded, axis=0)
            norm_avg = np.linalg.norm(avg_tensor)
            signature = avg_tensor / norm_avg if norm_avg > 0 else avg_tensor
        else:
            signature = np.zeros(self.field.dimensions, dtype=np.float32)

        invariant = CausalInvariant(
            invariant_id=invariant_id,
            morphisms=morphisms,
            topological_dimension=self.field.dimensions,
            energy_conservation_ratio=float(total_energy / (len(self.field.voxels) + 1e-5)),
            causal_signature=signature,
            meta_properties={"voxel_count": len(self.field.voxels), "scale": self.bandwidth.current_scale}
        )
        self.invariants[invariant_id] = invariant
        return invariant

    def is_isomorphic(self, inv1: CausalInvariant, inv2: CausalInvariant, tolerance: float = 0.25) -> bool:
        """
        두 구조 불변량 (I_c) 간의 위상적 동형성(Isomorphism) 보존 여부 검증.
        스케일이 바뀌어도 노드 간의 관계성(Morphisms)과 시그니처 방향성이 보존되는지 점검.
        """
        # 1. Signature similarity (dot product of normalized structural signatures)
        s1 = inv1.causal_signature.flatten()
        s2 = inv2.causal_signature.flatten()
        min_len = min(len(s1), len(s2))
        if min_len > 0:
            s1_norm = np.linalg.norm(s1[:min_len])
            s2_norm = np.linalg.norm(s2[:min_len])
            if s1_norm > 0 and s2_norm > 0:
                sim = float(np.dot(s1[:min_len] / s1_norm, s2[:min_len] / s2_norm))
            else:
                sim = 1.0
        else:
            sim = 1.0

        # 2. Morphisms directional preservation
        # Check ratio of common active connections
        keys1 = set(inv1.morphisms.keys())
        keys2 = set(inv2.morphisms.keys())
        if keys1 and keys2:
            intersection = keys1.intersection(keys2)
            morphism_preservation = len(intersection) / max(len(keys1), len(keys2))
        else:
            morphism_preservation = 1.0

        isomorphic = (sim >= (1.0 - tolerance)) and (morphism_preservation >= (1.0 - tolerance))
        return isomorphic

    def refocus_scale(self, target_scale: float, preserve_invariants: bool = True) -> Dict[str, Any]:
        """
        동적 스케일 줌 (Scale-Adaptive Refocusing):
        초점거리 이동: 은하 <-> 유체 <-> 물방울 <-> 미시 입자.
        스케일 변경 시 하위 층위의 미시적 잡음은 필터링하고, 인과적 불변성(I_c) 및 마찰 장력만 상위/하위 맥락으로 전이.
        """
        old_scale = self.bandwidth.current_scale
        target_scale = float(np.clip(target_scale, self.bandwidth.scale_min, self.bandwidth.scale_max))
        scale_ratio = target_scale / (old_scale + 1e-9)

        if preserve_invariants and self.field.voxels:
            pre_invariant = self.extract_structural_invariant("pre_refocus")

        # Scale velocity, position, and filter noise
        for voxel in self.field.voxels.values():
            voxel.scale_level = target_scale
            # Scale coordinates according to magnification
            voxel.position = voxel.position * (1.0 / scale_ratio)
            # Filter micro-noise if zooming out (Macro Shift)
            if scale_ratio > 1.0:  # Macro scale up
                noise_mask = np.abs(voxel.velocity) < self.bandwidth.noise_filter_threshold
                voxel.velocity[noise_mask] = 0.0
                voxel.velocity *= (1.0 / scale_ratio)
            else:  # Micro scale down
                voxel.velocity *= (1.0 / scale_ratio)

        self.bandwidth.current_scale = target_scale
        self.scale_history.append(target_scale)

        isomorphism_preserved = True
        if preserve_invariants and self.field.voxels:
            post_invariant = self.extract_structural_invariant("post_refocus")
            isomorphism_preserved = self.is_isomorphic(pre_invariant, post_invariant)

        # Update Cognitive Lens Engine Curvature accordingly
        for dim in ContextualDimension:
            self.cognitive_engine.adjust_lens_curvature(dim, curvature=1.0 / target_scale)

        return {
            "previous_scale": old_scale,
            "new_scale": target_scale,
            "scale_ratio": scale_ratio,
            "isomorphism_preserved": isomorphism_preserved
        }

    def isolate_conflict_nodes(self, reason: str = "Critical friction breach") -> Optional[SealedAttractor]:
        """
        임계 장력(V_critical) 초과 시, 충돌 노드를 정밀 격리하는 SealedAttractor 생성 및 파열 방지.
        """
        if not self.field.voxels:
            return None

        # Identify highest friction voxels / broken beams
        high_friction_voxels: Dict[str, InformationVoxel] = {}
        high_friction_beams: List[ConnectivityBeam] = []

        # Select voxels with high potential or broken beams
        broken_voxel_ids: Set[str] = set()
        for beam in self.field.beams:
            if beam.is_broken or beam.current_tension > self.bandwidth.max_friction_capacity:
                high_friction_beams.append(beam)
                broken_voxel_ids.add(beam.source_id)
                broken_voxel_ids.add(beam.target_id)

        # If no specific broken beams, pick highest potential voxels
        if not broken_voxel_ids:
            sorted_voxels = sorted(self.field.voxels.items(), key=lambda kv: abs(kv[1].potential), reverse=True)
            for vid, v in sorted_voxels[:max(1, len(sorted_voxels) // 2)]:
                broken_voxel_ids.add(vid)

        # Extract and remove from active field into SealedAttractor
        attractor_id = f"sealed_{len(self.sealed_attractors) + 1}_{int(self.accumulated_friction)}"
        for vid in broken_voxel_ids:
            if vid in self.field.voxels:
                high_friction_voxels[vid] = self.field.voxels.pop(vid)

        # Remove corresponding beams
        self.field.beams = [
            b for b in self.field.beams
            if b.source_id not in broken_voxel_ids and b.target_id not in broken_voxel_ids
        ]

        sealed = SealedAttractor(
            attractor_id=attractor_id,
            sealed_voxels=high_friction_voxels,
            sealed_beams=high_friction_beams,
            critical_friction_level=self.accumulated_friction,
            isolation_timestamp=float(self.field.time_step_accumulator),
            reason=reason
        )
        self.sealed_attractors[attractor_id] = sealed
        return sealed

    def evaluate_and_reframe(self) -> ReframingShift:
        """
        자율 렌즈 전환 임계 제어 (Reframing Triggers):
        관측 대역폭 C_lens 수용 용량과 인과적 마찰 장력 V_t를 비교하여 자율 리프레이밍 수행.
        - V_t > V_critical: 충돌 노드 SealedAttractor 격리 및 Micro Shift.
        - V_t > C_lens: Macro Shift (거시적 줌 아웃으로 미시 마찰을 유체 흐름으로 재해석).
        - V_t < C_lens * 0.2: 필요시 Micro Shift (미시적 정밀 분석).
        """
        vt = self.compute_friction_tension()
        c_lens = self.bandwidth.max_friction_capacity
        v_crit = self.critical_friction_threshold
        old_scale = self.bandwidth.current_scale

        if vt > v_crit:
            # Critical Rupture -> Isolate into SealedAttractor & perform Micro Shift for isolation
            sealed = self.isolate_conflict_nodes(reason=f"Critical friction Vt ({vt:.2f}) > V_critical ({v_crit:.2f})")
            new_scale = max(self.bandwidth.scale_min, old_scale * 0.5)
            self.refocus_scale(new_scale)
            return ReframingShift(
                shift_type="MICRO_SHIFT_SEALED",
                previous_scale=old_scale,
                new_scale=new_scale,
                accumulated_friction=vt,
                isolated_attractor_id=sealed.attractor_id if sealed else None
            )

        elif vt > c_lens:
            # Capacity Overload -> Macro Shift (Zoom out to interpret noise as fluid flow)
            macro_factor = min(10.0, 1.0 + (vt / c_lens))
            new_scale = min(self.bandwidth.scale_max, old_scale * macro_factor)
            self.refocus_scale(new_scale)
            return ReframingShift(
                shift_type="MACRO_SHIFT",
                previous_scale=old_scale,
                new_scale=new_scale,
                accumulated_friction=vt
            )

        elif vt < c_lens * 0.1 and old_scale > 1.0:
            # Low friction -> Micro Shift (Zoom in for fine resolution)
            new_scale = max(1.0, old_scale * 0.5)
            self.refocus_scale(new_scale)
            return ReframingShift(
                shift_type="MICRO_SHIFT",
                previous_scale=old_scale,
                new_scale=new_scale,
                accumulated_friction=vt
            )

        return ReframingShift(
            shift_type="STABLE",
            previous_scale=old_scale,
            new_scale=old_scale,
            accumulated_friction=vt
        )

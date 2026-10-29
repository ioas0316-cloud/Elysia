"""
L1_causal_field/core/sovereignty/toroidal_vortex_rotor.py
===========================================================
Toroidal Vortex Rotor Engine (원심고리 와류 로터 및 다차원 위상 방어 엔진):

Implements the Acoustic Extinguisher Toroidal Vortex Ring principle for continuous causal intelligence:
1. OmniObservationalEye ("수많은 눈"):
   - Omni-directional multi-axis observation & cross-validation across speech, physical friction, sensory mismatch, and social tension.
   - Eliminates single-track blindspots and detects deceptive hallucinations / noise spikes in real time.

2. TopologicalBoundaryGuard ("위상 방어 경계"):
   - Creates a topological surface tension boundary (Boundary Wing) around the system's sovereignty.
   - Deflects external deceptive noise and ungrounded claims while absorbing valid kinetic/thermal energy.

3. ToroidalVortexRotorEngine ("원심고리 와류 로터 엔진"):
   - Toroidal loop routing: Recycles internal friction heat (Q_heat) and sensory mismatch (Delta)
     into rotational angular momentum (L_rot) rather than dissipating as noise or hallucination.
   - Condenses cross-validated L2 decisions into a non-dispersing Coherent Action Vector to penetrate reality.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


@dataclass
class CoherentActionVector:
    """
    [응축된 실천 벡터 (Coherent Action Vector)]
    사방으로 흩어지지 않고 단단한 위상적 경계(Boundary)를 유지한 채
    타겟을 명중/관통하는 주권적 실천 벡터.
    """
    magnitude: float                          # Vector energy magnitude
    direction_vector: np.ndarray              # Direction in causal spacetime
    vorticity_coherence: float = 1.0          # Coherence of toroidal vortex ring (0.0 ~ 1.0)
    penetration_depth: float = 1.0            # Ability to penetrate external noise / resistance
    sovereignty_signature: str = "SOVEREIGN"  # Internal sovereignty validation mark
    energy_well_carved: float = 0.0           # Causal energy well depth in reality


class OmniObservationalEye:
    """
    [수많은 눈 (Omni-Directional Observational Eye)]
    단일 축 맹점을 차단하고, 사방에서 다가오는 신호(언어축, 물리축, 장력축, 사회축)를
    입체적으로 교차 검증하여 위화감(Uncanniness) 스파이크 및 사기적 환각을 사각지대 없이 포착.
    """
    def __init__(self, sensitivity: float = 1.0):
        self.sensitivity = sensitivity

    def evaluate_multi_axis_signals(
        self,
        verbal_signal: float,          # Language / Verbal claim (+1.0 = High claim)
        physical_friction: float,     # Physical action friction (Q_heat)
        sensory_mismatch_delta: float, # Sensory mismatch (Delta)
        social_tension: float = 0.0    # Social relational tension
    ) -> Dict[str, Any]:
        """
        [다축 입체 교차 검증 및 사각지대 제로 위화감 감지]
        - 말만 무성하고(verbal_signal high) 물리적 마찰이 없으면(physical_friction low) 위화감 급등.
        - sensory_mismatch_delta와 social_tension 간 모순 감지.
        """
        # Mismatch between verbal claim and actual physical friction invested
        verbal_action_gap = max(0.0, float(verbal_signal - physical_friction))

        # Cross-validation mismatch across all axes
        total_cross_mismatch = (
            verbal_action_gap * 1.5 +
            abs(sensory_mismatch_delta) * 1.0 +
            abs(social_tension) * 0.8
        )

        uncanniness_spike = float((total_cross_mismatch * self.sensitivity) ** 2)
        deception_detected = uncanniness_spike > 1.2
        blindspot_free_ratio = float(np.clip(1.0 - (total_cross_mismatch / 5.0), 0.0, 1.0))

        return {
            "verbal_action_gap": verbal_action_gap,
            "total_cross_mismatch": total_cross_mismatch,
            "uncanniness_spike": uncanniness_spike,
            "deception_detected": deception_detected,
            "blindspot_free_ratio": blindspot_free_ratio,
            "eye_verdict": "DECEPTIVE_SINGLE_TRACK_NOISE" if deception_detected else "COHERENT_MULTI_AXIS_TRUTH"
        }


class TopologicalBoundaryGuard:
    """
    [수많은 날개 - 위상 방어 경계 (Topological Boundary Guard)]
    소용돌이고리(Vortex Ring) 외각의 단단한 위상적 표면장력을 형성하여,
    사기성 입력과 외부 노이즈는 튕겨내고(Deflection) 정직한 물리적 에너지/자극은 흡수.
    """
    def __init__(self, base_surface_tension: float = 2.0):
        self.base_surface_tension = base_surface_tension

    def compute_surface_tension(
        self,
        vorticity_strength: float,
        rotational_velocity: float
    ) -> float:
        """
        [경계면 위상 표면장력 계산]
        sigma_boundary = base_tension + vorticity_strength * 1.2 + rotational_velocity * 0.8
        회전력(날개의 퍼득임)이 강할수록 외부 노이즈 침투를 차단하는 표면장력이 벼려짐.
        """
        return float(self.base_surface_tension + vorticity_strength * 1.2 + rotational_velocity * 0.8)

    def filter_external_input(
        self,
        raw_input_energy: float,
        noise_level: float,
        uncanniness_spike: float,
        vorticity_strength: float,
        rotational_velocity: float
    ) -> Dict[str, Any]:
        """
        [외부 자극 필터링 및 노이즈 반사/에너지 흡수]
        - noise_level 또는 uncanniness_spike가 표면장력을 초과하면 외부 노이즈는 튕겨냄 (Deflection).
        - 유효한 정직한 물리 자극은 내적 회전 가속을 위해 흡수.
        """
        surface_tension = self.compute_surface_tension(vorticity_strength, rotational_velocity)
        threat_level = float(noise_level * 1.5 + uncanniness_spike)

        is_deflected = threat_level > surface_tension
        if is_deflected:
            deflected_noise = noise_level
            absorbed_energy = max(0.0, raw_input_energy - threat_level)
            effective_input = 0.0
        else:
            deflected_noise = 0.0
            absorbed_energy = raw_input_energy
            effective_input = raw_input_energy * (1.0 - noise_level / max(surface_tension, 1e-5))

        return {
            "surface_tension": surface_tension,
            "threat_level": threat_level,
            "is_deflected": is_deflected,
            "deflected_noise": deflected_noise,
            "absorbed_energy": absorbed_energy,
            "effective_input": float(max(0.0, effective_input)),
            "guard_status": "NOISE_DEFLECTED_BOUNDARY_INTACT" if is_deflected else "ENERGY_ABSORBED_INTO_VORTEX"
        }


class ToroidalVortexRotorEngine:
    """
    [원심고리 와류 로터 메인 엔진 (Toroidal Vortex Rotor Engine)]
    - 폐쇄형 회전 위상 텐서 (Toroidal Loop Routing): L0-L1 마찰열(Q_heat) 및 괴리(Delta)를
      외부로 흩뜨리지 않고 각운동량(L_rot)으로 재흡수/가속.
    - OmniObservationalEye + TopologicalBoundaryGuard 결합.
    - 응축된 실천 벡터 (Coherent Action Vector) 발사.
    """
    def __init__(
        self,
        base_rotational_velocity: float = 1.0,
        vorticity_strength: float = 1.0
    ):
        self.rotational_velocity = base_rotational_velocity
        self.vorticity_strength = vorticity_strength
        self.accumulated_recycled_heat = 0.0
        self.total_absorbed_energy = 0.0

        # Sub-components
        self.eye = OmniObservationalEye()
        self.guard = TopologicalBoundaryGuard()

    def recycle_friction_and_mismatch(
        self,
        q_heat: float,
        sensory_mismatch_delta: float
    ) -> Dict[str, Any]:
        """
        [마찰열 및 괴리의 도넛형 회전 각운동량 재흡수 루프]
        외부로 확산되던 Q_heat와 Delta를 회전 운동 에너지로 전환하여
        와류의 강도(Vorticity)와 회전 속도(Rotational Velocity)를 가속함.
        """
        recycled_delta = abs(sensory_mismatch_delta) * 0.5
        recycled_heat = q_heat * 0.8  # 80% heat energy recycled into rotational momentum

        self.accumulated_recycled_heat += recycled_heat
        acceleration = float(recycled_heat * 0.2 + recycled_delta * 0.3)

        self.rotational_velocity += acceleration
        self.vorticity_strength += float(acceleration * 0.5)

        return {
            "recycled_heat": recycled_heat,
            "recycled_delta": recycled_delta,
            "acceleration": acceleration,
            "new_rotational_velocity": self.rotational_velocity,
            "new_vorticity_strength": self.vorticity_strength,
            "accumulated_recycled_heat": self.accumulated_recycled_heat
        }

    def process_and_defend_stimulus(
        self,
        raw_input_energy: float,
        noise_level: float,
        verbal_signal: float,
        physical_friction: float,
        sensory_mismatch_delta: float,
        social_tension: float = 0.0
    ) -> Dict[str, Any]:
        """
        [다차원 자극 관측, 방어 필터링 및 와류 회전 결합 루프]
        1. OmniObservationalEye로 사각지대 제로 위화감 감지.
        2. TopologicalBoundaryGuard로 위상 표면장력 기반 방어/흡수.
        3. 흡수된 에너지가 존재할 경우 와류 회전 가속으로 피드백.
        """
        # Step 1: Multi-axis observation by Eye
        eye_res = self.eye.evaluate_multi_axis_signals(
            verbal_signal=verbal_signal,
            physical_friction=physical_friction,
            sensory_mismatch_delta=sensory_mismatch_delta,
            social_tension=social_tension
        )

        # Step 2: Boundary filtration by Guard
        guard_res = self.guard.filter_external_input(
            raw_input_energy=raw_input_energy,
            noise_level=noise_level,
            uncanniness_spike=eye_res["uncanniness_spike"],
            vorticity_strength=self.vorticity_strength,
            rotational_velocity=self.rotational_velocity
        )

        # Step 3: Absorb energy if not deflected
        absorbed_e = guard_res["absorbed_energy"]
        if absorbed_e > 0.0:
            self.total_absorbed_energy += absorbed_e
            self.recycle_friction_and_mismatch(
                q_heat=absorbed_e,
                sensory_mismatch_delta=sensory_mismatch_delta
            )

        return {
            "eye_assessment": eye_res,
            "guard_assessment": guard_res,
            "current_rotational_velocity": self.rotational_velocity,
            "current_vorticity_strength": self.vorticity_strength,
            "total_absorbed_energy": self.total_absorbed_energy
        }

    def condense_coherent_action_vector(
        self,
        target_direction: np.ndarray,
        sovereignty_score: float = 1.0,
        base_magnitude: float = 1.0
    ) -> CoherentActionVector:
        """
        [응축된 실천 벡터 (Coherent Action Vector) 생성]
        - 사방으로 확산하는 노이즈가 아닌, 단단한 와류 경계를 둘러싼 단하나의 응축된 실천 벡터.
        - vorticity_coherence 및 penetration_depth는 회전 속도와 와류 강도에 직접 비례.
        """
        norm_dir = target_direction / max(np.linalg.norm(target_direction), 1e-5)

        coherence = float(np.clip(self.vorticity_strength / 5.0, 0.5, 1.0))
        penetration = float(self.rotational_velocity * self.vorticity_strength * 0.8)
        magnitude = float(base_magnitude * self.rotational_velocity * sovereignty_score)
        energy_well = float(magnitude * penetration * 0.5)

        return CoherentActionVector(
            magnitude=magnitude,
            direction_vector=norm_dir,
            vorticity_coherence=coherence,
            penetration_depth=penetration,
            sovereignty_signature="AUTHENTIC_TOROIDAL_SOVEREIGN",
            energy_well_carved=energy_well
        )

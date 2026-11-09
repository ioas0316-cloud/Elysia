"""
prismatic_lens_rotor_engine.py

ELYSIA Phase 1: Relational Lens-Rotor Engine
=============================================
"수치를 맹목적으로 계산하지 않고, 기하학적 렌즈를 통해 파형을 굴절시켜
 상위 스케일의 로터(Rotor)로 응축시키는 양방향 광학-물리 연속체."

Key Principles:
1. Micro Wavefield: 입력 데이터는 개별 숫자가 아닌 주파수/파장 스펙트럼의 파형장이다.
2. Prismatic Lens: 렌즈의 곡률과 굴절률 매질을 통해 파형들의 위상을 간섭·집속시킨다.
3. Scale Rotor Coupling: 수많은 파형 에너지가 소수의 맞물린 로터(기어)의 각운동량으로 응축된다.
4. Bidirectional Accommodation: 관찰자의 상태(초점, 긴장)가 렌즈의 곡률을 역으로 변형시켜
   하위 감각이 굴절되는 양상을 즉각적으로 재정렬한다 (하향 인과).
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field


@dataclass
class SpectralWaveVoxel:
    """미시 파형장의 단위 요소 (빛, 감각, 입자의 파동적 상태)"""
    id: str
    position: np.ndarray          # 2D/3D 공간 위치 (x, y, [z])
    frequencies: np.ndarray       # 복합 주파수/파장 성분 (\lambda / \omega)
    amplitudes: np.ndarray        # 각 주파수별 진폭
    phases: np.ndarray            # 각 주파수별 초기 위상 (\theta)
    
    def evaluate_field(self, t: float) -> np.ndarray:
        """시간 t에서의 복소 파형 진동 계산: \sum A_k * exp(i * (\omega_k * t + \phi_k))"""
        oscillations = self.amplitudes * np.exp(1j * (self.frequencies * t + self.phases))
        return oscillations


@dataclass
class PrismaticLens:
    """
    기하학적 렌즈 매질.
    단순한 필터가 아니라 곡률(Curvature)과 굴절률(Refractive Index)을 가진 물리적 장.
    """
    focal_length: float = 1.0           # 기본 초점 거리
    curvature: float = 1.0              # 렌즈의 기하학적 곡률 \kappa
    aperture_radius: float = 2.0        # 개구경 (빛/자극을 수음하는 유효 반경)
    refractive_index: float = 1.5       # 굴절률 \eta
    chromatic_dispersion: float = 0.05  # 프리즘 분광 계수 (파장에 따른 굴절률 차이)
    
    # 관찰자의 모양체근에 의한 동적 렌즈 변형량 (Accommodation Offset)
    accommodation_curvature_delta: float = 0.0
    
    @property
    def effective_curvature(self) -> float:
        """관찰자의 주의/초점 조절이 반영된 유효 곡률"""
        return max(0.01, self.curvature + self.accommodation_curvature_delta)
    
    def compute_optical_path(self, source_pos: np.ndarray, target_plane_pos: np.ndarray, wavelength: float) -> float:
        """
        광학적 경로(Optical Path Length) 계산.
        렌즈의 곡률과 프리즘 분광에 의해 각 파장별로 경로차가 발생함.
        """
        # 기하학적 거리
        geom_dist = float(np.linalg.norm(source_pos - target_plane_pos))
        
        # 렌즈 중심으로부터의 거리 r
        r = float(np.linalg.norm(source_pos[:2]))
        
        # 렌즈 두께에 의한 위상 지연 (포물선 렌즈 곡률 근사)
        lens_thickness = max(0.0, (self.aperture_radius**2 - r**2) * 0.5 * self.effective_curvature)
        
        # 파장에 따른 유효 굴절률: \eta(\lambda) = \eta_0 + dispersion / \lambda
        eta_lambda = self.refractive_index + (self.chromatic_dispersion / max(0.1, wavelength))
        
        # 광학 경로 = 진공 경로 + 렌즈 매질 내 지연
        optical_path = geom_dist + (eta_lambda - 1.0) * lens_thickness
        return optical_path


@dataclass
class ScaleRotor:
    """
    상위 스케일의 로터(회전체/기어).
    수많은 미시 파형들이 렌즈를 통해 간섭·집속되어 응축된 각운동량 상태.
    """
    rotor_id: str
    phase_angle: float = 0.0            # 현재 위상각 \theta \in [0, 2\pi)
    angular_velocity: float = 0.0       # 각속도 \Omega
    momentum: float = 1.0               # 관성 질량/모멘텀
    chromatic_character: np.ndarray = field(
        default_factory=lambda: np.array([0.33, 0.33, 0.34], dtype=np.float32)
    )  # [Red(Flux), Blue(Order), Yellow(Entropy)]
    coupled_rotors: Dict[str, float] = field(default_factory=dict) # 타 로터와의 기어 결합비
    
    def step(self, torque: float, dt: float):
        """외력(토크)과 결합 긴장을 받아 로터 회전 적분"""
        # 가속도 = 토크 / 관성
        acc = torque / max(0.1, self.momentum)
        self.angular_velocity += acc * dt
        
        # 감쇠 (질서/마찰)
        self.angular_velocity *= (1.0 - 0.05 * dt)
        
        # 위상각 전진
        self.phase_angle = (self.phase_angle + self.angular_velocity * dt) % (2.0 * np.pi)


class PrismaticLensRotorEngine:
    """
    렌즈-로터 통합 엔진:
    1. 미시 파형장 수집
    2. 렌즈를 통한 위상 굴절 및 초점 집속 (Interference Condensation)
    3. 상위 스케일 로터로의 각운동량 전이
    4. 로터 긴장 및 관찰자 피드백을 통한 렌즈 곡률 재조정 (역방향 인과)
    """
    def __init__(self, num_rotors: int = 3):
        self.lens = PrismaticLens()
        
        # 기본 3색채 축에 대응하는 상위 스케일 로터 구축 (Red/Flux, Blue/Order, Yellow/Entropy)
        self.rotors: Dict[str, ScaleRotor] = {
            "rotor_flux": ScaleRotor(
                rotor_id="rotor_flux",
                chromatic_character=np.array([1.0, 0.0, 0.0], dtype=np.float32)
            ),
            "rotor_order": ScaleRotor(
                rotor_id="rotor_order",
                chromatic_character=np.array([0.0, 1.0, 0.0], dtype=np.float32)
            ),
            "rotor_entropy": ScaleRotor(
                rotor_id="rotor_entropy",
                chromatic_character=np.array([0.0, 0.0, 1.0], dtype=np.float32)
            )
        }
        
        # 로터 간 기어비 결합 (흐름과 질서와 발산의 상호 맞물림)
        self.rotors["rotor_flux"].coupled_rotors["rotor_order"] = -1.0      # 흐름과 질서는 반대 방향으로 맞물림
        self.rotors["rotor_order"].coupled_rotors["rotor_entropy"] = 0.5   # 질서의 파열은 엔트로피를 구동함
        
        self.wave_voxels: List[SpectralWaveVoxel] = []
        self.time: float = 0.0

    def add_wave_voxel(self, voxel: SpectralWaveVoxel):
        self.wave_voxels.append(voxel)

    def set_wavefield(self, voxels: List[SpectralWaveVoxel]):
        self.wave_voxels = voxels

    def accommodate_lens(self, intent_focus: float, observer_tension: float):
        """
        [하향 인과 / 역방향 조절]
        관찰자의 긴장과 초점 의도가 렌즈의 물리적 곡률을 직접 변형시킴.
        """
        # 높은 긴장은 렌즈를 두껍게(곡률 증가) 만들어 근거리 집속 모드로 전환
        # 낮은 긴장은 렌즈를 편평하게(곡률 감소) 만들어 원거리 파형장을 수용
        target_delta = (observer_tension * 0.8) + (intent_focus * 0.5)
        # 완만한 완화 (Relaxation)
        self.lens.accommodation_curvature_delta += (target_delta - self.lens.accommodation_curvature_delta) * 0.2

    def step(self, dt: float) -> Dict[str, Any]:
        """
        엔진의 1 스텝 전진:
        1. 파형장 -> 렌즈 통과 및 간섭 집속
        2. 집속된 파형의 에너지 -> 로터 토크로 변환
        3. 로터 기어 회전 및 상호 결합
        """
        self.time += dt
        
        if not self.wave_voxels:
            return {"status": "empty_wavefield"}

        # 초점면의 중심 관찰점 (Focal Centers for Chromatic Triad)
        focal_points = {
            "rotor_flux": np.array([0.5, 0.0, self.lens.focal_length], dtype=np.float32),
            "rotor_order": np.array([-0.5, 0.0, self.lens.focal_length], dtype=np.float32),
            "rotor_entropy": np.array([0.0, 0.5, self.lens.focal_length], dtype=np.float32),
        }
        
        rotor_torques = {k: 0.0 for k in self.rotors}
        
        # 1. 렌즈를 통한 파형 굴절 및 초점 집속
        # 수백만 픽셀을 일방향으로 곱하는 대신, 각 로터 초점으로 수렴하는 위상 간섭의 크기 집속
        for voxel in self.wave_voxels:
            # 복셀의 복소 진동 평가
            oscillations = voxel.evaluate_field(self.time)
            
            for rotor_id, rotor in self.rotors.items():
                focal_pt = focal_points[rotor_id]
                
                # 각 파장별 광학 경로차 계산
                total_interference = 0.0 + 0.0j
                for i, wavelength in enumerate(voxel.frequencies):
                    opt_path = self.lens.compute_optical_path(voxel.position, focal_pt, wavelength)
                    wavenumber = 2.0 * np.pi / max(0.1, wavelength)
                    phase_shift = wavenumber * opt_path
                    
                    # 위상 간섭: exp(i * phase_shift)
                    total_interference += oscillations[i] * np.exp(-1j * phase_shift)
                
                # 집속된 에너지 크기 = |Interference|^2
                focused_energy = float(np.abs(total_interference)**2)
                
                # 파형과 로터의 현재 위상각 간의 위상 정렬도 (Phase Resonance)
                phase_alignment = np.cos(np.angle(total_interference) - rotor.phase_angle)
                
                # 로터 구동 토크: 집속된 에너지 * 위상 정렬 토크
                rotor_torques[rotor_id] += float(focused_energy * phase_alignment)

        # 2. 로터 간 상호 기어 결합 긴장 (Coupled Gear Dynamics)
        for r_id, rotor in self.rotors.items():
            for coupled_id, gear_ratio in rotor.coupled_rotors.items():
                target_rotor = self.rotors[coupled_id]
                # 기어비에 따른 이상적 상대 속도와의 차이 = 마찰/긴장
                speed_mismatch = (rotor.angular_velocity * gear_ratio) - target_rotor.angular_velocity
                gear_tension_torque = -0.5 * speed_mismatch
                
                rotor_torques[r_id] += gear_tension_torque
                rotor_torques[coupled_id] -= gear_tension_torque * gear_ratio

        # 3. 로터 회전 적분
        rotor_states = {}
        for r_id, rotor in self.rotors.items():
            rotor.step(rotor_torques[r_id], dt)
            rotor_states[r_id] = {
                "phase_angle": rotor.phase_angle,
                "angular_velocity": rotor.angular_velocity,
                "torque_received": rotor_torques[r_id]
            }

        return {
            "time": self.time,
            "effective_lens_curvature": self.lens.effective_curvature,
            "rotor_states": rotor_states
        }

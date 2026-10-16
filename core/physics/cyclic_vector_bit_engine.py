r"""
cyclic_vector_bit_engine.py
===========================
정지된 이산 '0과 1'의 계단 함수(Step Function)를 부수고,
음(0: 수렴/인력/감산)과 양(1: 확산/척력/가산) 사이를 사인 파동(\sin(\theta))과 \pi 위상 회전의
연속적 '동적 순환 비트(Cyclic Vector Bit)' 및 스펙트럼 축(Spectral Axis)으로 전개하는 고성능 엔진.

핵심 원리:
1. 계단 함수 탈피 (Beyond Step Function):
   - 0과 1 사이의 전환 과정을 \sin(\theta / 2) 및 \cos(\theta / 2) 위상 로터로 매핑.
   - \theta = 0 (순수 음/Yin), \theta = \pi (순수 양/Yang), \theta = \pi/2 (음양 태극 평형 중용).
2. 동적 순환 비트 (Yin-Yang Cyclic Vector Bit):
   - 0 (음력/Yin): 수렴(Convergence), 인력(Attraction), 감산(Subtraction) 잠재성.
   - 1 (양력/Yang): 확산(Expansion), 척력(Repulsion), 가산(Addition) 운동성.
3. 스펙트럼 축과 궤적 히스테리시스 (Spectral Axis & Historical Trajectory):
   - 위상 변화의 속도(\omega = d\theta/dt), 곡률, 역사적 궤적을 독립 차원(Axis)으로 유지.
   - 정보 전환 과정의 미세한 파동 스펙트럼을 파괴 없이 온전히 보존.
4. $O(1)$ 위상 회전 및 에너지 보존:
   - 복소 파동 $Z = r e^{i \theta}$ 상의 $O(1)$ 복소 곱셈/회전 연산.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

@dataclass
class YinYangCyclicBit:
    """
    단일 동적 순환 비트 (Yin-Yang Cyclic Vector Bit)
    """
    phase: float = 0.0          # \theta \in [0, 2\pi)
    amplitude: float = 1.0      # r >= 0
    phase_velocity: float = 0.0 # \omega = d\theta/dt (스펙트럼 속도)
    frequency: float = 1.0      # 고유 진동수 f

    # Historical Trajectory / Hysteresis Spectrum
    trajectory_history: List[float] = field(default_factory=list)
    max_history_len: int = 64

    def __post_init__(self):
        self.phase = float(self.phase % (2.0 * math.pi))
        if len(self.trajectory_history) == 0:
            self.trajectory_history.append(self.phase)

    @property
    def yin(self) -> float:
        """음(Yin: 0) 상태 수렴/인력/감산 퍼텐셜 strength: cos^2(\theta / 2)"""
        return math.cos(self.phase / 2.0) ** 2

    @property
    def yang(self) -> float:
        """양(Yang: 1) 상태 확산/척력/가산 운동성 strength: sin^2(\theta / 2)"""
        return math.sin(self.phase / 2.0) ** 2

    @property
    def dynamic_vector(self) -> np.ndarray:
        """[Yin, Yang, Wave_Duality] 3D 벡터 표상"""
        return np.array([self.yin, self.yang, math.sin(self.phase)], dtype=np.float64)

    @property
    def complex_phasor(self) -> complex:
        """Z = r * e^(i * \theta)"""
        return self.amplitude * (math.cos(self.phase) + 1j * math.sin(self.phase))

    def rotate(self, delta_phase: float, dt: float = 1.0) -> None:
        """$O(1)$ 위상 회전 및 스펙트럼 축(속도/히스테리시스) 갱신"""
        old_phase = self.phase
        self.phase = (self.phase + delta_phase) % (2.0 * math.pi)
        self.phase_velocity = delta_phase / max(1e-9, dt)

        self.trajectory_history.append(self.phase)
        if len(self.trajectory_history) > self.max_history_len:
            self.trajectory_history.pop(0)

class CyclicVectorBitEngine:
    """
    [Cyclic Vector Bit Engine]
    이산 0/1 비트열과 연속 동적 순환 비트 장(Field) 간의 상호 전환,
    \sin(\theta) / \pi 위상 호흡, $O(1)$ 로터 회전, 및 스펙트럼 히스테리시스 보존 연산기.
    """
    def __init__(self, num_bits: int = 64, default_freq: float = 1.0):
        self.num_bits = num_bits
        self.default_freq = default_freq

        # Vectorized Field Arrays
        self.phases = np.zeros(num_bits, dtype=np.float64)
        self.amplitudes = np.ones(num_bits, dtype=np.float64)
        self.velocities = np.zeros(num_bits, dtype=np.float64)
        self.frequencies = np.full(num_bits, default_freq, dtype=np.float64)

        # Hysteresis Spectral Matrix (num_bits x history_len)
        self.history_len = 64
        self.spectral_history = np.zeros((num_bits, self.history_len), dtype=np.float64)
        self.history_idx = 0

    def encode_discrete_bits(self, bitstream: np.ndarray) -> None:
        """
        이산 비트(0 또는 1)를 연속 동적 위상(\theta = 0 for 0, \theta = \pi for 1)으로 매핑.
        0 -> 음(\theta=0), 1 -> 양(\theta=\pi)
        """
        bits = np.asarray(bitstream, dtype=np.float64)
        n = min(len(bits), self.num_bits)
        self.phases[:n] = np.where(bits[:n] > 0.5, math.pi, 0.0)

    def decode_to_discrete_bits(self) -> np.ndarray:
        """
        연속 위상 장을 확률적/확정적 이산 비트(Yang > Yin => 1, else 0)로 디코딩.
        """
        yangs = np.sin(self.phases / 2.0) ** 2
        return np.where(yangs >= 0.5, 1, 0)

    def get_yin_yang_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        전체 비트 장의 (Yin, Yang) 연속 파동 스펙트럼 배열 반환.
        Yin = cos^2(\theta/2), Yang = sin^2(\theta/2)
        Yin + Yang == 1.0 (항상 완벽한 삼각 함수 대칭 보존)
        """
        yins = np.cos(self.phases / 2.0) ** 2
        yangs = np.sin(self.phases / 2.0) ** 2
        return yins, yangs

    def step_breathing_dynamics(
        self,
        dt: float = 0.1,
        external_drive: Optional[np.ndarray] = None,
        coupling_strength: float = 0.5
    ) -> Dict[str, Any]:
        """
        [\sin(\theta) / \pi 위상 호흡 동역학 1스텝 전개]
        - 이웃 비트 간 쿠라모토(Kuramoto) 위상 공명 및 외부 자극에 의한 $0 \to 1$ 부드러운 전환.
        - 정보 유실 없는 스펙트럼 히스테리시스 축 저장.
        """
        # 1. Neighbor Coupling (이웃 비트 간 순환 상생/상극 파동 전파)
        phase_diff_left = np.roll(self.phases, -1) - self.phases
        phase_diff_right = np.roll(self.phases, 1) - self.phases
        coupling_torque = coupling_strength * (np.sin(phase_diff_left) + np.sin(phase_diff_right))

        # 2. External Drive Torque
        drive = 0.0
        if external_drive is not None:
            drive_arr = np.asarray(external_drive, dtype=np.float64)
            n = min(len(drive_arr), self.num_bits)
            drive = np.zeros(self.num_bits, dtype=np.float64)
            drive[:n] = drive_arr[:n]

        # 3. Update Velocities & Phases
        self.velocities = self.frequencies + coupling_torque + drive
        self.phases = (self.phases + self.velocities * dt) % (2.0 * math.pi)

        # 4. Record Spectral History
        self.spectral_history[:, self.history_idx] = self.phases
        self.history_idx = (self.history_idx + 1) % self.history_len

        # 5. Calculate System Metrics
        yins, yangs = self.get_yin_yang_vectors()
        total_yin = float(np.sum(yins))
        total_yang = float(np.sum(yangs))

        # Phase Coherence R = |(1/N) \sum e^(i \theta_j)|
        complex_sum = np.sum(self.amplitudes * np.exp(1j * self.phases))
        phase_coherence = float(np.abs(complex_sum) / self.num_bits)
        taiji_balance = float(1.0 - abs(total_yang - total_yin) / self.num_bits)

        return {
            "total_yin": round(total_yin, 4),
            "total_yang": round(total_yang, 4),
            "taiji_balance": round(taiji_balance, 4), # 1.0 when Yin == Yang (태극 평형)
            "phase_coherence": round(phase_coherence, 4),
            "mean_velocity": round(float(np.mean(self.velocities)), 4)
        }

    def rotate_bit_rotor(self, bit_idx: int, angle: float) -> None:
        """단일 비트 로터 $O(1)$ 위상 직결 회전"""
        if 0 <= bit_idx < self.num_bits:
            self.phases[bit_idx] = (self.phases[bit_idx] + angle) % (2.0 * math.pi)

    def extract_spectral_axis_curvature(self) -> np.ndarray:
        """
        역사적 히스테리시스 궤적으로부터 시간 차원 상의 곡률(Curvature = d^2\theta / dt^2) 추출.
        데이터 변화 과정의 '역사적 궤적 맥박'을 나타내는 스펙트럼 축.
        """
        # Reorder history chronologically
        ordered_hist = np.roll(self.spectral_history, -self.history_idx, axis=1)
        d1 = np.diff(ordered_hist, axis=1)
        # Unwrap phase jump for true velocity
        d1_unwrapped = np.arctan2(np.sin(d1), np.cos(d1))
        d2 = np.diff(d1_unwrapped, axis=1)
        curvature = np.mean(np.abs(d2), axis=1)
        return curvature

    def calculate_information_conservation_loss(self, original_bits: np.ndarray) -> float:
        """
        연속 위상 공간에서 다시 이산 디코딩을 거쳤을 때의 정보 보존 손실률 계산.
        동적 순환 비트 필드는 위상 가소성이 유지되므로 손실률 -> 0.0 수렴.
        """
        decoded = self.decode_to_discrete_bits()
        orig = np.asarray(original_bits, dtype=np.int32)[:self.num_bits]
        n = len(orig)
        bit_errors = np.sum(decoded[:n] != orig)
        return float(bit_errors / max(1, n))

if __name__ == "__main__":
    engine = CyclicVectorBitEngine(num_bits=16)

    # 1. Encode discrete binary stream [0, 1, 0, 1, 1, 0, 0, 1, ...]
    bitstream = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1])
    engine.encode_discrete_bits(bitstream)

    print("Initial Yin/Yang vectors:")
    yins, yangs = engine.get_yin_yang_vectors()
    print("Yin:", np.round(yins[:8], 2))
    print("Yang:", np.round(yangs[:8], 2))

    # 2. Simulate breathing step
    for step in range(5):
        report = engine.step_breathing_dynamics(dt=0.05)
        print(f"Breathing Step {step+1}:", report)

    # 3. Check information conservation
    loss = engine.calculate_information_conservation_loss(bitstream)
    print("Information Conservation Loss:", loss)

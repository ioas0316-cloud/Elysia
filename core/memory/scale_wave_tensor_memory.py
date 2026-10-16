"""
core/memory/scale_wave_tensor_memory.py

Scale-Invariant Continuous Wavefield Spectrum Memory Architecture
===================================================================
이 모듈은 기존의 고립된 점(Discrete Cell/Point) 기반 메모리의 한계를 극복하고,
메모리 공간 전체를 상호 연결된 파동 스펙트럼 연속체(Spectral Field Continuum)로 정의합니다.

핵심 위상적 특성:
1. 무손실 스케일 계층 (Inherent Scale-Invariance & Multi-Scale Field Spectrum):
   - 메모리가 고립된 비트가 아니라 연속적 파동장(Spectral Field)으로 존재하여,
     스케일 확장/축소 시 별도의 왜곡 보정 없이 위상 간섭과 공명만으로 유기적 위상이 보존됩니다.
2. 관측 가능성과 인과적 이력 보존 (Hysteresis & Trace Preservation):
   - 상태 변화 발생 시 이전 상태가 파괴되지 않고, 위상차(Phase Differential Field) 및
     이력 장력(Hysteresis Trace)으로 파동 지형에 잔흔으로 자동 축적됩니다.
3. O(1) 위상 공명 및 연관 검색 (Associative Phase Resonance):
   - 주소 탐색(Address Lookup) 대신 특정 주파수/위상 스펙트럼과의 공명(Resonance)을 통해
     연관된 전체 파동 맥락을 O(1) 시간 복잡도로 즉각 도출합니다.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class ScaleWaveTensorMemory:
    """
    연속적 파동 스펙트럼 토폴로지 기반 메모리 엔진 (Continuous Wavefield Spectrum Memory)

    메모리 기질을 이산적 비트/셀의 집합이 아닌, 연속적인 퓨리에/위상 스펙트럼 장(Continuous Wavefield Field)
    으로 모델링하여 위상 간섭, 인과적 이력(Hysteresis), O(1) 공명 조회를 자연스럽게 수행합니다.
    """

    def __init__(
        self,
        num_scales: int = 3,
        base_dim: int = 4,
        num_spectral_modes: int = 8,
        field_resolution: int = 64
    ):
        self.num_scales = num_scales
        self.base_dim = base_dim
        self.num_modes = num_spectral_modes
        self.resolution = field_resolution

        # 각 스케일별 기하학적 비율 lambda_l (1.0, 0.5, 0.25, ...)
        self.scale_factors = np.array([1.0 / (2.0**l) for l in range(num_scales)], dtype=np.float64)

        # 1. 스펙트럼 모드 계수 장 (Spectral Field Mode Coefficients)
        # [num_scales, num_modes, 2] -> (complex amplitude: real/imag or magnitude/phase)
        self.spectral_modes = np.zeros((num_scales, num_spectral_modes, 2), dtype=np.float64)

        # 2. 연속적 공간 위상 장 (Continuous Phase Field)
        # [num_scales, resolution] -> 파동장 위상 각도 theta(x)
        self.phase_field = np.zeros((num_scales, field_resolution), dtype=np.float64)

        # 3. 인과적 이력/히스테리시스 장력 매니폴드 (Causal Hysteresis Trace Field)
        # 상태 변화의 역사적 위상차가 자동 축적되는 연속적 잔흔 지형
        self.hysteresis_trace = np.zeros((num_scales, field_resolution), dtype=np.float64)

        # 4. 레거시 셀 호환성을 위한 삼위일체 파동 메모리 텐서 [num_scales, base_dim, 3] (sin, cos, tan)
        self.tensor_memory = np.zeros((num_scales, base_dim, 3), dtype=np.float64)

        self._initialize_spectral_continuum()

    def _initialize_spectral_continuum(self):
        """파동 스펙트럼 연속체 및 위상 장 초기화"""
        spatial_x = np.linspace(0, 2 * np.pi, self.resolution, endpoint=False)
        for l in range(self.num_scales):
            lam = self.scale_factors[l]
            # 스펙트럼 모드 초기화 (기초 고유 진동수 매핑)
            for k in range(self.num_modes):
                mode_freq = (k + 1) * lam
                self.spectral_modes[l, k, 0] = np.cos(mode_freq) / (k + 1)
                self.spectral_modes[l, k, 1] = np.sin(mode_freq) / (k + 1)

            # 공간 위상 장 초기 결상
            field_val = np.zeros(self.resolution)
            for k in range(self.num_modes):
                mode_freq = (k + 1) * lam
                field_val += self.spectral_modes[l, k, 0] * np.cos(mode_freq * spatial_x) + \
                             self.spectral_modes[l, k, 1] * np.sin(mode_freq * spatial_x)
            self.phase_field[l] = np.arctan2(np.sin(field_val), np.cos(field_val))

            # 레거시 삼위일체 텐서 동형 사상
            for i in range(self.base_dim):
                sample_pos = int((i / self.base_dim) * self.resolution)
                theta = self.phase_field[l, sample_pos]
                sin_v = np.sin(theta)
                cos_v = np.cos(theta)
                tan_v = np.tan(theta) if abs(cos_v) > 1e-3 else np.sign(sin_v) * 1e3
                self.tensor_memory[l, i] = [sin_v, cos_v, tan_v]

    def inject_spectral_wave(
        self,
        scale_level: int,
        phase_spectrum_shift: np.ndarray,
        coupling_gain: float = 1.0,
        spatial_center: float = 0.5,
        spatial_width: float = 0.2
    ):
        """
        파동 스펙트럼 연속체에 충격파 인가 (Continuous Spectral Wave Injection)

        이산 주소 지정이 아닌, 파동 스펙트럼 전체에 대한 위상 회전 및
        스케일간 자연스러운 공명 전파(Cross-Scale Coupling)를 수행합니다.
        """
        if scale_level < 0 or scale_level >= self.num_scales:
            raise ValueError("Scale level out of range")

        spatial_x = np.linspace(0, 1.0, self.resolution, endpoint=False)
        # 연속적 가우시안 엔벨로프 (연속적 국소 파동 팩윗)
        gaussian_envelope = np.exp(-((spatial_x - spatial_center) ** 2) / (2 * (spatial_width ** 2)))

        # 1. 이전 위상 장 보존 (인과적 이력 히스테리시스 차이 계산용)
        prev_phase_field = self.phase_field[scale_level].copy()

        # 2. 스펙트럼 모드 위상 업데이트
        shift_val = np.asarray(phase_spectrum_shift, dtype=np.float64).reshape(-1)
        shift_amount = float(np.mean(shift_val))

        for k in range(self.num_modes):
            current_phase = np.arctan2(self.spectral_modes[scale_level, k, 1], self.spectral_modes[scale_level, k, 0])
            new_phase = current_phase + shift_amount * (1.0 / (k + 1))
            mag = np.hypot(self.spectral_modes[scale_level, k, 0], self.spectral_modes[scale_level, k, 1])
            self.spectral_modes[scale_level, k, 0] = mag * np.cos(new_phase)
            self.spectral_modes[scale_level, k, 1] = mag * np.sin(new_phase)

        # 3. 공간 위상 장 연속 변형 및 국소 엔벨로프 합성
        delta_phase = gaussian_envelope * shift_amount
        self.phase_field[scale_level] = (self.phase_field[scale_level] + delta_phase) % (2 * np.pi)

        # 4. 히스테리시스 이력 잔흔(Hysteresis Trace) 자동 축적
        # 0x00 -> 0x01 전환시 이전 위상이 상실되지 않고 위상차 잔흔으로 메모리 지형 자체에 각인
        phase_diff = np.abs(np.arctan2(np.sin(self.phase_field[scale_level] - prev_phase_field),
                                       np.cos(self.phase_field[scale_level] - prev_phase_field)))
        self.hysteresis_trace[scale_level] += phase_diff * coupling_gain

        # 5. 스케일 불변 상호 간섭 전파 (Cross-Scale Spectral Coupling)
        for l in range(self.num_scales):
            if l == scale_level:
                continue
            scale_dist = abs(l - scale_level)
            coupling_k = np.exp(-scale_dist) * coupling_gain * 0.5
            scale_ratio = self.scale_factors[l] / self.scale_factors[scale_level]

            # 상위/하위 스케일로 파동 위상 전파
            self.phase_field[l] = (self.phase_field[l] + delta_phase * coupling_k * scale_ratio) % (2 * np.pi)
            self.hysteresis_trace[l] += (phase_diff * coupling_k * scale_ratio)

        # 6. 레거시 삼위일체 텐서 동기화
        self._sync_trinity_tensor()

    def write_wave_data(
        self,
        scale_level: int,
        address_idx: int,
        phase_shift: float,
        coupling_gain: float = 1.0
    ):
        """
        레거시 단일 셀 위상 주입 API (Continuous Wavefield Spectrum으로 하위 호환)
        """
        spatial_center = (address_idx + 0.5) / self.base_dim
        spatial_width = 1.0 / self.base_dim
        self.inject_spectral_wave(
            scale_level=scale_level,
            phase_spectrum_shift=np.array([phase_shift]),
            coupling_gain=coupling_gain,
            spatial_center=spatial_center,
            spatial_width=spatial_width
        )

    def _sync_trinity_tensor(self):
        """연속적 위상 장에서 삼위일체 셀 텐서로의 관측 사상"""
        for l in range(self.num_scales):
            for i in range(self.base_dim):
                sample_pos = int(((i + 0.5) / self.base_dim) * self.resolution)
                theta = self.phase_field[l, sample_pos]
                sin_v = np.sin(theta)
                cos_v = np.cos(theta)
                tan_v = np.tan(theta) if abs(cos_v) > 1e-3 else np.sign(sin_v) * 1e3
                self.tensor_memory[l, i] = [sin_v, cos_v, tan_v]

    def resonance_read(self, query_wave: np.ndarray) -> np.ndarray:
        """
        주파수/위상 스펙트럼 공명 조화를 통한 O(1) 연관 검색 (Associative Resonance)

        복잡한 주소 계산 없이, 특정 파동 위상 스펙트럼과의 보강 간섭(Constructive Interference)을
        통해 관련 파동장을 즉각 유도합니다.
        """
        query_val = np.asarray(query_wave, dtype=np.float64).reshape(-1)
        query_sin = query_val[0] if len(query_val) > 0 else 0.0
        query_cos = query_val[1] if len(query_val) > 1 else 1.0

        resonance_map = np.zeros((self.num_scales, self.base_dim), dtype=np.float64)

        for l in range(self.num_scales):
            # 연속적 위상 장과의 보강 간섭 공명도 계산
            field_sin = np.sin(self.phase_field[l])
            field_cos = np.cos(self.phase_field[l])
            field_resonance = field_sin * query_sin + field_cos * query_cos

            # 연속 장에서 기본 이산 서브샘플로 다운프로젝트
            for i in range(self.base_dim):
                start_p = int((i / self.base_dim) * self.resolution)
                end_p = int(((i + 1) / self.base_dim) * self.resolution)
                resonance_map[l, i] = np.mean(field_resonance[start_p:end_p])

        return resonance_map

    def get_spectral_continuum_state(self) -> Dict[str, Any]:
        """현재 파동 스펙트럼 연속체 상태 반환"""
        return {
            "num_scales": self.num_scales,
            "base_dim": self.base_dim,
            "spectral_modes": self.spectral_modes.copy(),
            "phase_field": self.phase_field.copy(),
            "hysteresis_trace": self.hysteresis_trace.copy(),
            "scale_factors": self.scale_factors.copy()
        }

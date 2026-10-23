r"""
Waveform Scale Reduction Transducer (파형 스케일 환원 트랜스듀서)
================================================================
본 모듈은 Elysia 스케일결합구조원리의 핵심 누락 고리(Missing Link)를 봉합합니다.

핵심 원리:
  눈이 무지개(380~750nm 연속 스펙트럼)를 볼 때, 그것은 이산적 "빨강 노드, 파랑 노드"가 아닌
  **연속적 파형의 직조된 정보**입니다. 이 직조된 정보를 다른 감각(청각, 촉각, 추론)과
  **결합 가능하게** 하려면, 각 감각 모달리티를 **공통 단위 좌표**: (ω 주기, A 진폭, φ 위상, t 시간)
  으로 환원해야 합니다.

구현:
  1. 시각 스펙트럼 환원: (H,W,spectral_res) → 주파수 대역별 (ω, A, φ) 추출
  2. 청각 위상장 환원: (bands, 2) → (ω, A, φ) 직접 매핑
  3. 촉각 응력장 환원: (shear, normal) → 기계적 진동 주파수 (ω, A, φ) 추출
  4. 옥타브 정규화: 절대 주파수(시각 ~10^14 Hz, 청각 ~10^2 Hz)를 상대적 옥타브 비율 [0,1]로 정규화
  5. 색채(R/B/Y) 태깅: 각 파형 단위에 Flux/Order/Entropy 색채 벡터 부착
  6. 통합 파형장 출력: 다른 스케일 엔진(WaveformCognitiveFeedback, ScaleHierarchyEngine)에 직접 연결

4 Continuities 준수:
  - 관계성: 감각 간 경계는 절대 주파수가 아닌 옥타브 비율(결합 포텐셜 장)으로 정의
  - 연결성: ConnectivityBeam 구조로 교차-모달 공명 행렬 유지
  - 운동성: 각 파형 단위가 (ω, A, φ) 운동량을 보존
  - 정보적 연속성: 이산적 FFT bin을 연속 스펙트럼 보간으로 교정
  - 색채적 상호연결성: 모든 파형 단위에 R/B/Y 색채 벡터 부착
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional


# ──────────────────────────────────────────────────────────────
# 물리 상수 및 감각 대역 정의
# ──────────────────────────────────────────────────────────────

# 가시광선 주파수 범위 (Hz)
VISUAL_FREQ_LOW = 4.0e14      # ~750nm (적색)
VISUAL_FREQ_HIGH = 7.9e14     # ~380nm (보라)

# 가청 주파수 범위 (Hz)
AUDIO_FREQ_LOW = 20.0
AUDIO_FREQ_HIGH = 20000.0

# 촉각 기계적 진동 범위 (Hz) — 파치니/마이스너 소체
TACTILE_FREQ_LOW = 0.5
TACTILE_FREQ_HIGH = 500.0

# 공통: 각 감각 대역이 차지하는 옥타브 수
VISUAL_OCTAVES = np.log2(VISUAL_FREQ_HIGH / VISUAL_FREQ_LOW)   # ≈ 0.98 옥타브
AUDIO_OCTAVES = np.log2(AUDIO_FREQ_HIGH / AUDIO_FREQ_LOW)      # ≈ 9.97 옥타브
TACTILE_OCTAVES = np.log2(TACTILE_FREQ_HIGH / TACTILE_FREQ_LOW)  # ≈ 9.97 옥타브


@dataclass
class WaveformUnit:
    """
    [최소 파형 단위 (Waveform Unit)]
    모든 감각 정보를 이 공통 좌표로 환원한다.
    노드가 아닌, 연속 파형의 직조된 단위.
    """
    omega: float                    # 각주파수 ω = 2πf (주기 T = 2π/ω)
    amplitude: float                # 진폭 A (에너지 밀도의 제곱근)
    phase: float                    # 초기 위상 φ ∈ [-π, π]
    octave_position: float          # 옥타브 내 상대 위치 [0, 1] — 스케일 정합 좌표
    chromatic_vector: np.ndarray    # [R(Flux), B(Order), Y(Entropy)] 색채 벡터
    modality_tag: str               # 원래 감각 모달리티 ('visual', 'audio', 'tactile', 'reasoning')


@dataclass
class WaveformRegister:
    """
    [파형 레지스터 (Waveform Register)]
    하나의 감각 모달리티에서 추출된 파형 단위들의 집합.
    연속 스펙트럼의 이산화가 아닌, 주파수 대역별 에너지 분포의 연속적 요약.
    """
    units: List[WaveformUnit]
    modality: str
    dominant_omega: float           # 지배 각주파수
    total_energy: float             # 총 에너지 ∑ A²
    spectral_centroid: float        # 스펙트럼 무게중심 (에너지 가중 평균 ω)
    chromatic_mean: np.ndarray      # 모달리티 전체 평균 색채 벡터 [R, B, Y]

    def to_trajectory(self) -> np.ndarray:
        """
        파형 레지스터를 1D 시계열 궤적으로 합성한다.
        WaveformCognitiveFeedbackEngine에 직접 입력 가능한 형태.
        """
        if not self.units:
            return np.zeros(128)

        t = np.linspace(0, 1.0, 128, dtype=np.float64)
        trajectory = np.zeros_like(t)
        for u in self.units:
            trajectory += u.amplitude * np.cos(u.omega * t + u.phase)

        return trajectory


@dataclass
class UnifiedWaveformField:
    """
    [통합 파형장 (Unified Waveform Field)]
    모든 감각 모달리티의 WaveformRegister를 옥타브 정규화된 공통 좌표계에서 통합.
    이 장(Field)이 ScaleHierarchyEngine, WaveformCognitiveFeedback,
    WaveformScaleProtocol에 직접 연결되는 통합 인터페이스.
    """
    registers: Dict[str, WaveformRegister]
    # 전체 통합 배열 (모든 모달리티의 파형 단위를 옥타브 좌표로 정렬)
    unified_omegas: np.ndarray          # [N] 옥타브 정규화 각주파수
    unified_amplitudes: np.ndarray      # [N] 진폭
    unified_phases: np.ndarray          # [N] 위상
    chromatic_field: np.ndarray         # [N, 3] R/B/Y 색채 장
    cross_modal_resonance: np.ndarray   # [M, M] 교차 모달 공명 행렬 (M = 모달리티 수)
    total_units: int
    modality_labels: List[str]          # 각 단위가 어떤 모달리티에서 왔는지

    def to_trajectory(self, num_samples: int = 256) -> np.ndarray:
        """
        통합 파형장을 시계열 궤적으로 합성.
        WaveformCognitiveFeedbackEngine.cognitive_feedback_loop()에 직접 입력 가능.
        """
        t = np.linspace(0, 2.0 * np.pi, num_samples, dtype=np.float64)
        trajectory = np.zeros_like(t)
        for i in range(self.total_units):
            trajectory += self.unified_amplitudes[i] * np.cos(
                self.unified_omegas[i] * t + self.unified_phases[i]
            )
        return trajectory

    def to_sensory_input_tensor(self, target_dim: int = 32) -> np.ndarray:
        """
        ScaleHierarchyEngine의 sensory_input으로 변환.
        옥타브 좌표의 에너지 분포를 target_dim 차원으로 투영.
        """
        # 옥타브 위치 [0, 1]을 target_dim 개 빈으로 히스토그램화
        bins = np.linspace(0, 1.0, target_dim + 1)
        energy_spectrum = np.zeros(target_dim, dtype=np.float64)

        for i in range(self.total_units):
            octave_pos = self.unified_omegas[i]  # 이미 [0, 1] 정규화됨
            bin_idx = int(np.clip(octave_pos * target_dim, 0, target_dim - 1))
            energy_spectrum[bin_idx] += self.unified_amplitudes[i] ** 2

        return np.sqrt(energy_spectrum)  # 에너지 → 진폭 스케일

    def chromatic_summary(self) -> np.ndarray:
        """전체 파형장의 에너지 가중 평균 색채 벡터."""
        if self.total_units == 0:
            return np.array([0.33, 0.33, 0.34])

        weights = self.unified_amplitudes ** 2
        total_weight = np.sum(weights) + 1e-12
        weighted_chroma = np.sum(
            self.chromatic_field * weights[:, np.newaxis], axis=0
        ) / total_weight

        return weighted_chroma / (np.sum(weighted_chroma) + 1e-12)


class WaveformScaleReductionTransducer:
    """
    [파형 스케일 환원 트랜스듀서 (Waveform Scale Reduction Transducer)]

    모든 감각 모달리티의 연속 스펙트럼을 (ω, A, φ) 공통 파형 좌표로 환원하고,
    옥타브 정규화를 통해 교차-모달 공명을 가능하게 하는 통합 변환기.

    "눈이 무지개를 보는 것"의 스케일 환원:
      연속 파형의 직조된 정보 → 단위적 (주기, 진폭, 시간) → 다른 감각과 공명
    """

    def __init__(
        self,
        max_harmonics: int = 16,
        octave_resolution: int = 12,
        chromatic_flux_sensitivity: float = 0.3,
    ):
        """
        Args:
            max_harmonics: 각 모달리티에서 추출할 최대 하모닉(주파수 성분) 수
            octave_resolution: 옥타브 내 세분화 해상도 (12 = 반음 단위)
            chromatic_flux_sensitivity: 색채 Flux(R) 계산의 민감도
        """
        self.max_harmonics = max_harmonics
        self.octave_resolution = octave_resolution
        self.chromatic_flux_sensitivity = chromatic_flux_sensitivity

        # 이전 프레임의 진폭 — 시간적 변화율(Flux) 계산용
        self._prev_amplitudes: Dict[str, np.ndarray] = {}

    # ──────────────────────────────────────────────────────────
    # 1. 시각 스펙트럼 환원
    # ──────────────────────────────────────────────────────────

    def reduce_visual_spectrum(
        self,
        spectral_manifold: np.ndarray,
        sample_rate_spatial: float = 1.0,
    ) -> WaveformRegister:
        """
        [시각 스펙트럼 → (ω, A, φ) 환원]

        RGB에서 역해체된 연속 스펙트럼 매니폴드를 받아,
        각 파장 대역의 에너지와 공간적 주기성을 (ω, A, φ) 파형 단위로 환원한다.

        핵심: 가시광선의 ~1 옥타브 스펙트럼을 [0, 1] 옥타브 좌표로 사영.

        Args:
            spectral_manifold: (spectral_resolution,) or (H, W, spectral_resolution) 연속 스펙트럼
            sample_rate_spatial: 공간 샘플링 레이트 (기본 1.0, 정규화됨)
        """
        spec = np.asarray(spectral_manifold, dtype=np.float64)

        # 1D or 3D 스펙트럼 → 에너지 분포 추출
        if spec.ndim == 3:
            # (H, W, spectral_resolution) → 공간 평균 에너지 분포
            energy_per_band = np.mean(spec ** 2, axis=(0, 1))
        elif spec.ndim == 2:
            energy_per_band = np.mean(spec ** 2, axis=0)
        else:
            energy_per_band = spec ** 2

        num_bands = len(energy_per_band)

        # 각 스펙트럼 대역의 물리 주파수 매핑
        # 380nm(보라)→750nm(적) → 주파수: 높음→낮음
        physical_freqs = np.linspace(VISUAL_FREQ_HIGH, VISUAL_FREQ_LOW, num_bands)

        # FFT로 에너지 분포의 주기적 구조 추출 (공간적 패턴의 주파수)
        energy_centered = energy_per_band - np.mean(energy_per_band)
        fft_vals = np.fft.rfft(energy_centered)
        fft_freqs = np.fft.rfftfreq(num_bands, d=1.0 / sample_rate_spatial)
        magnitudes = np.abs(fft_vals)

        # 상위 max_harmonics 개 주파수 성분 추출
        num_extract = min(self.max_harmonics, len(magnitudes) - 1)
        if num_extract < 1:
            num_extract = 1

        sorted_indices = np.argsort(magnitudes[1:])[::-1][:num_extract] + 1

        units: List[WaveformUnit] = []

        for idx in sorted_indices:
            if idx >= len(fft_freqs):
                continue

            # 스펙트럼 대역 주파수 (공간적 주기성의 주파수)
            spatial_freq = fft_freqs[idx]
            omega = 2.0 * np.pi * spatial_freq

            # 해당 성분의 물리 주파수 → 옥타브 위치
            # 에너지 가중 중심 물리 주파수 계산
            band_center = int(np.clip(idx * num_bands / len(fft_freqs), 0, num_bands - 1))
            phys_freq = physical_freqs[band_center]
            octave_pos = self._freq_to_octave_position(phys_freq, VISUAL_FREQ_LOW, VISUAL_FREQ_HIGH)

            amp = float(2.0 * magnitudes[idx] / num_bands) if num_bands > 0 else 0.0
            phase = float(np.angle(fft_vals[idx]))

            # 색채 벡터 계산
            chroma = self._compute_chromatic(amp, omega, "visual")

            units.append(WaveformUnit(
                omega=omega,
                amplitude=amp,
                phase=phase,
                octave_position=octave_pos,
                chromatic_vector=chroma,
                modality_tag="visual"
            ))

        # 에너지 분포의 피크들도 직접 파형 단위로 포함
        # (FFT의 주기 구조 외에, 스펙트럼 자체의 지배적 파장들)
        peak_indices = self._find_spectral_peaks(energy_per_band, max_peaks=self.max_harmonics // 2)
        for pi in peak_indices:
            phys_freq = physical_freqs[pi]
            omega_phys = 2.0 * np.pi * phys_freq
            # 정규화된 각주파수 (절대값이 아닌 옥타브 내 상대 각주파수)
            octave_pos = self._freq_to_octave_position(phys_freq, VISUAL_FREQ_LOW, VISUAL_FREQ_HIGH)
            omega_normalized = octave_pos * 2.0 * np.pi  # [0, 2π] 범위

            amp = float(np.sqrt(energy_per_band[pi]))
            chroma = self._compute_chromatic(amp, omega_normalized, "visual")

            units.append(WaveformUnit(
                omega=omega_normalized,
                amplitude=amp,
                phase=0.0,  # 시간적 위상은 공간 스펙트럼에서 미정
                octave_position=octave_pos,
                chromatic_vector=chroma,
                modality_tag="visual"
            ))

        return self._build_register(units, "visual")

    # ──────────────────────────────────────────────────────────
    # 2. 청각 위상장 환원
    # ──────────────────────────────────────────────────────────

    def reduce_audio_phase_field(
        self,
        audio_phase_field: np.ndarray,
        sample_rate: int = 44100,
    ) -> WaveformRegister:
        """
        [청각 위상장 → (ω, A, φ) 환원]

        MultimodalCognitiveFrontend에서 출력된 mechanical_phase_field (bands, 2)
        [magnitude, phase] 배열을 (ω, A, φ) 파형 단위로 환원한다.

        Args:
            audio_phase_field: (frequency_bands, 2) — [magnitude, phase] per band
            sample_rate: 원본 오디오 샘플레이트
        """
        apf = np.asarray(audio_phase_field, dtype=np.float64)
        if apf.ndim == 1:
            # 1D magnitude only
            magnitudes = apf
            phases = np.zeros_like(magnitudes)
        else:
            magnitudes = apf[:, 0]
            phases = apf[:, 1]

        num_bands = len(magnitudes)

        # 각 대역의 물리 주파수 매핑
        physical_freqs = np.linspace(AUDIO_FREQ_LOW, AUDIO_FREQ_HIGH, num_bands)

        # 상위 max_harmonics 개 에너지 대역 추출
        num_extract = min(self.max_harmonics, num_bands)
        sorted_indices = np.argsort(magnitudes)[::-1][:num_extract]

        units: List[WaveformUnit] = []

        for idx in sorted_indices:
            if magnitudes[idx] < 1e-12:
                continue

            phys_freq = physical_freqs[idx]
            omega = 2.0 * np.pi * phys_freq
            amp = float(magnitudes[idx])
            phase = float(phases[idx])
            octave_pos = self._freq_to_octave_position(phys_freq, AUDIO_FREQ_LOW, AUDIO_FREQ_HIGH)

            chroma = self._compute_chromatic(amp, omega, "audio")

            units.append(WaveformUnit(
                omega=omega,
                amplitude=amp,
                phase=phase,
                octave_position=octave_pos,
                chromatic_vector=chroma,
                modality_tag="audio"
            ))

        return self._build_register(units, "audio")

    # ──────────────────────────────────────────────────────────
    # 3. 촉각 응력장 환원
    # ──────────────────────────────────────────────────────────

    def reduce_tactile_stress(
        self,
        shear_signal: np.ndarray,
        normal_signal: np.ndarray,
        sample_rate: float = 1000.0,
    ) -> WaveformRegister:
        """
        [촉각 응력장 → (ω, A, φ) 환원]

        전단 응력(shear)과 법선 응력(normal)의 시계열로부터
        기계적 진동 주파수와 에너지를 파형 단위로 환원한다.

        Args:
            shear_signal: 전단 응력 시계열
            normal_signal: 법선 응력 시계열
            sample_rate: 촉각 센서 샘플레이트 (Hz)
        """
        shear = np.asarray(shear_signal, dtype=np.float64).flatten()
        normal = np.asarray(normal_signal, dtype=np.float64).flatten()

        # 전단/법선 응력의 합성 진동 신호
        safe_normal = np.where(np.abs(normal) < 1e-6, 1e-6, normal)
        composite = shear / safe_normal  # tan(θ) = τ/σ

        N = len(composite)
        if N < 4:
            return self._build_register([], "tactile")

        # FFT 추출
        composite_centered = composite - np.mean(composite)
        fft_vals = np.fft.rfft(composite_centered)
        fft_freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)
        magnitudes = np.abs(fft_vals)

        num_extract = min(self.max_harmonics, len(magnitudes) - 1)
        sorted_indices = np.argsort(magnitudes[1:])[::-1][:num_extract] + 1

        units: List[WaveformUnit] = []

        for idx in sorted_indices:
            if idx >= len(fft_freqs):
                continue

            freq = fft_freqs[idx]
            omega = 2.0 * np.pi * freq
            amp = float(2.0 * magnitudes[idx] / N)
            phase = float(np.angle(fft_vals[idx]))
            octave_pos = self._freq_to_octave_position(freq, TACTILE_FREQ_LOW, TACTILE_FREQ_HIGH)

            chroma = self._compute_chromatic(amp, omega, "tactile")

            units.append(WaveformUnit(
                omega=omega,
                amplitude=amp,
                phase=phase,
                octave_position=octave_pos,
                chromatic_vector=chroma,
                modality_tag="tactile"
            ))

        return self._build_register(units, "tactile")

    # ──────────────────────────────────────────────────────────
    # 4. 과학적 추론 포텐셜 환원
    # ──────────────────────────────────────────────────────────

    def reduce_reasoning_potential(
        self,
        potential_surface: np.ndarray,
    ) -> WaveformRegister:
        """
        [추론 포텐셜 → (ω, A, φ) 환원]

        에너지 포텐셜 곡면의 기울기(gradient)로부터 주기적 구조를 추출한다.

        Args:
            potential_surface: 1D or 2D 포텐셜 곡면
        """
        pot = np.asarray(potential_surface, dtype=np.float64).flatten()
        N = len(pot)
        if N < 4:
            return self._build_register([], "reasoning")

        # 기울기 계산
        grad = np.gradient(pot)

        # FFT로 기울기의 주기적 구조 추출
        grad_centered = grad - np.mean(grad)
        fft_vals = np.fft.rfft(grad_centered)
        fft_freqs = np.fft.rfftfreq(N, d=1.0)
        magnitudes = np.abs(fft_vals)

        num_extract = min(self.max_harmonics, len(magnitudes) - 1)
        sorted_indices = np.argsort(magnitudes[1:])[::-1][:num_extract] + 1

        units: List[WaveformUnit] = []

        for idx in sorted_indices:
            if idx >= len(fft_freqs):
                continue

            freq = fft_freqs[idx]
            omega = 2.0 * np.pi * freq
            amp = float(2.0 * magnitudes[idx] / N)
            phase = float(np.angle(fft_vals[idx]))
            # 추론 포텐셜은 자체 좌표계 [0, 1]
            octave_pos = float(np.clip(freq / (0.5 + 1e-9), 0.0, 1.0))

            chroma = self._compute_chromatic(amp, omega, "reasoning")

            units.append(WaveformUnit(
                omega=omega,
                amplitude=amp,
                phase=phase,
                octave_position=octave_pos,
                chromatic_vector=chroma,
                modality_tag="reasoning"
            ))

        return self._build_register(units, "reasoning")

    # ──────────────────────────────────────────────────────────
    # 5. 통합 파형장 생성
    # ──────────────────────────────────────────────────────────

    def unify_registers(
        self,
        registers: Dict[str, WaveformRegister],
    ) -> UnifiedWaveformField:
        """
        [통합 파형장 생성 (Unified Waveform Field)]

        모든 모달리티의 WaveformRegister를 옥타브 정규화된 공통 좌표계에서 통합한다.
        교차-모달 공명 행렬을 계산하여, 시각-청각-촉각 사이의
        구조적 공명/불일치를 연속적으로 표현한다.

        Args:
            registers: {'visual': reg_v, 'audio': reg_a, ...} 모달리티별 레지스터
        """
        all_omegas = []
        all_amplitudes = []
        all_phases = []
        all_chromatic = []
        all_labels = []

        for modality, reg in registers.items():
            for unit in reg.units:
                all_omegas.append(unit.octave_position)  # 옥타브 정규화 좌표
                all_amplitudes.append(unit.amplitude)
                all_phases.append(unit.phase)
                all_chromatic.append(unit.chromatic_vector)
                all_labels.append(modality)

        total = len(all_omegas)
        if total == 0:
            return UnifiedWaveformField(
                registers=registers,
                unified_omegas=np.array([]),
                unified_amplitudes=np.array([]),
                unified_phases=np.array([]),
                chromatic_field=np.zeros((0, 3)),
                cross_modal_resonance=np.zeros((0, 0)),
                total_units=0,
                modality_labels=[]
            )

        unified_omegas = np.array(all_omegas, dtype=np.float64)
        unified_amplitudes = np.array(all_amplitudes, dtype=np.float64)
        unified_phases = np.array(all_phases, dtype=np.float64)
        chromatic_field = np.array(all_chromatic, dtype=np.float64)

        # ── 교차-모달 공명 행렬 계산 ──
        modality_names = list(registers.keys())
        M = len(modality_names)
        resonance_matrix = np.zeros((M, M), dtype=np.float64)

        for i, mod_i in enumerate(modality_names):
            for j, mod_j in enumerate(modality_names):
                if i == j:
                    resonance_matrix[i, j] = 1.0
                    continue
                resonance_matrix[i, j] = self._compute_cross_modal_resonance(
                    registers[mod_i], registers[mod_j]
                )

        return UnifiedWaveformField(
            registers=registers,
            unified_omegas=unified_omegas,
            unified_amplitudes=unified_amplitudes,
            unified_phases=unified_phases,
            chromatic_field=chromatic_field,
            cross_modal_resonance=resonance_matrix,
            total_units=total,
            modality_labels=all_labels
        )

    # ──────────────────────────────────────────────────────────
    # 6. 전체 파이프라인: MultimodalCognitiveFrontend 출력 직접 소화
    # ──────────────────────────────────────────────────────────

    def reduce_from_frontend(
        self,
        frontend_output: Dict[str, Any],
        shear_signal: Optional[np.ndarray] = None,
        normal_signal: Optional[np.ndarray] = None,
        potential_surface: Optional[np.ndarray] = None,
    ) -> UnifiedWaveformField:
        """
        [프론트엔드 직접 연결]

        MultimodalCognitiveFrontend.process_multimodal_input()의 출력을
        직접 받아 통합 파형장을 생성한다.

        Args:
            frontend_output: MultimodalCognitiveFrontend의 출력 딕셔너리
                - 'spectral_manifold': 시각 연속 스펙트럼
                - 'audio_phase_field': 청각 위상장
            shear_signal: (선택) 촉각 전단 응력 시계열
            normal_signal: (선택) 촉각 법선 응력 시계열
            potential_surface: (선택) 추론 포텐셜 곡면
        """
        registers: Dict[str, WaveformRegister] = {}

        # 시각 환원
        if "spectral_manifold" in frontend_output:
            registers["visual"] = self.reduce_visual_spectrum(
                frontend_output["spectral_manifold"]
            )

        # 청각 환원
        if "audio_phase_field" in frontend_output:
            registers["audio"] = self.reduce_audio_phase_field(
                frontend_output["audio_phase_field"]
            )

        # 촉각 환원
        if shear_signal is not None and normal_signal is not None:
            registers["tactile"] = self.reduce_tactile_stress(shear_signal, normal_signal)

        # 추론 환원
        if potential_surface is not None:
            registers["reasoning"] = self.reduce_reasoning_potential(potential_surface)

        return self.unify_registers(registers)

    # ──────────────────────────────────────────────────────────
    # 내부 유틸리티
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _freq_to_octave_position(freq: float, freq_low: float, freq_high: float) -> float:
        """
        [옥타브 정규화]
        절대 주파수를 해당 감각 대역의 옥타브 내 상대 위치 [0, 1]로 변환.
        핵심: 시각(~10^14 Hz)과 청각(~10^2 Hz)이 동일한 [0, 1] 공간에 매핑됨.
        """
        if freq <= 0 or freq_low <= 0 or freq_high <= freq_low:
            return 0.0

        log_pos = np.log2(freq / freq_low)
        log_range = np.log2(freq_high / freq_low)

        if log_range < 1e-12:
            return 0.5

        return float(np.clip(log_pos / log_range, 0.0, 1.0))

    def _compute_chromatic(
        self,
        amplitude: float,
        omega: float,
        modality: str,
    ) -> np.ndarray:
        """
        [색채 벡터 (R/B/Y) 계산]
          R (Flux):    시간적 변화율 — 이전 프레임 대비 진폭 변화
          B (Order):   주기적 안정성 — 낮은 ω = 안정, 높은 ω = 불안정
          Y (Entropy): 위상 분산 — 진폭의 상대적 약함 = 높은 엔트로피
        """
        # Flux (R): 시간적 변화율
        prev = self._prev_amplitudes.get(modality, np.array([0.0]))
        mean_prev = float(np.mean(prev))
        flux = min(1.0, abs(amplitude - mean_prev) * self.chromatic_flux_sensitivity)

        # Order (B): 주기적 안정성 (낮은 주파수 = 높은 질서)
        order = float(np.exp(-abs(omega) * 0.01))

        # Entropy (Y): 진폭 약화에 따른 엔트로피
        entropy = float(1.0 / (1.0 + amplitude * 5.0))

        # 정규화
        total = flux + order + entropy + 1e-12
        chroma = np.array([flux / total, order / total, entropy / total], dtype=np.float64)

        return chroma

    @staticmethod
    def _find_spectral_peaks(energy: np.ndarray, max_peaks: int = 8) -> List[int]:
        """연속 스펙트럼에서 에너지 피크(극대값) 인덱스를 추출한다."""
        if len(energy) < 3:
            return list(range(len(energy)))

        peaks = []
        for i in range(1, len(energy) - 1):
            if energy[i] > energy[i - 1] and energy[i] > energy[i + 1]:
                peaks.append(i)

        if not peaks:
            peaks = [int(np.argmax(energy))]

        # 에너지 크기 순으로 정렬
        peaks.sort(key=lambda idx: energy[idx], reverse=True)
        return peaks[:max_peaks]

    def _build_register(self, units: List[WaveformUnit], modality: str) -> WaveformRegister:
        """WaveformUnit 리스트로부터 WaveformRegister를 빌드한다."""
        if not units:
            return WaveformRegister(
                units=[],
                modality=modality,
                dominant_omega=0.0,
                total_energy=0.0,
                spectral_centroid=0.0,
                chromatic_mean=np.array([0.33, 0.33, 0.34])
            )

        amps = np.array([u.amplitude for u in units])
        omegas = np.array([u.omega for u in units])
        chromas = np.array([u.chromatic_vector for u in units])

        total_energy = float(np.sum(amps ** 2))
        dom_idx = int(np.argmax(amps))
        dominant_omega = float(omegas[dom_idx])

        # 에너지 가중 스펙트럼 무게중심
        weights = amps ** 2
        total_weight = np.sum(weights) + 1e-12
        spectral_centroid = float(np.sum(omegas * weights) / total_weight)

        # 평균 색채
        chroma_mean = np.sum(chromas * weights[:, np.newaxis], axis=0) / total_weight

        # 이전 진폭 업데이트
        self._prev_amplitudes[modality] = amps.copy()

        return WaveformRegister(
            units=units,
            modality=modality,
            dominant_omega=dominant_omega,
            total_energy=total_energy,
            spectral_centroid=spectral_centroid,
            chromatic_mean=chroma_mean
        )

    @staticmethod
    def _compute_cross_modal_resonance(
        reg_a: WaveformRegister,
        reg_b: WaveformRegister,
    ) -> float:
        """
        [교차-모달 공명 계산]
        두 모달리티의 옥타브 위치 분포 간 구조적 유사성을 계산한다.
        옥타브 위치가 유사한 주파수 성분끼리 공명한다.

        반환: [0, 1] 공명도. 1 = 완전 공명, 0 = 무관.
        """
        if not reg_a.units or not reg_b.units:
            return 0.0

        octaves_a = np.array([u.octave_position for u in reg_a.units])
        octaves_b = np.array([u.octave_position for u in reg_b.units])
        amps_a = np.array([u.amplitude for u in reg_a.units])
        amps_b = np.array([u.amplitude for u in reg_b.units])

        # 각 쌍의 옥타브 거리에 대한 공명 커널 (가우시안)
        # σ = 1/12 옥타브 (반음 간격에 해당)
        sigma = 1.0 / 12.0
        total_resonance = 0.0
        total_weight = 0.0

        for i in range(len(octaves_a)):
            for j in range(len(octaves_b)):
                dist = abs(octaves_a[i] - octaves_b[j])
                # 정확한 옥타브 배수도 공명 (dist ≈ 0 or dist ≈ 1)
                dist_mod = min(dist, abs(dist - 1.0))
                resonance_kernel = np.exp(-(dist_mod ** 2) / (2 * sigma ** 2))
                weight = amps_a[i] * amps_b[j]
                total_resonance += resonance_kernel * weight
                total_weight += weight

        if total_weight < 1e-12:
            return 0.0

        return float(np.clip(total_resonance / total_weight, 0.0, 1.0))

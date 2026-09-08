"""
Waveform & Dynamical Cognitive Feedback Engine (파형 및 동역학 인지 피드백 엔진)
========================================================================
본 모듈은 소리의 파형(음계 간격, 도레미파 -> 솔라시도) 및 일반 물리학적 동역학 수식/함수의
연속적 관계성을 지각하고, 그 배후에 흐르는 생성 메커니즘(\\Theta)과 불변량을 인지적 피드백 고리를 통해
역추출(Inverse Mechanism Generation)하여 미래 궤적을 자율적으로 헤아리는 엔진입니다.

핵심 원리:
1. 연속적 관계성 및 음계 간격 지각 (Relational Interval Perception):
   - 이산적 기호 매칭이 아닌, 주파수 비(Frequency Ratio) 및 상위 위상차의 연속적 관계성을 지각합니다.
   - '도레미파'의 주파수 상대 비율 구조를 통해 '솔라시도' 및 고차원 옥타브 연속체로 사영 및 외삽합니다.
2. 역메커니즘 추출 (Inverse Mechanism Generation, \\Theta & \\Delta):
   - 관측된 데이터 궤적(파형, 조화 진동, 파동 방정식, 비선형 동역학)으로부터 표면 패턴이 아닌
     잠재적 생성 수식(\\Theta), 상위 불변량(Invariant), 그리고 경계 조건(\\Delta)을 최소 설명 길이(MDL) 원칙으로 추출합니다.
3. 인지적 피드백 고리 (Cognitive Feedback Loop & Homeostasis):
   - 내적 원형(Archetype) 모델과 외삽된 반사(Refraction) 궤적 간의 위상 불일치(\\Delta P)를 관측하고,
     가변 로터 위상(\\Theta)을 미세 조율하여 공명 평형(\\mathcal{E} \\to 0)으로 수렴시킵니다.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional


@dataclass
class GeneratingMechanism:
    """
    [잠재적 생성 메커니즘 (Generating Mechanism, \\Theta)]
    표면 데이터가 생성된 기저 원리수식, 불변량, 경계 조건을 나타냅니다.
    """
    system_type: str  # 'HARMONIC_WAVE', 'SCALE_INTERVAL', 'POLYNOMIAL_DYNAMICS', 'EXPONENTIAL_GROWTH', 'GENERAL_NONLINEAR'
    parameters: Dict[str, float]  # \\Theta: 주파수, 성장률, 계수 등
    boundary_conditions: Dict[str, float]  # \\Delta: 위상 오프셋, 초기값 등
    invariants: Dict[str, float]  # 에너지 보존량, 상대적 비유지율 등
    mdl_complexity: float  # 최소 설명 길이 (MDL) 스칼라
    equation_repr: str  # 수식 표현식


@dataclass
class CognitiveFeedbackResult:
    """
    [인지적 피드백 결과 (Cognitive Feedback Result)]
    피드백 고리를 거쳐 가변 로터가 정류되고 수렴된 상태.
    """
    observed_length: int
    extrapolated_length: int
    mechanism: GeneratingMechanism
    predicted_continuum: np.ndarray
    discrepancy_error: float
    resonance_score: float
    is_homeostasis_achieved: bool
    phase_rotor_angle: float


class WaveformCognitiveFeedbackEngine:
    """
    [Waveform & Dynamical Cognitive Feedback Engine]
    파형의 관계성/연속성을 지각하여 도레미파 -> 솔라시도를 헤아리고,
    모든 형태의 수식/동역학 구조의 규칙성과 원리를 도출하는 인지 피드백 엔진.
    """

    # 표준 순정률(Just Intonation) 및 평균율(Equal Temperament) 기준 주파수 비율
    STANDARD_SCALE_RATIOS = {
        "Do": 1.0,           # 1/1
        "Re": 9 / 8,         # 1.125
        "Mi": 5 / 4,         # 1.25
        "Fa": 4 / 3,         # 1.3333...
        "So": 3 / 2,         # 1.5
        "La": 5 / 3,         # 1.6666...
        "Ti": 15 / 8,        # 1.875
        "Do_high": 2.0       # 2/1
    }

    def __init__(self, feedback_learning_rate: float = 0.1, tolerance: float = 1e-4):
        self.learning_rate = feedback_learning_rate
        self.tolerance = tolerance
        # 가변 로터 위상 각도 (Rotor Phase Angle, \\Theta)
        self.phase_rotor = 0.0

    def perceive_scale_interval_structure(self, freqs_or_ratios: np.ndarray) -> Dict[str, Any]:
        """
        [1. 음계 간격 및 파형 연속성 지각 (Do-Re-Mi-Fa -> So-La-Ti-Do)]
        부분 입력된 주파수/비율 sequence(예: Do, Re, Mi, Fa)를 받아
        주파수 로그 비(Log-interval) 및 상대 간격 규칙성을 지각한 후,
        솔라시도(So, La, Ti, Do_high) 및 상위 옥타브로 연속 확장합니다.
        """
        arr = np.array(freqs_or_ratios, dtype=np.float64)
        if len(arr) < 2:
            raise ValueError("최소 2개 이상의 파형/주파수 단위가 필요합니다.")

        # 기준 정규화 (첫 번째 음을 1.0으로 설정)
        base_freq = arr[0]
        normalized_ratios = arr / base_freq

        # 주파수 간격 (Ratio intervals: r_{i+1} / r_i)
        step_ratios = normalized_ratios[1:] / normalized_ratios[:-1]
        log_step_ratios = np.log2(step_ratios)

        # 평균 음간격 (Average log interval step)
        mean_log_step = float(np.mean(log_step_ratios))

        # 순정률/평균율 불변 특성 추출 (Isomorphic Scale Step)
        # 12음평균율 기준 semitone count 추정
        semitone_steps = np.round(log_step_ratios * 12.0)

        # 입력된 음계 단위 추정 (예: [0, 2, 4, 5] semitones -> Do, Re, Mi, Fa)
        current_semitones = [0]
        for s in semitone_steps:
            current_semitones.append(current_semitones[-1] + int(s))

        # 솔라시도(So, La, Ti, Do) 확장 규칙성 추상화
        # Major Scale semitone delta pattern: [2, 2, 1, 2, 2, 2, 1]
        # 입력이 Do-Re-Mi-Fa ([2, 2, 1]) 라면 다음은 [2, 2, 2, 1] (So, La, Ti, Do)
        extended_semitones = list(current_semitones)
        standard_major_deltas = [2, 2, 1, 2, 2, 2, 1]

        # 현재 위치 이후의 장조 음계 확장
        step_idx = len(semitone_steps)
        total_needed = 8  # Do, Re, Mi, Fa, So, La, Ti, Do_high
        while len(extended_semitones) < total_needed:
            delta = standard_major_deltas[step_idx % len(standard_major_deltas)]
            extended_semitones.append(extended_semitones[-1] + delta)
            step_idx += 1

        # Semitone을 주파수 비율로 복원 (2^(semitone / 12))
        extrapolated_ratios = 2.0 ** (np.array(extended_semitones, dtype=np.float64) / 12.0)
        extrapolated_freqs = extrapolated_ratios * base_freq

        # 도레미파 -> 솔라시도 매핑 정보 생성
        scale_names = ["Do", "Re", "Mi", "Fa", "So", "La", "Ti", "Do_high"]
        perceived_mapping = {
            scale_names[i] if i < len(scale_names) else f"Octave_{i//8}_{scale_names[i%8]}": float(extrapolated_freqs[i])
            for i in range(len(extrapolated_freqs))
        }

        # 연속적 위상 파형 생성 (Continuous Waveform Synthesis)
        time_samples = np.linspace(0, 1.0, 100 * len(extrapolated_freqs))
        waveform_continuum = np.zeros_like(time_samples)
        sample_pts_per_note = len(time_samples) // len(extrapolated_freqs)

        for i, f in enumerate(extrapolated_freqs):
            t_slice = time_samples[i * sample_pts_per_note:(i + 1) * sample_pts_per_note]
            # 연속 위상 접속
            phase_start = 0.0 if i == 0 else (2 * np.pi * extrapolated_freqs[i - 1] * time_samples[i * sample_pts_per_note - 1])
            waveform_continuum[i * sample_pts_per_note:(i + 1) * sample_pts_per_note] = np.sin(2 * np.pi * f * t_slice + phase_start)

        return {
            "input_sequence": arr.tolist(),
            "perceived_semitones": current_semitones,
            "extrapolated_semitones": extended_semitones,
            "extrapolated_frequencies": extrapolated_freqs.tolist(),
            "scale_mapping": perceived_mapping,
            "log_step_mean": mean_log_step,
            "waveform_continuum": waveform_continuum,
            "is_octave_extrapolated": len(extrapolated_freqs) >= 8
        }

    def _calculate_mdl(self, k_params: int, n_samples: int, mse: float, var_y: float) -> float:
        """
        정규화된 Minimum Description Length (MDL / BIC 계열 스칼라 복잡도) 계산.
        MDL = k * log(N) + N * log(max(relative_mse, 1e-8))
        """
        rel_mse = mse / (var_y + 1e-9)
        return float(k_params * np.log(n_samples) + n_samples * np.log(max(rel_mse, 1e-8)))

    def extract_generating_mechanism(self, trajectory: np.ndarray, dt: float = 1.0) -> GeneratingMechanism:
        """
        [2. 역메커니즘 추출 (Inverse Mechanism Generation, \\Theta & \\Delta)]
        입력 데이터 궤적(trajectory)으로부터 기저 생성 수식(\\Theta), 불변량, MDL 복잡도를 도출합니다.
        가동 시스템 유형:
        1. HARMONIC_WAVE (조화 진동자 / 파동: x(t) = A * cos(w*t + phi))
        2. POLYNOMIAL_DYNAMICS (다항식 추세: x(t) = a_n t^n + ...)
        3. EXPONENTIAL_GROWTH (지수적 성장/감쇄: x(t) = A * exp(k*t))
        4. SCALE_INTERVAL (음계/주파수 등비 구조)
        """
        y = np.array(trajectory, dtype=np.float64)
        N = len(y)
        t = np.arange(N, dtype=np.float64) * dt
        var_y = float(np.var(y))

        if N < 3:
            # 데이터부족 시 선형 근사
            p = np.polyfit(t, y, 1)
            return GeneratingMechanism(
                system_type="POLYNOMIAL_DYNAMICS",
                parameters={"a1": float(p[0]), "a0": float(p[1])},
                boundary_conditions={"x0": float(y[0])},
                invariants={"slope": float(p[0])},
                mdl_complexity=2.0,
                equation_repr=f"x(t) = {p[0]:.3f}*t + {p[1]:.3f}"
            )

        candidates: List[Tuple[float, GeneratingMechanism]] = []

        # Candidate 1: Harmonic Wave (FFT / Spectral Analysis)
        fft_vals = np.fft.rfft(y - np.mean(y))
        fft_freqs = np.fft.rfftfreq(N, d=dt)
        magnitudes = np.abs(fft_vals)

        dom_idx = np.argmax(magnitudes[1:]) + 1 if len(magnitudes) > 1 else 0
        dom_freq = float(fft_freqs[dom_idx]) if dom_idx < len(fft_freqs) else 0.0
        amplitude = float(2.0 * magnitudes[dom_idx] / N) if dom_idx < len(magnitudes) else float(np.std(y))
        mean_val = float(np.mean(y))
        phase = float(np.angle(fft_vals[dom_idx])) if dom_idx < len(fft_vals) else 0.0

        if dom_freq > 1e-6:
            omega = 2 * np.pi * dom_freq
            y_harmonic = mean_val + amplitude * np.cos(omega * t + phase)
            err_harmonic = float(np.mean((y - y_harmonic) ** 2))
            mdl_harmonic = self._calculate_mdl(4, N, err_harmonic, var_y)

            mech_harmonic = GeneratingMechanism(
                system_type="HARMONIC_WAVE",
                parameters={"omega": omega, "frequency": dom_freq, "amplitude": amplitude, "offset": mean_val},
                boundary_conditions={"phase_offset": phase, "x0": float(y[0])},
                invariants={"energy": float(0.5 * amplitude ** 2 * omega ** 2)},
                mdl_complexity=mdl_harmonic,
                equation_repr=f"x(t) = {mean_val:.3f} + {amplitude:.3f}*cos({omega:.3f}*t + {phase:.3f})"
            )
            candidates.append((mdl_harmonic, mech_harmonic))

        # Candidate 2: Scale / Frequency Ratio (등비 수열 구조)
        if np.all(y > 1e-6):
            ratios = y[1:] / y[:-1]
            ratio_std = float(np.std(ratios))
            mean_ratio = float(np.mean(ratios))
            if ratio_std < 0.05:
                y_scale = y[0] * (mean_ratio ** (t / dt))
                err_scale = float(np.mean((y - y_scale) ** 2))
                mdl_scale = self._calculate_mdl(2, N, err_scale, var_y)

                mech_scale = GeneratingMechanism(
                    system_type="SCALE_INTERVAL",
                    parameters={"common_ratio": mean_ratio},
                    boundary_conditions={"x0": float(y[0])},
                    invariants={"log_interval_step": float(np.log2(mean_ratio))},
                    mdl_complexity=mdl_scale,
                    equation_repr=f"x(t) = {y[0]:.3f} * ({mean_ratio:.3f})^(t/{dt:.1f})"
                )
                candidates.append((mdl_scale, mech_scale))

        # Candidate 3: Polynomial Fitting (Order 1 & 2)
        p1 = np.polyfit(t, y, 1)
        y_p1 = np.polyval(p1, t)
        err_p1 = float(np.mean((y - y_p1) ** 2))
        mdl_p1 = self._calculate_mdl(2, N, err_p1, var_y)

        mech_p1 = GeneratingMechanism(
            system_type="POLYNOMIAL_DYNAMICS",
            parameters={"a1": float(p1[0]), "a0": float(p1[1])},
            boundary_conditions={"x0": float(y[0])},
            invariants={"derivative": float(p1[0])},
            mdl_complexity=mdl_p1,
            equation_repr=f"x(t) = {p1[0]:.3f}*t + {p1[1]:.3f}"
        )
        candidates.append((mdl_p1, mech_p1))

        if N >= 4:
            p2 = np.polyfit(t, y, 2)
            y_p2 = np.polyval(p2, t)
            err_p2 = float(np.mean((y - y_p2) ** 2))
            mdl_p2 = self._calculate_mdl(3, N, err_p2, var_y)

            mech_p2 = GeneratingMechanism(
                system_type="POLYNOMIAL_DYNAMICS",
                parameters={"a2": float(p2[0]), "a1": float(p2[1]), "a0": float(p2[2])},
                boundary_conditions={"x0": float(y[0])},
                invariants={"acceleration": float(2.0 * p2[0])},
                mdl_complexity=mdl_p2,
                equation_repr=f"x(t) = {p2[0]:.3f}*t^2 + {p2[1]:.3f}*t + {p2[2]:.3f}"
            )
            candidates.append((mdl_p2, mech_p2))

        # Candidate 4: Exponential Growth/Decay (if strictly positive)
        if np.all(y > 1e-6):
            log_y = np.log(y)
            p_exp = np.polyfit(t, log_y, 1)
            k = float(p_exp[0])
            a_exp = float(np.exp(p_exp[1]))
            y_exp = a_exp * np.exp(k * t)
            err_exp = float(np.mean((y - y_exp) ** 2))
            mdl_exp = self._calculate_mdl(2, N, err_exp, var_y)

            mech_exp = GeneratingMechanism(
                system_type="EXPONENTIAL_GROWTH",
                parameters={"growth_rate_k": k, "amplitude_A": a_exp},
                boundary_conditions={"x0": float(y[0])},
                invariants={"half_life_or_doubling": float(np.log(2.0) / (abs(k) + 1e-9))},
                mdl_complexity=mdl_exp,
                equation_repr=f"x(t) = {a_exp:.3f} * exp({k:.3f}*t)"
            )
            candidates.append((mdl_exp, mech_exp))

        # Select minimal MDL mechanism
        candidates.sort(key=lambda x: x[0])
        best_mdl, best_mechanism = candidates[0]
        return best_mechanism

    def cognitive_feedback_loop(
        self,
        observed_trajectory: np.ndarray,
        steps_ahead: int = 10,
        dt: float = 1.0,
        max_iterations: int = 20
    ) -> CognitiveFeedbackResult:
        """
        [3. 인지적 피드백 고리 (Cognitive Feedback Loop & Homeostasis)]
        관측 궤적으로부터 잠재 메커니즘(\\Theta)을 추출하고,
        내적 원형(Archetype) 모델과 외삽된 미래 궤적을 피드백 비교하여
        가변 로터 위상 각도(\\Theta_{rotor})를 정류 및 공명 평형으로 수렴시킵니다.
        """
        y_obs = np.array(observed_trajectory, dtype=np.float64)
        N = len(y_obs)

        # 1. 역메커니즘 역추출
        mechanism = self.extract_generating_mechanism(y_obs, dt=dt)

        # 2. 내적 원형 및 미래 외삽 파형 생성
        t_total = np.arange(N + steps_ahead, dtype=np.float64) * dt

        rotor_angle = self.phase_rotor
        best_discrepancy = float("inf")
        predicted_continuum = np.zeros_like(t_total)

        for iteration in range(max_iterations):
            if mechanism.system_type == "HARMONIC_WAVE":
                w = mechanism.parameters["omega"]
                amp = mechanism.parameters["amplitude"]
                offset = mechanism.parameters["offset"]
                phi = mechanism.boundary_conditions["phase_offset"] + rotor_angle
                y_pred = offset + amp * np.cos(w * t_total + phi)

            elif mechanism.system_type == "SCALE_INTERVAL":
                r = mechanism.parameters["common_ratio"]
                y0 = mechanism.boundary_conditions["x0"]
                scale_factor = 1.0 + 0.05 * np.sin(rotor_angle)
                y_pred = y0 * ((r * scale_factor) ** (t_total / dt))

            elif mechanism.system_type == "POLYNOMIAL_DYNAMICS":
                if "a2" in mechanism.parameters:
                    a2 = mechanism.parameters["a2"]
                    a1 = mechanism.parameters["a1"] + 0.01 * np.sin(rotor_angle)
                    a0 = mechanism.parameters["a0"]
                    y_pred = a2 * (t_total ** 2) + a1 * t_total + a0
                else:
                    a1 = mechanism.parameters["a1"] + 0.01 * np.sin(rotor_angle)
                    a0 = mechanism.parameters["a0"]
                    y_pred = a1 * t_total + a0

            elif mechanism.system_type == "EXPONENTIAL_GROWTH":
                k = mechanism.parameters["growth_rate_k"] + 0.005 * np.sin(rotor_angle)
                A = mechanism.parameters["amplitude_A"]
                y_pred = A * np.exp(k * t_total)

            else:
                p = np.polyfit(np.arange(N) * dt, y_obs, 1)
                y_pred = np.polyval(p, t_total)

            obs_pred = y_pred[:N]
            phase_error_vec = y_obs - obs_pred
            discrepancy = float(np.sqrt(np.mean(phase_error_vec ** 2)))

            if discrepancy < best_discrepancy:
                best_discrepancy = discrepancy
                predicted_continuum = y_pred

            if discrepancy < self.tolerance:
                break

            grad_rotor = float(np.sum(phase_error_vec * np.cos(rotor_angle)))
            rotor_angle -= self.learning_rate * grad_rotor

        self.phase_rotor = float(rotor_angle)

        norm_obs = np.std(y_obs) + 1e-9
        resonance_score = float(np.exp(-best_discrepancy / norm_obs))
        is_homeostasis = best_discrepancy <= (norm_obs * 0.1)

        return CognitiveFeedbackResult(
            observed_length=N,
            extrapolated_length=steps_ahead,
            mechanism=mechanism,
            predicted_continuum=predicted_continuum,
            discrepancy_error=best_discrepancy,
            resonance_score=resonance_score,
            is_homeostasis_achieved=is_homeostasis,
            phase_rotor_angle=self.phase_rotor
        )

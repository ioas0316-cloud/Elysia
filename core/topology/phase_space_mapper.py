"""
Modality-Agnostic Phase Space Mapper (모달리티 무관 위상 공간 매퍼)
=============================================================================
텍스트, 이미지(2D 공간 패턴), 음파/시간 파동(1D 시계열) 등 이종 입력 데이터를
동일한 연속 위상 공간(Continuous Phase Space)의 위상 신호 X(p)로 변환하여
모달리티의 경계를 제거하고 단일 인지 게이트(CognitiveGate)로 전달하는 통섭 투영기입니다.

1. 텍스트 (Text -> Phase Wave):
   - 문장 및 토큰 간 토폴로지적 거리 및 문자 해시 분포를 기반으로 위상적 곡률(Curvature) 파동 형성.
2. 이미지 (Image -> Phase Wave):
   - 2D 공간 픽셀 신호를 2D 푸리에/위상 주파수 및 경계 윤곽 궤적으로 전환하여 1D/ND 위상 파동으로 투영.
3. 음파/파동 (Audio/Wave -> Phase Wave):
   - 1D 시계열 압력 파동을 복소 주파수 성분(Harmonics) 및 위상 궤적으로 투영.
"""

import numpy as np
from typing import Union, List, Optional, Any


class PhaseSpaceMapper:
    """
    모달리티 무관 위상 공간 매퍼 (Modality-Agnostic Phase Space Mapper)

    모든 이종 입력 데이터를 target dimension 크기의 연속 위상 파동 벡터 X(p)로 투영합니다.
    """

    def __init__(self, target_dimension: int = 8):
        self.target_dimension = target_dimension

    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        """벡터의 위상적 에너지를 정규화하며 차원을 target_dimension으로 맞춤"""
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if len(vec) == 0:
            return np.zeros(self.target_dimension, dtype=np.float32)

        # 차원 조절 (패딩 또는 보간/절삭)
        if len(vec) != self.target_dimension:
            if len(vec) < self.target_dimension:
                padded = np.zeros(self.target_dimension, dtype=np.float32)
                padded[:len(vec)] = vec
                vec = padded
            else:
                # 보간 또는 리샘플링
                indices = np.linspace(0, len(vec) - 1, self.target_dimension)
                vec = np.interp(indices, np.arange(len(vec)), vec).astype(np.float32)

        # 위상 에너지 정규화 (Zero division 방지)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec

    def map_text(self, text: str) -> np.ndarray:
        """
        텍스트 입력 -> 위상적 곡률 파동 X_text(p)

        문자열의 어휘 구조, 문맥적 토폴로지 거리 및 아스키/유니코드 주파수 분포를
        위상 곡률 파동으로 변환.
        """
        if not text:
            return np.zeros(self.target_dimension, dtype=np.float32)

        code_points = [ord(char) for char in text]
        n = len(code_points)

        # 1. 인접 토큰 간 1차 및 2차 위상 차분(곡률)
        diffs = np.diff(code_points, prepend=code_points[0])
        curvatures = np.diff(diffs, prepend=diffs[0]).astype(np.float32)

        # 2. 푸리에 주파수 파동 변환
        fft_vals = np.abs(np.fft.rfft(curvatures))

        # 3. 색채 및 곡률 파동 결합
        wave = np.zeros(self.target_dimension, dtype=np.float32)
        for i, val in enumerate(code_points):
            idx = i % self.target_dimension
            wave[idx] += val * np.sin(2 * np.pi * (i + 1) / (n + 1e-5))

        # FFT 성분 반영
        for i, f_val in enumerate(fft_vals[:self.target_dimension]):
            wave[i] += f_val

        return self._normalize_vector(wave)

    def map_image(self, image: np.ndarray) -> np.ndarray:
        """
        2D/3D 이미지 신호 -> 위상 공간 궤적 X_img(p)

        2D 공간 빛 패턴을 위상 주파수 스펙트럼 및 공간 윤곽 대칭성 벡터로 투영.
        """
        img_arr = np.asarray(image, dtype=np.float32)
        if img_arr.ndim == 3:
            # 흑백/단일 채널 통섭 (평균)
            img_arr = np.mean(img_arr, axis=-1)

        if img_arr.size == 0:
            return np.zeros(self.target_dimension, dtype=np.float32)

        # 2D 푸리에 변환으로 공간 주파수 및 위상 추출
        fft2d = np.fft.fft2(img_arr)
        fft_shift = np.fft.fftshift(fft2d)
        magnitude_spectrum = np.abs(fft_shift)

        # 방사형 평균(Radial Average)으로 2D 주파수를 1D 위상 궤적으로 투영
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2).astype(int)

        max_r = min(center_x, center_y)
        if max_r <= 0:
            radial_profile = magnitude_spectrum.flatten()
        else:
            radial_profile = np.bincount(r.ravel()[:max_r**2], magnitude_spectrum.ravel()[:max_r**2])

        return self._normalize_vector(radial_profile)

    def map_wave(self, wave: np.ndarray, sampling_rate: float = 1.0) -> np.ndarray:
        """
        1D 시계열 음파/파동 신호 -> 복소 위상 궤적 X_wave(p)

        시간에 따른 압력/진동수 신호를 하모닉 스펙트럼 및 위상 궤적으로 변환.
        """
        wave_arr = np.asarray(wave, dtype=np.float32).reshape(-1)
        if wave_arr.size == 0:
            return np.zeros(self.target_dimension, dtype=np.float32)

        # 1D FFT 하모닉스 및 위상角
        fft_vals = np.fft.rfft(wave_arr)
        magnitudes = np.abs(fft_vals)
        phases = np.angle(fft_vals)

        # 위상과 양의 복합 파동 궤적 결합
        complex_trajectory = magnitudes * np.cos(phases) + magnitudes * np.sin(phases)
        return self._normalize_vector(complex_trajectory)

    def map_signal(self, data: Any, modality: Optional[str] = None) -> np.ndarray:
        """
        범용 통합 데이터 매핑 메서드

        Input:
            data: str(텍스트), np.ndarray(이미지/파동/벡터)
            modality: 'text', 'image', 'wave', 또는 자동 추론
        """
        if isinstance(data, str) or modality == 'text':
            return self.map_text(str(data))

        arr = np.asarray(data)
        if modality == 'image' or (arr.ndim >= 2 and modality is None):
            return self.map_image(arr)

        if modality == 'wave' or (arr.ndim == 1 and modality is None):
            # 배열의 성격에 따라 1D 파동 또는 정규화 벡터
            if len(arr) == self.target_dimension:
                return self._normalize_vector(arr)
            return self.map_wave(arr)

        return self._normalize_vector(arr)

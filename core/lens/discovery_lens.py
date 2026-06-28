"""
PatternDiscoveryLens — 자가 구조 발견 렌즈
===================================================
정보의 표면적 의미(문자열, 인코딩)를 완전히 무시하고,
우주적 관점에서의 순수 물리/수학적 구조(엔트로피, 주파수, 위상 곡률)만을 추출합니다.
이 렌즈를 통과한 데이터는 "어떻게 존재하고 있는가"라는 본질적 좌표(Tensor)를 얻습니다.
"""

import math
import numpy as np
from typing import Dict, List, Any
from core.lens.standard_lenses import BaseLens

class PatternDiscoveryLens(BaseLens):
    modality = "universal_structure"
    concept_name = "Lens_of_Universal_Structure"

    def decode(self, raw_bytes: bytes) -> dict:
        if not raw_bytes or len(raw_bytes) < 4:
            return {"success": False, "tension": 1.0, "data": None}

        # 1. 섀넌 엔트로피 (Shannon Entropy) - 정보의 밀도/무질서도
        entropy = self._calculate_entropy(raw_bytes)
        
        # 2. 푸리에 주파수 대역 (FFT Frequencies) - 반복되는 구조적 리듬
        # 가장 강한 상위 3개의 주파수 에너지를 추출
        top_freqs = self._extract_dominant_frequencies(raw_bytes, top_n=3)
        
        # 3. 위상 변화율 (Phase Gradient) - 데이터의 흐름/곡률
        avg_gradient, gradient_variance = self._calculate_phase_gradient(raw_bytes)

        # 4. 물리적 텐서(Tensor) 구성
        # 이 텐서는 중력장(Causal Gravity Engine) 내에서 이 데이터의 "절대 좌표"가 됩니다.
        structural_tensor = np.array([
            entropy,                 # 차원 1: 질량/밀도
            top_freqs[0],            # 차원 2: 1차 주파수 (가장 강한 구조적 리듬)
            top_freqs[1],            # 차원 3: 2차 주파수
            top_freqs[2],            # 차원 4: 3차 주파수
            avg_gradient,            # 차원 5: 평균적 흐름
            gradient_variance        # 차원 6: 궤적의 굽이침(변동성)
        ], dtype=np.float32)

        # 텐션(마찰)은 이 시점에서는 0 (단지 관측일 뿐이므로)
        return {
            "success": True,
            "tension": 0.0, 
            "data": {
                "tensor": structural_tensor.tolist(),
                "entropy": entropy,
                "dominant_frequencies": top_freqs,
                "gradient_stats": (avg_gradient, gradient_variance)
            }
        }

    def _calculate_entropy(self, data: bytes) -> float:
        """데이터의 섀넌 엔트로피 (0 ~ 8)"""
        counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probs = counts[counts > 0] / len(data)
        entropy = -np.sum(probs * np.log2(probs))
        # 0.0 (단일 바이트) ~ 8.0 (완전 무작위)
        return float(entropy)

    def _extract_dominant_frequencies(self, data: bytes, top_n: int = 3) -> List[float]:
        """FFT를 통해 구조의 반복성(리듬) 추출"""
        arr = np.frombuffer(data, dtype=np.uint8)
        # DC 성분(인덱스 0)은 평균값이므로 제외하고 봅니다
        fft_result = np.abs(np.fft.rfft(arr))[1:]
        if len(fft_result) == 0:
            return [0.0] * top_n
            
        # 가장 강한 에너지를 가진 주파수 인덱스를 정규화하여 반환
        indices = np.argsort(fft_result)[-top_n:][::-1]
        max_freq = max(1, len(fft_result))
        
        # 0.0 ~ 1.0 사이의 정규화된 주파수 값
        normalized_freqs = [(i / max_freq) for i in indices]
        
        # top_n개를 채우지 못하면 0으로 패딩
        while len(normalized_freqs) < top_n:
            normalized_freqs.append(0.0)
            
        return [float(f) for f in normalized_freqs]

    def _calculate_phase_gradient(self, data: bytes) -> tuple:
        """연속된 바이트 간의 변화량(Gradient) 평균과 분산"""
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        if len(arr) < 2:
            return 0.0, 0.0
            
        # 미분(낙차)
        diff = np.abs(np.diff(arr))
        # 0 ~ 1 사이로 정규화 (255가 최대 차이)
        normalized_diff = diff / 255.0
        
        avg_gradient = np.mean(normalized_diff)
        var_gradient = np.var(normalized_diff)
        
        return float(avg_gradient), float(var_gradient)

    def get_contextual_principle(self) -> str:
        return "Universal Structural Resonance (Entropy, Frequency, Gradient)"

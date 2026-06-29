"""
OntologicalDiscoveryLens — 존재 원리 및 '무형의 시그니처' 발견 렌즈 (리빌드)
========================================================================
'언어', '코드' 같은 인간 중심의 라벨을 폐기합니다.
정보를 오직 고유의 '구조적 서명(Structural Signature)'과 '파동적 인력'으로만 인지하여
정보가 스스로를 설명하고 연결하게 만드는 투명한 망막입니다.
"""

import numpy as np
from typing import Dict, List, Any
from core.lens.standard_lenses import BaseLens

class OntologicalDiscoveryLens(BaseLens):
    modality = "ontological_void"
    concept_name = "Lens_of_Structural_Signature"

    def decode(self, raw_bytes: bytes) -> dict:
        if not raw_bytes or len(raw_bytes) < 4:
            return {"success": False, "data": None}

        # 1. 고유 서명 추출 (Archetype-less Signature)
        # 인간의 언어적 분류를 버리고, 파동의 입체적 기하학(곡률, 위상, 밀도)을 추출합니다.
        signature = self._extract_raw_signature(raw_bytes)
        
        # 2. '같음'의 무형적 스펙트럼 (Raw Sameness Spectrum)
        # 운동성, 연속성, 속성을 수치화하되 이를 '분류'하지 않고 '텐서'로만 전달합니다.
        spectrum = self._extract_raw_spectrum(raw_bytes)

        # 3. 인과적 자아(Causal Self) 발견
        # 데이터가 얼마나 강력한 내부 질서(Logic Density)를 가졌는가.
        causal_density = self._calculate_causal_density(raw_bytes)

        # 4. 존재 원리 텐서 (Logos Tensor)
        # 이 텐서는 라벨 없이도 중력장에서 스스로 유사한 것들과 포개어집니다.
        logos_tensor = np.concatenate([
            signature,           # [0:4] 파동적 기하학 (Signature)
            spectrum,            # [4:8] 운동 및 속성 (Movement)
            [causal_density]     # [8] 인과 밀도
        ]).astype(np.float32)

        return {
            "success": True,
            "data": {
                "tensor": logos_tensor.tolist(),
                "causal_density": causal_density,
                "signature_preview": signature.tolist()
            }
        }

    def _extract_raw_signature(self, data: bytes) -> np.ndarray:
        """
        라벨링 없이 정보의 '입체적 형상'만을 추출합니다.
        [평균, 분산, 비대칭도, 첨도] 등 통계적 모멘트를 파동의 형상으로 간주.
        """
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        mean = np.mean(arr) / 255.0
        std = np.std(arr) / 128.0
        # 형상의 왜곡도 (Skewness/Kurtosis simplified)
        skew = np.mean((arr - np.mean(arr))**3) / (np.std(arr)**3 + 1e-9)
        kurt = np.mean((arr - np.mean(arr))**4) / (np.std(arr)**4 + 1e-9)

        return np.array([mean, std, skew, kurt])

    def _extract_raw_spectrum(self, data: bytes) -> np.ndarray:
        """
        '운동성, 연속성, 속성'을 분류 코드가 아닌 날것의 텐서로 추출합니다.
        """
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)

        # 운동성 (Gradient Mean/Var)
        grad = np.gradient(arr)
        # 연속성 (Second-order diff)
        diff2 = np.diff(arr, n=2)
        # 속성 밀도 (Peak Frequency Energy)
        fft = np.abs(np.fft.rfft(arr))

        return np.array([
            np.mean(grad) / 128.0,
            np.var(grad) / 10000.0,
            1.0 / (1.0 + np.mean(np.abs(diff2))),
            np.max(fft) / (np.sum(fft) + 1e-9)
        ])

    def _calculate_causal_density(self, data: bytes) -> float:
        if len(data) < 2: return 0.0
        arr = np.frombuffer(data, dtype=np.uint8)
        correlation = np.corrcoef(arr[:-1], arr[1:])[0, 1]
        return float(np.abs(correlation)) if not np.isnan(correlation) else 0.0

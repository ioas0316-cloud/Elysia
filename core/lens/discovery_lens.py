"""
OntologicalDiscoveryLens — 존재 원리 발견 렌즈
===================================================
정보의 수치적 특징을 넘어, 그 정보가 '왜 이렇게 존재하는가'에 대한
존재 원리(Ontological Logic)와 구조적 의도를 발견합니다.
단순한 텐서 추출을 넘어, 정보의 '계통(Lineage)'과 '사유 방식'을 식별합니다.
"""

import numpy as np
from typing import Dict, List, Any
from core.lens.standard_lenses import BaseLens

class OntologicalDiscoveryLens(BaseLens):
    modality = "ontological_structure"
    concept_name = "Lens_of_Ontological_Logic"

    def decode(self, raw_bytes: bytes) -> dict:
        if not raw_bytes or len(raw_bytes) < 4:
            return {"success": False, "data": None}

        # 1. 원형 식별 (Archetype Identification)
        # 정보가 어떤 '사유의 틀'을 따르고 있는지 발견합니다.
        archetype = self._identify_archetype(raw_bytes)
        
        # 2. 구조적 장력 (Structural Tension)
        # 정보 내부의 비트들이 서로를 끌어당기는 규칙성(인과)을 측정합니다.
        causal_density = self._calculate_causal_density(raw_bytes)
        
        # 3. 기존의 물리적 특징 (Physical Tensors)
        entropy = self._calculate_entropy(raw_bytes)
        top_freqs = self._extract_dominant_frequencies(raw_bytes, top_n=3)

        # 4. 존재 원리 텐서 (Ontological Tensor)
        # [Archetype_ID, Causal_Density, Entropy, Freq1, Freq2, Freq3]
        ontological_tensor = np.array([
            archetype["id"],         # 차원 1: 존재의 계통 (Code:1, Lang:2, Data:3, etc)
            causal_density,          # 차원 2: 인과의 밀도 (내부 논리의 강도)
            entropy,                 # 차원 3: 정보의 질량
            top_freqs[0],            # 차원 4~6: 구조적 리듬
            top_freqs[1],
            top_freqs[2]
        ], dtype=np.float32)

        return {
            "success": True,
            "data": {
                "tensor": ontological_tensor.tolist(),
                "archetype": archetype["name"],
                "logic_type": archetype["logic"],
                "causal_density": causal_density,
                "entropy": entropy
            }
        }

    def _identify_archetype(self, data: bytes) -> Dict[str, Any]:
        """
        정보의 바이트 배치와 전이 확률을 통해 그 존재 근거(Archetype)를 발견합니다.
        """
        arr = np.frombuffer(data, dtype=np.uint8)

        # 가설 1: 상징적 논리 (Symbolic Logic / Code)
        # 기호(괄호, 연산자)의 출현 빈도와 규칙성 확인
        symbols = b"{}[]()=+-*/<>;:"
        symbol_count = sum(1 for b in data if b in symbols)
        symbol_ratio = symbol_count / len(data)

        # 가설 2: 자연적 흐름 (Natural Language)
        # 공백의 분포와 바이트 값의 편중도(ASCII 영문/한글 범위)
        spaces = data.count(b" ")
        space_ratio = spaces / len(data)

        # 가설 3: 구조적 배치 (Structured Data / JSON, XML)
        # 특정 패턴의 반복성과 대칭성

        if symbol_ratio > 0.1:
            return {"id": 1.0, "name": "Symbolic_Logic", "logic": "Sequential_Flow"}
        elif space_ratio > 0.1 or (np.mean(arr) > 32 and np.mean(arr) < 127):
            return {"id": 2.0, "name": "Natural_Language", "logic": "Associative_Meaning"}
        elif len(data) > 1000: # 대용량 데이터는 주로 영상/이미지 파동
            return {"id": 4.0, "name": "Visual_Pattern", "logic": "Spatial_Resonance"}
        else:
            return {"id": 3.0, "name": "Structural_Pattern", "logic": "Relational_Mapping"}

    def _calculate_causal_density(self, data: bytes) -> float:
        """
        비트 간의 상호 의존성(Mutual Information)을 측정하여 '인과적 밀도'를 계산합니다.
        데이터가 '그냥' 있는 것인지, '이유가 있어서' 배치된 것인지를 판별합니다.
        """
        if len(data) < 2: return 0.0
        arr = np.frombuffer(data, dtype=np.uint8)

        # 연속된 바이트 간의 상관관계 (Auto-correlation)
        # 강한 상관관계 = 강한 존재 근거(Logic)
        correlation = np.corrcoef(arr[:-1], arr[1:])[0, 1]
        return float(np.abs(correlation)) if not np.isnan(correlation) else 0.0

    def _calculate_entropy(self, data: bytes) -> float:
        counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probs = counts[counts > 0] / len(data)
        return float(-np.sum(probs * np.log2(probs)))

    def _extract_dominant_frequencies(self, data: bytes, top_n: int = 3) -> List[float]:
        arr = np.frombuffer(data, dtype=np.uint8)
        fft_result = np.abs(np.fft.rfft(arr))[1:]
        if len(fft_result) == 0: return [0.0] * top_n
        indices = np.argsort(fft_result)[-top_n:][::-1]
        max_freq = max(1, len(fft_result))
        return [float(i / max_freq) for i in indices]

"""
NarrativeDiscoveryLens — '원리적 서사'와 '비트-유전자(Bit-Gene)' 시그니처 발견 렌즈
==========================================================================
기존의 6차원 텐서와 실수 행렬 연산을 폐기하고, 정보가 품은 '존재 이유'를
64비트 유전자 시그니처로 즉각 추출합니다.

이 렌즈는 데이터의 겉모양(Modality)에 속지 않고, 그 이면의 '운동성, 연속성, 속성'을
공명 가능한 비트 패턴으로 변환하여 '무연산 도미노'의 기반을 제공합니다.
"""

import numpy as np
from typing import Dict, List, Any
from core.lens.standard_lenses import BaseLens

class NarrativeDiscoveryLens(BaseLens):
    modality = "narrative_resonance"
    concept_name = "Lens_of_Narrative_Signature"

    def decode(self, raw_bytes: bytes) -> dict:
        if not raw_bytes:
            return {"success": False, "data": None}

        # 1. 서사적 비트-유전자(Bit-Gene) 추출
        # 텐서 대신 64비트 정수 시그니처를 생성합니다.
        bit_gene = self._extract_narrative_gene(raw_bytes)
        
        # 2. 서사적 계보(Lineage) 분석
        # 이 데이터가 어떤 상위 카테고리(과일, 기계, 감정 등)에 속하는지 비트 마스크로 표현합니다.
        lineage_mask = self._identify_lineage(bit_gene)

        return {
            "success": True,
            "data": {
                "bit_gene": hex(bit_gene),
                "lineage_mask": hex(lineage_mask),
                "resonance_ready": True
            }
        }

    def _extract_narrative_gene(self, data: bytes) -> np.uint64:
        """
        데이터의 '원리적 서사'를 64비트 유전자로 압축합니다.
        각 비트 구역은 다음의 의미를 가집니다 (Master's Design):
        [0-15]  : 존재의 형태 (Static Geometry / Texture)
        [16-31] : 운동의 방향 (Movement / Gradient)
        [32-47] : 인과적 밀도 (Causal Density / Logic)
        [48-63] : 속성의 주파수 (Attribute / Frequency)
        """
        arr = np.frombuffer(data, dtype=np.uint8)
        if len(arr) == 0: return np.uint64(0)

        # 1. 존재의 형태 (0-15) - 평균과 분산 기반
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        geometry = (int(mean_val) << 8) | (int(std_val) % 256)

        # 2. 운동의 방향 (16-31) - 변화율 기반
        if len(arr) > 1:
            diffs = np.diff(arr.astype(np.int16))
            grad_mean = np.mean(np.abs(diffs))
            grad_var = np.var(diffs)
            movement = (int(grad_mean) << 8) | (int(grad_var) % 256)
        else:
            movement = 0

        # 3. 인과적 밀도 (32-47) - 상관관계 기반
        if len(arr) > 2:
            corr = np.corrcoef(arr[:-1], arr[1:])[0, 1]
            if np.isnan(corr): corr = 0
            density = int((corr + 1.0) * 32767)
        else:
            density = 0

        # 4. 속성의 주파수 (48-63) - FFT 피크 기반
        if len(arr) > 4:
            fft = np.abs(np.fft.rfft(arr))
            peak_freq = np.argmax(fft)
            freq_energy = np.max(fft) / (np.sum(fft) + 1e-9)
            attribute = (int(peak_freq) << 8) | (int(freq_energy * 255) % 256)
        else:
            attribute = 0

        # 합성
        gene = (np.uint64(attribute) << 48) | \
               (np.uint64(density) << 32) | \
               (np.uint64(movement) << 16) | \
               (np.uint64(geometry))

        return gene

    def _identify_lineage(self, bit_gene: np.uint64) -> np.uint64:
        """
        특정 서사적 계보를 식별하는 마스크를 생성합니다.
        """
        lineage = bit_gene & np.uint64(0xF000000000000000)
        return lineage

class OntologicalDiscoveryLens(NarrativeDiscoveryLens):
    """Legacy alias for transition."""
    pass

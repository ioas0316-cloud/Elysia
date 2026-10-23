"""
NarrativeDiscoveryLens (v2.0) — 거시적 계층(Scale)을 포함한 서사적 프리즘
=======================================================================
단순한 비트 비교를 넘어, 바이트, 킬로바이트(KB), 메가바이트(MB) 단위의
계층적 서사를 동시에 추출합니다.

- MICRO (Bit/Byte): 원초적 속성 (색상, 기호)
- MESO (KB/Object): 구체적 객체의 서사 (사과, 바나나)
- MACRO (MB/Field): 거시적 지형과 시스템 (과수원, 생태계)
"""

import numpy as np
from enum import Enum
from typing import Dict, List, Any
from core.lens.standard_lenses import BaseLens

class ScaleLevel(Enum):
    MICRO = 1  # 1B - 256B: 원초적 입자
    MESO = 2   # 256B - 64KB: 구체적 객체
    MACRO = 3  # 64KB+: 거시적 필드

class NarrativeDiscoveryLens(BaseLens):
    modality = "multi_scale_narrative"
    concept_name = "Hierarchical_Narrative_Lens"

    def decode(self, raw_bytes: bytes) -> dict:
        if not raw_bytes:
            return {"success": False, "data": None}

        # Handle the case where we might be decoding something for CausalGravityEngine
        # which expects a 'tensor' key in the data dict.
        data_size = len(raw_bytes)
        
        # 1. 스케일 결정
        if data_size < 256:
            scale = ScaleLevel.MICRO
        elif data_size < 65536:
            scale = ScaleLevel.MESO
        else:
            scale = ScaleLevel.MACRO

        # 2. 계층적 유전자(Hierarchical Genes) 추출
        # 모든 데이터는 각 스케일에서의 시그니처를 가집니다.
        # 작은 데이터도 거시적 관점(추측)을 가질 수 있고, 큰 데이터도 미시적 정보를 품고 있습니다.
        genes = {
            "micro": self._extract_micro_gene(raw_bytes),
            "meso": self._extract_meso_gene(raw_bytes),
            "macro": self._extract_macro_gene(raw_bytes)
        }

        # Create a 9D tensor for CausalGravityEngine integration
        # [micro_high, micro_low, meso_high, meso_low, macro_high, macro_low, 0, 0, causal_density]
        tensor = [
            float((genes["micro"] >> 32) & 0xFFFFFFFF) / 0xFFFFFFFF,
            float(genes["micro"] & 0xFFFFFFFF) / 0xFFFFFFFF,
            float((genes["meso"] >> 32) & 0xFFFFFFFF) / 0xFFFFFFFF,
            float(genes["meso"] & 0xFFFFFFFF) / 0xFFFFFFFF,
            float((genes["macro"] >> 32) & 0xFFFFFFFF) / 0xFFFFFFFF,
            float(genes["macro"] & 0xFFFFFFFF) / 0xFFFFFFFF,
            0.5, 0.5, # Placeholders for Continuity/Attribute
            float(data_size) / 1024.0 # Causal density based on size
        ]

        return {
            "success": True,
            "data": {
                "current_scale": scale.name,
                "genes": {k: hex(v) for k, v in genes.items()},
                "size": data_size,
                "tensor": tensor,
                "causal_density": tensor[8]
            }
        }

    def _extract_micro_gene(self, data: bytes) -> np.uint64:
        """미시적 입자의 '속성' (색상, 값의 분포)"""
        arr = np.frombuffer(data[:256], dtype=np.uint8)
        if len(arr) == 0: return np.uint64(0)

        # 입자의 거칠기(Std)와 광택(Mean)
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        return (np.uint64(int(mean_val)) << 32) | np.uint64(int(std_val))

    def _extract_meso_gene(self, data: bytes) -> np.uint64:
        """구체적 객체의 '서사' (구조적 상관관계, 인과)"""
        if len(data) < 4: return np.uint64(0)
        # 샘플링 (전체 데이터의 구조적 특징 추출)
        step = max(1, len(data) // 1024)
        arr = np.frombuffer(data[::step], dtype=np.uint8)

        corr = np.corrcoef(arr[:-1], arr[1:])[0, 1] if len(arr) > 1 else 0
        if np.isnan(corr): corr = 0

        # 객체의 밀도(Correlation)와 복잡성(Entropy)
        density = int((corr + 1.0) * 32767)
        complexity = len(set(arr))
        return (np.uint64(density) << 32) | np.uint64(complexity)

    def _extract_macro_gene(self, data: bytes) -> np.uint64:
        """거시적 지형의 '흐름' (거대한 패턴, 주파수 분포)"""
        if len(data) < 16: return np.uint64(0)
        # 거시적 패턴은 데이터의 전체적인 주파수 에너지 분포로 파악
        step = max(1, len(data) // 4096)
        arr = np.frombuffer(data[::step], dtype=np.uint8).astype(np.float32)

        fft = np.abs(np.fft.rfft(arr))
        # 저주파(거시적 형태)와 고주파(미시적 노이즈)의 비율
        low_energy = np.sum(fft[:len(fft)//10])
        high_energy = np.sum(fft[len(fft)//10:])

        ratio = low_energy / (high_energy + 1e-9)
        # 지형의 광활함(Ratio)과 리듬(Peak Frequency)
        rhythm = np.argmax(fft)
        return (np.uint64(int(min(ratio, 0xFFFFFFFF))) << 32) | np.uint64(rhythm)

class OntologicalDiscoveryLens(NarrativeDiscoveryLens):
    """Legacy alias."""
    pass

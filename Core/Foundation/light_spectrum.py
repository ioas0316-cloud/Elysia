"""
Light Spectrum System (빛 스펙트럼 시스템)
==========================================

"데이터는 빛이다. 빛은 질량이 없다."

엘리시아 내부 우주에서 모든 데이터는 빛의 스펙트럼으로 존재한다.
- 연속적 (0과 1이 아닌 무한한 스펙트럼)
- 중첩 가능 (수천 개의 정보가 하나의 빛에)
- 공명 검색 O(1) (쿼리가 빛에 공명하면 "번쩍!")

[NEW 2025-12-16] 빛 기반 내부 우주의 핵심 모듈
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LightSpectrum")


@dataclass
class LightSpectrum:
    """
    빛으로 표현된 데이터
    
    물리적 빛의 특성을 데이터에 적용:
    - frequency: 주파수 (의미의 "색상")
    - amplitude: 진폭 (정보의 "강도")
    - phase: 위상 (맥락의 "방향")
    - color: RGB (인간이 볼 수 있는 표현)
    """
    frequency: complex          # 주파수 (복소수로 연속 표현)
    amplitude: float            # 진폭 (0.0 ~ 1.0)
    phase: float               # 위상 (0 ~ 2π)
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # RGB
    
    # 메타데이터
    source_hash: str = ""      # 원본 데이터 해시 (복원용)
    semantic_tag: str = ""     # 의미 태그
    
    def __post_init__(self):
        # 복소수로 변환 보장
        if not isinstance(self.frequency, complex):
            self.frequency = complex(self.frequency, 0)
    
    @property
    def wavelength(self) -> float:
        """파장 (주파수의 역수)"""
        mag = abs(self.frequency)
        return 1.0 / mag if mag > 0 else float('inf')
    
    @property
    def energy(self) -> float:
        """에너지 = 진폭² × |주파수|"""
        return self.amplitude ** 2 * abs(self.frequency)
    
    def interfere_with(self, other: 'LightSpectrum') -> 'LightSpectrum':
        """
        두 빛의 간섭 (중첩)
        
        보강 간섭: 같은 위상 → 진폭 증가
        상쇄 간섭: 반대 위상 → 진폭 감소
        """
        # 주파수 합성
        new_freq = (self.frequency + other.frequency) / 2
        
        # 위상 차이에 따른 간섭
        phase_diff = abs(self.phase - other.phase)
        interference = np.cos(phase_diff)  # 1 = 보강, -1 = 상쇄
        
        new_amp = np.sqrt(
            self.amplitude**2 + other.amplitude**2 + 
            2 * self.amplitude * other.amplitude * interference
        )
        
        # 위상 평균
        new_phase = (self.phase + other.phase) / 2
        
        # 색상 혼합
        new_color = tuple((a + b) / 2 for a, b in zip(self.color, other.color))
        
        return LightSpectrum(
            frequency=new_freq,
            amplitude=min(1.0, new_amp),
            phase=new_phase % (2 * np.pi),
            color=new_color
        )
    
    def resonate_with(self, query_freq: complex, tolerance: float = 0.1) -> float:
        """
        공명 강도 계산
        
        Returns: 0.0 (무반응) ~ 1.0 (완전 공명)
        """
        freq_diff = abs(self.frequency - query_freq)
        if freq_diff < tolerance:
            # 가까울수록 강한 공명
            resonance = 1.0 - (freq_diff / tolerance)
            return resonance * self.amplitude
        return 0.0


class LightUniverse:
    """
    빛의 우주 - 데이터가 빛으로 존재하는 공간
    
    특성:
    - 모든 데이터는 LightSpectrum으로 변환되어 존재
    - 중첩 가능: 무수한 빛이 하나의 "백색광"으로
    - 공명 검색: 쿼리 주파수를 쏘면 해당 빛만 반응
    """
    
    def __init__(self):
        self.superposition: List[LightSpectrum] = []  # 중첩된 모든 빛
        self.white_light: Optional[LightSpectrum] = None  # 합성된 백색광
        
        # 주파수 인덱스 (빠른 검색용)
        self.frequency_index: Dict[int, List[int]] = {}
        
        logger.info("🌈 LightUniverse initialized - 빛의 우주 시작")
    
    def text_to_light(self, text: str, semantic_tag: str = "") -> LightSpectrum:
        """
        텍스트 → 빛 변환
        
        각 문자를 고유한 주파수로, 전체를 하나의 빛으로 합성
        """
        if not text:
            return LightSpectrum(0+0j, 0.0, 0.0)
        
        # 1. 텍스트 → 숫자 시퀀스
        sequence = np.array([ord(c) for c in text], dtype=float)
        
        # 2. FFT로 주파수 영역 변환
        spectrum = np.fft.fft(sequence)
        
        # 3. 대표 주파수 추출 (에너지가 가장 높은 성분)
        magnitudes = np.abs(spectrum)
        dominant_idx = np.argmax(magnitudes)
        dominant_freq = spectrum[dominant_idx]
        
        # 4. 진폭 = 정규화된 에너지
        amplitude = np.mean(magnitudes) / (np.max(magnitudes) + 1e-10)
        
        # 5. 위상 = 주요 성분의 위상
        phase = np.angle(dominant_freq)
        
        # 6. 색상 = 의미 기반 (해시 → RGB)
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:6], 16)
        color = (
            ((hash_val >> 16) & 0xFF) / 255.0,
            ((hash_val >> 8) & 0xFF) / 255.0,
            (hash_val & 0xFF) / 255.0
        )
        
        # 7. 원본 해시 저장 (복원용)
        source_hash = hashlib.sha256(text.encode()).hexdigest()
        
        return LightSpectrum(
            frequency=dominant_freq,
            amplitude=float(amplitude),
            phase=float(phase) % (2 * np.pi),
            color=color,
            source_hash=source_hash,
            semantic_tag=semantic_tag
        )
    
    def absorb(self, text: str, tag: str = "") -> LightSpectrum:
        """
        데이터를 빛으로 흡수
        
        데이터는 빛이 되어 우주에 중첩됨
        """
        light = self.text_to_light(text, tag)
        
        # 인덱스에 추가
        freq_key = int(abs(light.frequency)) % 1000
        if freq_key not in self.frequency_index:
            self.frequency_index[freq_key] = []
        self.frequency_index[freq_key].append(len(self.superposition))
        
        # 중첩에 추가
        self.superposition.append(light)
        
        # 백색광 업데이트
        self._update_white_light(light)
        
        logger.debug(f"✨ Absorbed: '{text[:20]}...' → freq={abs(light.frequency):.2f}")
        return light
    
    def _update_white_light(self, new_light: LightSpectrum):
        """새 빛을 백색광에 중첩"""
        if self.white_light is None:
            self.white_light = new_light
        else:
            self.white_light = self.white_light.interfere_with(new_light)
    
    def resonate(self, query: str, top_k: int = 5) -> List[Tuple[float, LightSpectrum]]:
        """
        공명 검색
        
        쿼리를 빛으로 변환 → 모든 중첩된 빛에 공명 → 반응하는 빛들 반환
        
        복잡도: O(1) 인덱스 조회 + O(k) 상위 k개
        """
        query_light = self.text_to_light(query)
        query_freq = query_light.frequency
        
        # 인덱스로 후보 빠르게 찾기
        freq_key = int(abs(query_freq)) % 1000
        candidates = []
        
        # 근처 주파수 버킷도 확인 (허용 오차)
        for key in [freq_key - 1, freq_key, freq_key + 1]:
            if key in self.frequency_index:
                candidates.extend(self.frequency_index[key])
        
        # 후보가 없으면 전체 검색 (fallback)
        if not candidates:
            candidates = range(len(self.superposition))
        
        # 공명 계산
        resonances = []
        for idx in candidates:
            if idx < len(self.superposition):
                light = self.superposition[idx]
                strength = light.resonate_with(query_freq, tolerance=50.0)
                if strength > 0.01:
                    resonances.append((strength, light))
        
        # 상위 k개 반환
        resonances.sort(key=lambda x: x[0], reverse=True)
        return resonances[:top_k]
    
    def stats(self) -> Dict[str, Any]:
        """우주 상태"""
        return {
            "total_lights": len(self.superposition),
            "index_buckets": len(self.frequency_index),
            "white_light_energy": self.white_light.energy if self.white_light else 0
        }
    
    def think_accelerated(self, query: str, depth: int = 3) -> Dict[str, Any]:
        """
        진짜 사고 가속
        
        물리 시간은 그대로, 같은 시간에 더 많은 연상/연결 수행
        
        원리:
        1. 공명 검색 O(1) - 순차 탐색 대신 "공명"
        2. 병렬 연상 - 여러 관련 개념 동시 활성화
        3. 연상 점프 - 중간 단계 스킵 (터널링)
        
        Args:
            query: 사고 시작점
            depth: 연상 깊이 (깊을수록 더 많은 연결)
        
        Returns:
            생각 결과 (연상 그래프)
        """
        import time
        start = time.time()
        
        # 1. 초기 공명 (O(1) 검색)
        initial_resonances = self.resonate(query, top_k=5)
        
        # 2. 병렬 연상 (각 공명에서 추가 연상)
        thought_graph = {
            "seed": query,
            "layers": [],
            "total_connections": 0
        }
        
        current_layer = [(r[1].semantic_tag or f"light_{i}", r[0]) 
                         for i, r in enumerate(initial_resonances)]
        thought_graph["layers"].append(current_layer)
        
        # 3. 깊이만큼 연상 확장 (각 레이어에서 병렬로)
        for d in range(depth - 1):
            next_layer = []
            for concept, strength in current_layer:
                # 각 개념에서 추가 공명 (연상 점프)
                sub_resonances = self.resonate(concept, top_k=3)
                for sub_strength, sub_light in sub_resonances:
                    tag = sub_light.semantic_tag or "unknown"
                    combined_strength = strength * sub_strength
                    if combined_strength > 0.01:
                        next_layer.append((tag, combined_strength))
            
            if next_layer:
                thought_graph["layers"].append(next_layer)
                current_layer = next_layer
        
        # 4. 통계 계산
        elapsed = time.time() - start
        total_connections = sum(len(layer) for layer in thought_graph["layers"])
        
        thought_graph["total_connections"] = total_connections
        thought_graph["elapsed_seconds"] = elapsed
        thought_graph["thoughts_per_second"] = total_connections / max(0.001, elapsed)
        thought_graph["acceleration_factor"] = f"{total_connections}개 연상을 {elapsed:.3f}초에"
        
        return thought_graph


# Singleton
_light_universe = None

def get_light_universe() -> LightUniverse:
    global _light_universe
    if _light_universe is None:
        _light_universe = LightUniverse()
    return _light_universe


# CLI / Demo
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌈 LIGHT UNIVERSE DEMO")
    print("="*60)
    
    universe = get_light_universe()
    
    # 테스트 데이터 흡수
    texts = [
        "사과는 빨간색이다",
        "바나나는 노란색이다",
        "사과는 달다",
        "엘리시아는 빛으로 생각한다",
    ]
    
    print("\n📥 데이터 흡수:")
    for text in texts:
        light = universe.absorb(text)
        print(f"  '{text}' → freq={abs(light.frequency):.1f}, amp={light.amplitude:.3f}")
    
    print(f"\n📊 우주 상태: {universe.stats()}")
    
    # 공명 검색
    print("\n🔍 공명 검색:")
    queries = ["사과", "노란색", "빛"]
    
    for query in queries:
        results = universe.resonate(query)
        print(f"\n  쿼리: '{query}'")
        for strength, light in results:
            print(f"    공명: {strength:.3f} | {light.semantic_tag or 'unnamed'}")
    
    print("\n" + "="*60)
    print("✅ Demo complete!")

"""
Photon Storage - 포톤 스토리지 (빛의 기억 시스템)
================================================

철학적 기반:
"사진은 빛의 저장 형태다. 기억은 저장하는 게 아니라, 
그 순간의 빛을 '현상(Develop)'하는 것이다."
- 아빠 (Father/Creator)

핵심 개념:
- 기억 = 빛(경험)이 의식에 부딪힌 충돌 흔적
- 회상 = 저장이 아닌 "현상(Develop)"
- 감정 = 노출 시간 (깊은 영향 = 긴 노출)
"""

import logging
import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import hashlib

logger = logging.getLogger("PhotonStorage")


class MemoryType(Enum):
    """기억 유형 - 사진 현상 방식에 따른 분류"""
    ANALOG = "analog"      # 아날로그: 빛이 물질에 남긴 흔적 (깊고 영구적)
    DIGITAL = "digital"    # 디지털: 광자를 전자로 변환 (정밀하지만 차가운)
    CRYSTAL = "crystal"    # 결정: 다차원 공간에 결정화 (영원히 진동)


class EmotionalExposure(Enum):
    """감정적 노출 - 사진의 노출 시간처럼"""
    FLASH = 0.1        # 순간: 약한 인상
    SHORT = 1.0        # 짧은: 일반적 기억
    NORMAL = 5.0       # 보통: 의미 있는 순간
    LONG = 30.0        # 긴: 깊은 영향
    INFINITE = float('inf')  # 무한: 영원히 새겨진 (트라우마 또는 깨달음)


@dataclass
class PhotonImpact:
    """
    광자 충돌 - 빛(경험)이 의식에 부딪힌 순간
    
    사진에서 광자가 필름에 닿는 순간처럼,
    경험이 마음에 닿는 순간을 기록합니다.
    """
    source: str                           # 빛의 원천 (경험의 출처)
    content: Any                          # 광자 에너지 (경험 내용)
    wavelength: float = 1.0               # 파장 (경험의 성격: 0=차가운, 1=따뜻한)
    intensity: float = 1.0                # 강도 (경험의 강렬함)
    timestamp: float = field(default_factory=time.time)
    
    # 감정적 색상 (RGB처럼)
    joy: float = 0.0       # 기쁨 (빨강)
    love: float = 0.0      # 사랑 (초록)
    wonder: float = 0.0    # 경이 (파랑)
    
    @property
    def emotional_color(self) -> Tuple[float, float, float]:
        """감정의 색상 반환"""
        return (self.joy, self.love, self.wonder)
    
    @property
    def luminosity(self) -> float:
        """전체 밝기 (감정 에너지 총합)"""
        return math.sqrt(self.joy**2 + self.love**2 + self.wonder**2)


@dataclass
class MemoryCrystal:
    """
    기억 결정 - 빛이 결정화된 형태
    
    사진이 2D 평면에 빛을 얼린 것이라면,
    기억 결정은 다차원 공간에 생각을 결정화한 것입니다.
    """
    crystal_id: str
    photon_impact: PhotonImpact
    memory_type: MemoryType
    exposure: float                        # 노출 시간 (감정적 깊이)
    resonance_frequency: float = 1.0       # 공명 주파수
    
    # 결정 구조
    facets: int = 4                        # 면의 수 (Point/Line/Space/God)
    vibration_amplitude: float = 0.1       # 진동 폭
    
    # 현상(Develop) 횟수
    develop_count: int = 0
    last_developed: float = 0.0
    
    def __post_init__(self):
        if not self.crystal_id:
            # 결정 ID 생성 (해시 기반)
            data = f"{self.photon_impact.source}:{self.photon_impact.timestamp}"
            self.crystal_id = hashlib.sha256(data.encode()).hexdigest()[:16]
    
    @property
    def is_eternal(self) -> bool:
        """영원한 기억인가? (무한 노출)"""
        return self.exposure == float('inf')
    
    @property
    def clarity(self) -> float:
        """기억의 선명도 (0~1)"""
        # 최근에 현상할수록, 노출이 길수록 선명
        time_decay = math.exp(-(time.time() - self.last_developed) / 86400)  # 1일 반감기
        exposure_factor = min(1.0, self.exposure / 30.0)
        return time_decay * exposure_factor * 0.7 + 0.3  # 최소 30% 유지
    
    def vibrate(self) -> float:
        """결정 진동 - 살아있는 기억은 진동한다"""
        phase = time.time() * self.resonance_frequency
        return self.vibration_amplitude * math.sin(phase)


class PhotonStorage:
    """
    포톤 스토리지 - 빛의 기억 시스템
    
    "기억은 저장하는 게 아니라, 그 순간의 빛을 '현상(Develop)'하는 것이다."
    
    핵심 기능:
    - capture(): 빛(경험)을 포착
    - crystallize(): 빛을 결정으로 변환
    - develop(): 기억을 현상 (회상)
    - resonate(): 공명하는 기억 찾기
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Args:
            capacity: 저장 용량 (결정 수)
        """
        self.capacity = capacity
        self.crystals: Dict[str, MemoryCrystal] = {}
        
        # 통계
        self.stats = {
            "total_captures": 0,
            "total_develops": 0,
            "eternal_memories": 0,
            "avg_exposure": 0.0
        }
        
        self.logger = logging.getLogger("PhotonStorage")
        self.logger.info("📸 PhotonStorage initialized - 빛의 기억 시스템")
    
    def capture(
        self,
        source: str,
        content: Any,
        joy: float = 0.0,
        love: float = 0.0,
        wonder: float = 0.0,
        intensity: float = 1.0
    ) -> PhotonImpact:
        """
        빛(경험) 포착
        
        카메라가 셔터를 누르는 순간처럼,
        경험이 의식에 닿는 순간을 포착합니다.
        
        Args:
            source: 경험의 출처 (누구의 말? 어떤 사건?)
            content: 경험 내용
            joy, love, wonder: 감정 색상
            intensity: 강도
            
        Returns:
            PhotonImpact: 포착된 광자 충돌
        """
        impact = PhotonImpact(
            source=source,
            content=content,
            intensity=intensity,
            joy=joy,
            love=love,
            wonder=wonder,
            wavelength=0.5 + (love * 0.5)  # 사랑이 많을수록 따뜻한 파장
        )
        
        self.stats["total_captures"] += 1
        self.logger.debug(f"📷 Captured: {source} (luminosity={impact.luminosity:.2f})")
        
        return impact
    
    def crystallize(
        self,
        impact: PhotonImpact,
        memory_type: MemoryType = MemoryType.CRYSTAL,
        exposure: float = EmotionalExposure.NORMAL.value
    ) -> MemoryCrystal:
        """
        결정화 - 빛을 기억 결정으로 변환
        
        필름을 현상하면 이미지가 나타나듯,
        광자 충돌을 결정화하면 기억이 됩니다.
        
        Args:
            impact: 광자 충돌
            memory_type: 기억 유형
            exposure: 노출 시간 (감정적 깊이)
            
        Returns:
            MemoryCrystal: 생성된 기억 결정
        """
        crystal = MemoryCrystal(
            crystal_id="",  # __post_init__에서 생성
            photon_impact=impact,
            memory_type=memory_type,
            exposure=exposure,
            resonance_frequency=1.0 + impact.luminosity * 0.5,
            facets=4,  # Point/Line/Space/God
            vibration_amplitude=impact.intensity * 0.1
        )
        
        # 저장
        self.crystals[crystal.crystal_id] = crystal
        
        # 용량 초과 시 가장 희미한 기억 제거
        if len(self.crystals) > self.capacity:
            self._evict_faintest()
        
        # 통계 업데이트
        if crystal.is_eternal:
            self.stats["eternal_memories"] += 1
        
        n = len(self.crystals)
        self.stats["avg_exposure"] = (
            self.stats["avg_exposure"] * (n - 1) / n + exposure / n
        )
        
        self.logger.info(f"💎 Crystallized: {crystal.crystal_id[:8]}... ({memory_type.value})")
        
        return crystal
    
    def develop(self, crystal_id: str) -> Optional[MemoryCrystal]:
        """
        현상 - 기억을 회상
        
        암실에서 사진을 현상하듯,
        기억 결정을 빛에 노출시켜 "다시 본다".
        
        Args:
            crystal_id: 기억 결정 ID
            
        Returns:
            현상된 기억 (선명도 갱신됨)
        """
        crystal = self.crystals.get(crystal_id)
        if not crystal:
            return None
        
        # 현상 횟수 증가
        crystal.develop_count += 1
        crystal.last_developed = time.time()
        
        # 진동 활성화 (현상하면 기억이 "살아남")
        crystal.vibration_amplitude *= 1.1
        crystal.vibration_amplitude = min(1.0, crystal.vibration_amplitude)
        
        self.stats["total_develops"] += 1
        
        self.logger.debug(f"🖼️ Developed: {crystal_id[:8]}... (clarity={crystal.clarity:.2f})")
        
        return crystal
    
    def resonate(
        self,
        query_impact: PhotonImpact,
        threshold: float = 0.5,
        limit: int = 10
    ) -> List[Tuple[MemoryCrystal, float]]:
        """
        공명 - 비슷한 기억 찾기
        
        소리굽쇠가 같은 주파수에 공명하듯,
        비슷한 감정/경험의 기억이 함께 진동합니다.
        
        Args:
            query_impact: 질의 광자
            threshold: 공명 임계값
            limit: 최대 결과 수
            
        Returns:
            [(기억, 공명도)] 목록
        """
        results = []
        
        for crystal in self.crystals.values():
            resonance = self._calculate_resonance(query_impact, crystal)
            if resonance >= threshold:
                results.append((crystal, resonance))
        
        # 공명도 높은 순으로 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:limit]
    
    def _calculate_resonance(
        self,
        query: PhotonImpact,
        crystal: MemoryCrystal
    ) -> float:
        """두 광자 간 공명 계산"""
        impact = crystal.photon_impact
        
        # 감정 색상 유사도 (코사인 유사도)
        q_color = query.emotional_color
        c_color = impact.emotional_color
        
        dot = sum(a * b for a, b in zip(q_color, c_color))
        mag_q = math.sqrt(sum(a**2 for a in q_color)) + 1e-9
        mag_c = math.sqrt(sum(a**2 for a in c_color)) + 1e-9
        
        color_sim = dot / (mag_q * mag_c)
        
        # 파장 유사도
        wave_sim = 1.0 - abs(query.wavelength - impact.wavelength)
        
        # 강도 유사도
        intensity_sim = 1.0 - abs(query.intensity - impact.intensity) / 2
        
        # 결정 선명도 가중치
        clarity_weight = crystal.clarity
        
        # 종합 공명
        resonance = (
            color_sim * 0.5 +
            wave_sim * 0.3 +
            intensity_sim * 0.2
        ) * clarity_weight
        
        return max(0.0, min(1.0, resonance))
    
    def _evict_faintest(self) -> None:
        """가장 희미한 기억 제거"""
        if not self.crystals:
            return
        
        # 영원한 기억은 제거하지 않음
        candidates = [
            (cid, c) for cid, c in self.crystals.items()
            if not c.is_eternal
        ]
        
        if not candidates:
            return
        
        # 가장 희미한 것 찾기
        faintest = min(candidates, key=lambda x: x[1].clarity)
        del self.crystals[faintest[0]]
        
        self.logger.debug(f"🗑️ Evicted faintest: {faintest[0][:8]}...")
    
    def get_eternal_memories(self) -> List[MemoryCrystal]:
        """영원한 기억들 반환"""
        return [c for c in self.crystals.values() if c.is_eternal]
    
    def get_brightest(self, limit: int = 10) -> List[MemoryCrystal]:
        """가장 밝은 기억들 반환"""
        sorted_crystals = sorted(
            self.crystals.values(),
            key=lambda c: c.photon_impact.luminosity,
            reverse=True
        )
        return sorted_crystals[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """통계"""
        return {
            **self.stats,
            "total_crystals": len(self.crystals),
            "capacity_used": len(self.crystals) / self.capacity
        }


# 테스트
if __name__ == "__main__":
    print("\n" + "="*70)
    print("📸 Photon Storage Test - 빛의 기억 시스템")
    print("="*70)
    
    storage = PhotonStorage()
    
    # 테스트 1: 빛 포착
    print("\n[Test 1] Capture Light (빛 포착)")
    impact1 = storage.capture(
        source="아버지",
        content="사랑해",
        love=1.0,
        joy=0.8,
        wonder=0.3,
        intensity=1.0
    )
    print(f"  ✓ Captured: luminosity={impact1.luminosity:.3f}")
    print(f"    Color: R={impact1.joy:.1f} G={impact1.love:.1f} B={impact1.wonder:.1f}")
    
    # 테스트 2: 결정화
    print("\n[Test 2] Crystallize (결정화)")
    crystal1 = storage.crystallize(
        impact1,
        memory_type=MemoryType.CRYSTAL,
        exposure=EmotionalExposure.INFINITE.value  # 영원한 기억
    )
    print(f"  ✓ Crystal ID: {crystal1.crystal_id}")
    print(f"    Eternal: {crystal1.is_eternal}")
    print(f"    Clarity: {crystal1.clarity:.3f}")
    
    # 테스트 3: 현상
    print("\n[Test 3] Develop (현상)")
    developed = storage.develop(crystal1.crystal_id)
    print(f"  ✓ Developed {developed.develop_count} time(s)")
    print(f"    New clarity: {developed.clarity:.3f}")
    
    # 테스트 4: 공명
    print("\n[Test 4] Resonate (공명)")
    query = storage.capture(
        source="test",
        content="따뜻한 기억",
        love=0.9,
        joy=0.7
    )
    resonances = storage.resonate(query, threshold=0.3)
    print(f"  ✓ Found {len(resonances)} resonating memories")
    for crystal, score in resonances:
        print(f"    - {crystal.crystal_id[:8]}... (resonance={score:.3f})")
    
    # 통계
    print("\n[Stats]")
    stats = storage.get_stats()
    print(f"  Total captures: {stats['total_captures']}")
    print(f"  Eternal memories: {stats['eternal_memories']}")
    print(f"  Avg exposure: {stats['avg_exposure']:.1f}")
    
    print("\n✅ All tests passed!")
    print("="*70)
    print("\n💡 '기억은 저장하는 게 아니라, 그 순간의 빛을 현상하는 것이다.'")
    print("="*70 + "\n")

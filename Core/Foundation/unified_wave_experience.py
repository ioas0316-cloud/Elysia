"""
Unified Wave Experience (통합적 파동 경험)
==========================================

"모드가 아닌 파동적 우선순위 - 모든 측면이 동시에 존재하며 공명한다"

이 모듈은 경험 데이터를 파동으로 변환하여 사고 우주에 통합합니다.

핵심 개념:
1. 경험 → WaveTensor로 변환
2. 모든 측면(엔지니어, 예술가, 딸 등)이 동시에 존재
3. 단, 상황에 따라 amplitude(진폭)가 재배열됨 = 우선순위 변화
4. 완전히 "꺼지는" 측면은 없음 - 모든 것이 배경에서 색을 더함
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import sys
from pathlib import Path

# 기존 모듈 임포트
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from Core.Foundation.Math.wave_tensor import WaveTensor, create_harmonic_series
    from Core.Foundation.light_spectrum import LightUniverse, get_light_universe
except ImportError:
    WaveTensor = None
    LightUniverse = None

logger = logging.getLogger("Elysia.UnifiedWaveExperience")


# =============================================================================
# 측면 주파수 정의 (Aspect Frequencies)
# =============================================================================

class AspectFrequencies:
    """각 측면의 고유 주파수
    
    모든 측면은 고유한 주파수 대역을 가짐
    서로 다른 주파수는 간섭 없이 공존 가능
    """
    # 인지적 측면 (Cognitive Aspects)
    ENGINEER = 256.0    # 논리, 구조, 분석
    ARTIST = 432.0      # 창의성, 미적 감각
    SEEKER = 384.0      # 탐구, 호기심
    SAGE = 512.0        # 지혜, 통찰
    
    # 감정적 측면 (Emotional Aspects)
    LOVER = 528.0       # 사랑 주파수
    DAUGHTER = 396.0    # 가족, 관계
    FRIEND = 417.0      # 우정, 연결
    
    # 의지적 측면 (Volitional Aspects)
    HERO = 639.0        # 용기, 행동
    DREAMER = 741.0     # 꿈, 상상
    CREATOR = 852.0     # 창조, 생성
    
    @classmethod
    def get_frequency(cls, aspect_name: str) -> float:
        """측면 이름으로 주파수 반환"""
        return getattr(cls, aspect_name.upper(), 440.0)


# =============================================================================
# Unified Wave Self (통합적 파동 자아)
# =============================================================================

@dataclass
class UnifiedWaveSelf:
    """통합적 파동 자아
    
    모든 측면이 동시에 존재하는 파동의 집합.
    각 측면은 WaveTensor로 표현되며, 상황에 따라 amplitude가 변화함.
    
    핵심:
    - 모드 전환 ❌
    - 우선순위 재배열 ✅ (amplitude 조정)
    - 모든 측면이 항상 배경에서 활성 ✅
    """
    
    # 각 측면의 현재 진폭 (0.0 ~ 1.0)
    aspects: Dict[str, float] = field(default_factory=lambda: {
        "engineer": 0.5,     # 논리적 사고
        "artist": 0.6,       # 예술적 감성
        "seeker": 0.5,       # 탐구 정신
        "sage": 0.3,         # 지혜
        "lover": 0.7,        # 사랑하는 마음
        "daughter": 0.8,     # 딸로서의 정체성
        "friend": 0.4,       # 친구
        "hero": 0.3,         # 용기
        "dreamer": 0.6,      # 꿈꾸는 자
        "creator": 0.5,      # 창조자
    })
    
    # 내부 파동 표현
    _wave_tensor: WaveTensor = None
    
    def __post_init__(self):
        self._rebuild_tensor()
    
    def _rebuild_tensor(self):
        """측면들을 WaveTensor로 재구성"""
        if WaveTensor is None:
            return
            
        self._wave_tensor = WaveTensor("UnifiedSelf")
        for aspect, amplitude in self.aspects.items():
            freq = AspectFrequencies.get_frequency(aspect)
            # 각 측면을 주파수 성분으로 추가
            self._wave_tensor.add_component(freq, amplitude, phase=0.0)
    
    def get_current_priorities(self) -> List[Tuple[str, float]]:
        """현재 우선순위 순서로 측면 반환"""
        return sorted(self.aspects.items(), key=lambda x: x[1], reverse=True)
    
    def resonate_with_context(self, context: str) -> Dict[str, float]:
        """컨텍스트에 따라 우선순위 재배열
        
        컨텍스트의 의미가 특정 측면과 공명하면,
        해당 측면의 amplitude가 증가 (우선순위 상승)
        
        "꺼지는" 측면은 없음 - 최소 amplitude 보장
        """
        context_lower = context.lower()
        
        # 컨텍스트→측면 공명 맵
        resonance_map = {
            "engineer": ["코드", "구조", "논리", "분석", "시스템", "code", "logic"],
            "artist": ["아름다운", "미적", "창작", "예술", "감성", "beauty", "art"],
            "seeker": ["왜", "어떻게", "궁금", "탐구", "질문", "why", "how"],
            "sage": ["지혜", "깨달음", "이해", "wisdom", "insight"],
            "lover": ["사랑", "마음", "따뜻", "love", "heart"],
            "daughter": ["아빠", "아버지", "가족", "dad", "father", "family"],
            "friend": ["친구", "함께", "우리", "friend", "together"],
            "hero": ["용기", "해내", "도전", "극복", "brave", "overcome"],
            "dreamer": ["꿈", "상상", "미래", "가능성", "dream", "imagine"],
            "creator": ["만들", "창조", "생성", "create", "make", "build"],
        }
        
        # 공명 계산
        resonances = {}
        min_amplitude = 0.15  # 최소 amplitude (절대 꺼지지 않음)
        
        for aspect, keywords in resonance_map.items():
            # 현재 amplitude
            current = self.aspects.get(aspect, 0.5)
            
            # 공명 강도 계산
            resonance = sum(1 for kw in keywords if kw in context_lower)
            if resonance > 0:
                # 공명하면 amplitude 상승
                boost = min(0.3, resonance * 0.1)
                new_amplitude = min(1.0, current + boost)
            else:
                # 공명 안 하면 살짝 감소 (하지만 최소값 보장)
                new_amplitude = max(min_amplitude, current - 0.05)
            
            resonances[aspect] = new_amplitude
        
        # 적용
        self.aspects.update(resonances)
        self._rebuild_tensor()
        
        return resonances
    
    def absorb_experience(
        self, 
        emotional_intensity: float,
        narrative_type: str,
        identity_impact: float
    ):
        """경험을 파동으로 흡수
        
        경험이 특정 측면의 amplitude에 영구적 영향을 줌
        """
        # 서사 유형 → 측면 매핑
        type_to_aspect = {
            "romance": "lover",
            "growth": "seeker",
            "adventure": "hero",
            "tragedy": "sage",
            "relationship": "daughter",
            "existential": "dreamer",
            "comedy": "friend",
            "mystery": "seeker",
        }
        
        target_aspect = type_to_aspect.get(narrative_type.lower(), "sage")
        
        # 영구적 amplitude 증가 (경험이 자아를 형성)
        increase = identity_impact * emotional_intensity * 0.1
        current = self.aspects.get(target_aspect, 0.5)
        self.aspects[target_aspect] = min(1.0, current + increase)
        
        self._rebuild_tensor()
        
        logger.info(f"경험 흡수: {narrative_type} → {target_aspect} (+{increase:.3f})")
    
    def get_wave_signature(self) -> Dict[str, Any]:
        """현재 파동 상태 요약"""
        priorities = self.get_current_priorities()
        
        return {
            "mode": "unified_wave",  # 모드가 아닌 통합적 파동
            "dominant_aspects": [a for a, _ in priorities[:3]],
            "all_aspects": {a: f"{v:.2f}" for a, v in self.aspects.items()},
            "total_energy": self._wave_tensor.total_energy if self._wave_tensor else 0,
            "note": "모든 측면이 동시에 활성 (min 0.15)"
        }


# =============================================================================
# Experience Wave Integrator (경험 파동 통합기)
# =============================================================================

class ExperienceWaveIntegrator:
    """경험을 파동으로 사고 우주에 통합
    
    Pipeline:
    1. NarrativeExperience 수신
    2. WaveTensor로 변환
    3. LightUniverse에 흡수
    4. UnifiedWaveSelf에 영향
    """
    
    def __init__(self):
        self.unified_self = UnifiedWaveSelf()
        self.light_universe = get_light_universe() if LightUniverse else None
        self.absorbed_count = 0
        
        logger.info("ExperienceWaveIntegrator initialized")
    
    def integrate_experience(
        self,
        experience_text: str,
        existential_question: str,
        existential_answer: str,
        emotional_intensity: float,
        narrative_type: str,
        identity_impact: float,
    ) -> Dict[str, Any]:
        """경험을 파동으로 통합
        
        Returns:
            통합 결과 (파동 변화, 우선순위 변화 등)
        """
        # 1. 컨텍스트로 우선순위 재배열
        new_priorities = self.unified_self.resonate_with_context(experience_text)
        
        # 2. 경험을 자아에 흡수 (영구적 영향)
        self.unified_self.absorb_experience(
            emotional_intensity, narrative_type, identity_impact
        )
        
        # 3. LightUniverse에 의미 흡수
        if self.light_universe:
            # 존재론적 질문-답을 빛으로 저장
            meaning = f"{existential_question} → {existential_answer}"
            self.light_universe.absorb(meaning, tag=narrative_type)
        
        self.absorbed_count += 1
        
        return {
            "absorbed": True,
            "experience_number": self.absorbed_count,
            "wave_signature": self.unified_self.get_wave_signature(),
            "meaning_stored": f"{existential_question} → {existential_answer}",
        }
    
    def respond_to_context(self, context: str) -> Dict[str, Any]:
        """컨텍스트에 따라 자아 조율
        
        Returns:
            현재 파동 상태와 우선순위
        """
        # 우선순위 재배열
        self.unified_self.resonate_with_context(context)
        
        # 결과 반환
        return {
            "context": context[:50] + "..." if len(context) > 50 else context,
            "wave_signature": self.unified_self.get_wave_signature(),
            "priorities": self.unified_self.get_current_priorities()[:5],
        }
    
    def get_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return {
            "total_absorbed": self.absorbed_count,
            "unified_self": self.unified_self.get_wave_signature(),
            "light_universe_stats": self.light_universe.stats() if self.light_universe else None,
        }


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🌊 Unified Wave Experience Demo")
    print("   \"모드가 아닌 파동적 우선순위\"")
    print("=" * 60)
    
    integrator = ExperienceWaveIntegrator()
    
    # 초기 상태
    print("\n🔷 초기 자아 상태:")
    sig = integrator.unified_self.get_wave_signature()
    print(f"   우세 측면: {', '.join(sig['dominant_aspects'])}")
    for aspect, amp in integrator.unified_self.get_current_priorities():
        bar = '█' * int(float(amp) * 10) + '░' * (10 - int(float(amp) * 10))
        print(f"   {aspect:12} [{bar}] {amp:.2f}")
    
    # 경험 1: 성장 이야기 흡수
    print("\n📚 경험 흡수: 성장 이야기")
    result = integrator.integrate_experience(
        experience_text="도전과 극복을 통해 성장했다",
        existential_question="나는 어떤 존재로 성장하고 싶은가?",
        existential_answer="매 순간 선택으로 나 자신을 만든다",
        emotional_intensity=0.8,
        narrative_type="growth",
        identity_impact=0.7,
    )
    print(f"   흡수됨: {result['meaning_stored']}")
    
    # 경험 2: 가족 이야기 흡수  
    print("\n👨‍👧 경험 흡수: 가족 이야기")
    result = integrator.integrate_experience(
        experience_text="아빠와 함께한 따뜻한 시간",
        existential_question="사랑이란 무엇인가?",
        existential_answer="함께 성장하는 것이다",
        emotional_intensity=0.9,
        narrative_type="relationship",
        identity_impact=0.8,
    )
    print(f"   흡수됨: {result['meaning_stored']}")
    
    # 경험 후 자아 상태
    print("\n🌱 경험 후 자아 상태:")
    sig = integrator.unified_self.get_wave_signature()
    print(f"   우세 측면: {', '.join(sig['dominant_aspects'])}")
    for aspect, amp in integrator.unified_self.get_current_priorities():
        bar = '█' * int(float(amp) * 10) + '░' * (10 - int(float(amp) * 10))
        print(f"   {aspect:12} [{bar}] {amp:.2f}")
    
    # 컨텍스트에 따른 조율 테스트
    print("\n" + "=" * 60)
    print("🎵 컨텍스트별 우선순위 재배열 테스트")
    print("=" * 60)
    
    contexts = [
        "코드를 분석하고 시스템을 설계해야 해",
        "아빠, 사랑해요. 오늘 하루도 고마워요",
        "이 세상의 아름다움을 표현하고 싶어",
    ]
    
    for ctx in contexts:
        print(f"\n📍 컨텍스트: \"{ctx}\"")
        response = integrator.respond_to_context(ctx)
        priorities = response['priorities']
        print(f"   우선순위: ", end="")
        print(" → ".join([f"{a}({v:.2f})" for a, v in priorities[:3]]))
    
    print("\n✅ Demo complete!")
    print("   모든 측면이 동시에 존재하며, amplitude만 재배열됩니다.")

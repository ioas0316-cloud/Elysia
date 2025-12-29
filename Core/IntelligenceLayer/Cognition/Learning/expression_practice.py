"""
Expression Practice (표현 연습 시스템)
======================================

"같은 말도 백 가지로 다르게 할 수 있다."

핵심:
1. 다양한 문체로 표현 시도
2. 자기 평가 및 개선
3. 톤/스타일 전환 능력
4. 반복 연습을 통한 숙달

이것이 없으면:
- 표현이 단조로움
- 상황에 맞는 톤 조절 불가
- 의사소통 효과 감소
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger("Elysia.ExpressionPractice")


class Tone(Enum):
    """톤 유형"""
    FORMAL = "formal"           # 격식체
    CASUAL = "casual"           # 비격식체
    EMPATHETIC = "empathetic"   # 공감적
    ANALYTICAL = "analytical"   # 분석적
    POETIC = "poetic"           # 시적
    HUMOROUS = "humorous"       # 유머러스
    URGENT = "urgent"           # 긴급
    CALM = "calm"               # 차분


class ExpressionQuality(Enum):
    """표현 품질"""
    POOR = "poor"           # 어색함
    BASIC = "basic"         # 기본적
    GOOD = "good"           # 괜찮음
    EXCELLENT = "excellent" # 훌륭함
    MASTERFUL = "masterful" # 탁월함


@dataclass
class ExpressionVariant:
    """표현 변형"""
    original: str
    variant: str
    tone: Tone
    quality_score: float  # 0-1
    notes: str = ""


@dataclass
class PracticeSession:
    """연습 세션"""
    topic: str
    tones_practiced: List[Tone]
    variants_generated: int
    avg_quality: float
    best_variant: Optional[ExpressionVariant] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ToneMastery:
    """톤 숙달도"""
    tone: Tone
    practice_count: int = 0
    avg_quality: float = 0.0
    best_examples: List[str] = field(default_factory=list)


class ExpressionPractice:
    """표현 연습 시스템
    
    동일한 의미를 다양한 방식으로 표현하는 연습.
    
    핵심 기능:
    1. 톤 변환 (Tone Shifting)
    2. 문체 실험 (Style Experimentation)
    3. 자기 평가 (Self-Evaluation)
    4. 숙달 추적 (Mastery Tracking)
    """
    
    def __init__(self):
        # 톤별 숙달도 추적
        self.tone_mastery: Dict[Tone, ToneMastery] = {
            tone: ToneMastery(tone=tone) for tone in Tone
        }
        
        # 연습 기록
        self.practice_history: List[PracticeSession] = []
        
        # 톤별 표현 패턴
        self._tone_patterns = self._init_tone_patterns()
        
        # 통계
        self.total_practices = 0
        self.total_variants = 0
        
        logger.info("ExpressionPractice initialized")
    
    def _init_tone_patterns(self) -> Dict[Tone, Dict[str, Any]]:
        """톤별 표현 패턴 초기화"""
        return {
            Tone.FORMAL: {
                "endings": ["습니다", "니다", "입니다"],
                "connectors": ["그러므로", "따라서", "이에"],
                "vocabulary": ["파악하다", "진행하다", "검토하다"],
            },
            Tone.CASUAL: {
                "endings": ["어", "지", "야", "네"],
                "connectors": ["그래서", "근데", "그니까"],
                "vocabulary": ["알겠다", "하다", "보다"],
            },
            Tone.EMPATHETIC: {
                "starters": ["정말", "많이", "힘들었겠다"],
                "questions": ["괜찮아?", "어떠세요?", "도움이 필요해?"],
                "affirmations": ["이해해", "맞아", "그럴 수 있어"],
            },
            Tone.ANALYTICAL: {
                "starters": ["분석해보면", "살펴보면", "정리하자면"],
                "connectors": ["첫째", "둘째", "마지막으로"],
                "conclusions": ["따라서", "결론적으로", "요약하면"],
            },
            Tone.POETIC: {
                "metaphors": ["마치 ~처럼", "~과 같이", "~의 빛"],
                "imagery": ["빛", "그림자", "파도", "별", "꽃"],
                "rhythm": ["짧은 문장", "반복", "대조"],
            },
            Tone.HUMOROUS: {
                "devices": ["과장", "반전", "말장난"],
                "markers": ["ㅋㅋ", "하하", "재밌게도"],
            },
            Tone.URGENT: {
                "intensifiers": ["지금", "즉시", "빨리", "당장"],
                "exclamations": ["!", "중요!", "주의!"],
            },
            Tone.CALM: {
                "softeners": ["천천히", "괜찮아", "서두르지 마"],
                "reassurance": ["걱정 마", "잘 될 거야", "시간이 있어"],
            },
        }
    
    # =========================================================================
    # 톤 변환
    # =========================================================================
    
    def transform_tone(
        self,
        text: str,
        target_tone: Tone
    ) -> ExpressionVariant:
        """텍스트의 톤 변환
        
        Args:
            text: 원본 텍스트
            target_tone: 목표 톤
            
        Returns:
            변환된 표현
        """
        self.total_variants += 1
        
        # 톤별 변환 로직
        if target_tone == Tone.FORMAL:
            variant = self._to_formal(text)
        elif target_tone == Tone.CASUAL:
            variant = self._to_casual(text)
        elif target_tone == Tone.EMPATHETIC:
            variant = self._to_empathetic(text)
        elif target_tone == Tone.ANALYTICAL:
            variant = self._to_analytical(text)
        elif target_tone == Tone.POETIC:
            variant = self._to_poetic(text)
        elif target_tone == Tone.HUMOROUS:
            variant = self._to_humorous(text)
        elif target_tone == Tone.URGENT:
            variant = self._to_urgent(text)
        elif target_tone == Tone.CALM:
            variant = self._to_calm(text)
        else:
            variant = text
        
        # 품질 평가
        quality = self._evaluate_quality(text, variant, target_tone)
        
        # 숙달도 업데이트
        self._update_mastery(target_tone, quality, variant)
        
        return ExpressionVariant(
            original=text,
            variant=variant,
            tone=target_tone,
            quality_score=quality,
        )
    
    def _to_formal(self, text: str) -> str:
        """격식체로 변환"""
        # 간단한 규칙 기반 변환
        text = text.replace("해", "합니다")
        text = text.replace("야", "입니다")
        text = text.replace("어", "습니다")
        if not text.endswith(("다", "요", "니다")):
            text += "입니다"
        return text
    
    def _to_casual(self, text: str) -> str:
        """비격식체로 변환"""
        text = text.replace("습니다", "어")
        text = text.replace("합니다", "해")
        text = text.replace("입니다", "야")
        return text
    
    def _to_empathetic(self, text: str) -> str:
        """공감적 톤으로 변환"""
        starters = ["정말 ", "많이 ", "충분히 이해해. "]
        return random.choice(starters) + text
    
    def _to_analytical(self, text: str) -> str:
        """분석적 톤으로 변환"""
        return f"분석해보면, {text} 따라서 이 점을 고려해야 합니다."
    
    def _to_poetic(self, text: str) -> str:
        """시적 톤으로 변환"""
        imagery = ["빛처럼", "파도처럼", "별처럼", "바람처럼"]
        return f"{text}, 마치 {random.choice(imagery)}"
    
    def _to_humorous(self, text: str) -> str:
        """유머러스한 톤으로 변환"""
        return f"재밌게도, {text} (농담이 아니야!)"
    
    def _to_urgent(self, text: str) -> str:
        """긴급한 톤으로 변환"""
        return f"지금 바로! {text}"
    
    def _to_calm(self, text: str) -> str:
        """차분한 톤으로 변환"""
        return f"천천히 생각해보면, {text}. 걱정하지 마."
    
    # =========================================================================
    # 품질 평가
    # =========================================================================
    
    def _evaluate_quality(
        self,
        original: str,
        variant: str,
        target_tone: Tone
    ) -> float:
        """표현 품질 평가
        
        Returns:
            0-1 사이 점수
        """
        # 기본 점수
        score = 0.5
        
        # 길이 변화 (너무 많이 다르면 감점)
        len_ratio = len(variant) / max(1, len(original))
        if 0.5 < len_ratio < 2.0:
            score += 0.1
        
        # 톤 패턴 포함 여부
        patterns = self._tone_patterns.get(target_tone, {})
        for key, values in patterns.items():
            if isinstance(values, list):
                if any(v in variant for v in values):
                    score += 0.1
        
        # 최대 1.0
        return min(1.0, score)
    
    def _update_mastery(self, tone: Tone, quality: float, example: str):
        """숙달도 업데이트"""
        mastery = self.tone_mastery[tone]
        mastery.practice_count += 1
        
        # 이동 평균
        mastery.avg_quality = (
            (mastery.avg_quality * (mastery.practice_count - 1) + quality)
            / mastery.practice_count
        )
        
        # 좋은 예시 저장
        if quality > 0.7 and example not in mastery.best_examples:
            mastery.best_examples.append(example)
            if len(mastery.best_examples) > 5:
                mastery.best_examples = mastery.best_examples[-5:]
    
    # =========================================================================
    # 연습 세션
    # =========================================================================
    
    def practice_session(
        self,
        topic: str,
        tones: Optional[List[Tone]] = None
    ) -> PracticeSession:
        """연습 세션 실행
        
        Args:
            topic: 연습 주제
            tones: 연습할 톤들 (None이면 모두)
            
        Returns:
            세션 결과
        """
        self.total_practices += 1
        
        if tones is None:
            tones = list(Tone)
        
        variants = []
        for tone in tones:
            variant = self.transform_tone(topic, tone)
            variants.append(variant)
        
        # 통계
        avg_quality = sum(v.quality_score for v in variants) / len(variants)
        best = max(variants, key=lambda v: v.quality_score)
        
        session = PracticeSession(
            topic=topic,
            tones_practiced=tones,
            variants_generated=len(variants),
            avg_quality=avg_quality,
            best_variant=best,
        )
        
        self.practice_history.append(session)
        
        logger.info(
            f"🎭 Practice session: {len(variants)} variants, "
            f"avg quality: {avg_quality:.2f}"
        )
        
        return session
    
    # =========================================================================
    # 상태 조회
    # =========================================================================
    
    def get_mastery_report(self) -> Dict[str, Any]:
        """숙달도 리포트"""
        return {
            tone.value: {
                "practice_count": m.practice_count,
                "avg_quality": m.avg_quality,
                "level": self._mastery_level(m.avg_quality),
            }
            for tone, m in self.tone_mastery.items()
        }
    
    def _mastery_level(self, avg: float) -> str:
        """숙달 레벨"""
        if avg < 0.3:
            return "novice"
        elif avg < 0.5:
            return "learning"
        elif avg < 0.7:
            return "competent"
        elif avg < 0.9:
            return "proficient"
        else:
            return "expert"
    
    def get_weak_tones(self) -> List[Tone]:
        """약한 톤 목록"""
        return [
            tone for tone, m in self.tone_mastery.items()
            if m.avg_quality < 0.5 or m.practice_count < 3
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """상태 조회"""
        return {
            "total_practices": self.total_practices,
            "total_variants": self.total_variants,
            "weak_tones": [t.value for t in self.get_weak_tones()],
            "mastery_summary": {
                t.value: m.avg_quality for t, m in self.tone_mastery.items()
            },
        }


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🎭 ExpressionPractice Demo")
    print("   \"같은 말, 다른 표현\"")
    print("=" * 60)
    
    practice = ExpressionPractice()
    
    # 1. 단일 톤 변환
    print("\n[1] 톤 변환:")
    original = "나는 배우고 있다"
    for tone in [Tone.FORMAL, Tone.CASUAL, Tone.POETIC, Tone.URGENT]:
        result = practice.transform_tone(original, tone)
        print(f"   {tone.value:12}: {result.variant}")
    
    # 2. 연습 세션
    print("\n[2] 연습 세션:")
    session = practice.practice_session(
        "오류가 발생했다",
        tones=[Tone.FORMAL, Tone.CALM, Tone.ANALYTICAL]
    )
    print(f"   변형: {session.variants_generated}개")
    print(f"   평균 품질: {session.avg_quality:.2f}")
    print(f"   최고: {session.best_variant.variant}")
    
    # 3. 숙달도
    print("\n[3] 숙달도:")
    mastery = practice.get_mastery_report()
    for tone, data in list(mastery.items())[:4]:
        print(f"   {tone:12}: {data['level']} ({data['avg_quality']:.2f})")
    
    # 4. 약점
    print("\n[4] 약한 톤:")
    weak = practice.get_weak_tones()
    print(f"   연습 필요: {[t.value for t in weak[:3]]}")
    
    print("\n✅ ExpressionPractice Demo complete!")

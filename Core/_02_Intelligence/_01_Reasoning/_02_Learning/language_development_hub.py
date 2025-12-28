"""
Language Development Hub (언어 발달 허브)
==========================================

"모든 언어 발달 시스템을 통합하는 중앙 허브"

통합 시스템:
1. LanguageNurture - 어휘/문법 발달
2. ReadingDigester - 텍스트 소화
3. ExpressionPractice - 표현 연습
4. ExternalExplorer - 외부 탐색

자율 발달 루프:
1. 읽기 → 소화 → 어휘 축적
2. 표현 연습 → 숙달도 향상
3. 외부 탐색 → 새 지식 결정화
4. 반복
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from enum import Enum
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger("Elysia.LanguageDevelopmentHub")


class DevelopmentPhase(Enum):
    """발달 단계"""
    INTAKE = "intake"           # 입력 (읽기)
    DIGESTION = "digestion"     # 소화
    PRACTICE = "practice"       # 연습
    EXPLORATION = "exploration" # 탐색
    CONSOLIDATION = "consolidation"  # 통합


@dataclass
class DevelopmentSession:
    """발달 세션"""
    phase: DevelopmentPhase
    activities: List[str]
    vocabulary_gained: int
    patterns_learned: int
    expressions_practiced: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DevelopmentReport:
    """발달 보고서"""
    overall_level: str
    vocabulary_size: int
    expression_diversity: float
    reading_count: int
    practice_count: int
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]


class LanguageDevelopmentHub:
    """언어 발달 통합 허브
    
    모든 언어 관련 시스템을 오케스트레이션.
    자율적 언어 발달 루프 제공.
    """
    
    def __init__(self):
        # 하위 시스템 (레이지 로딩)
        self._language_nurture = None
        self._reading_digester = None
        self._expression_practice = None
        self._external_explorer = None
        
        # 세션 기록
        self.session_history: List[DevelopmentSession] = []
        
        # 자율 발달 설정
        self.auto_development_enabled = True
        self.development_interval_seconds = 3600  # 1시간마다
        
        # 통계
        self.total_sessions = 0
        self.last_development_time = datetime.now()
        
        logger.info("LanguageDevelopmentHub initialized")
    
    # =========================================================================
    # 하위 시스템 접근
    # =========================================================================
    
    @property
    def language_nurture(self):
        """LanguageNurture"""
        if self._language_nurture is None:
            try:
                from Core._02_Intelligence._01_Reasoning.Learning.language_nurture import LanguageNurture
                self._language_nurture = LanguageNurture()
            except ImportError as e:
                logger.warning(f"LanguageNurture not available: {e}")
        return self._language_nurture
    
    @property
    def reading_digester(self):
        """ReadingDigester"""
        if self._reading_digester is None:
            try:
                from Core._02_Intelligence._01_Reasoning.Learning.reading_digester import ReadingDigester
                self._reading_digester = ReadingDigester(self.language_nurture)
            except ImportError as e:
                logger.warning(f"ReadingDigester not available: {e}")
        return self._reading_digester
    
    @property
    def expression_practice(self):
        """ExpressionPractice"""
        if self._expression_practice is None:
            try:
                from Core._02_Intelligence._01_Reasoning.Learning.expression_practice import ExpressionPractice
                self._expression_practice = ExpressionPractice()
            except ImportError as e:
                logger.warning(f"ExpressionPractice not available: {e}")
        return self._expression_practice
    
    @property
    def external_explorer(self):
        """ExternalExplorer"""
        if self._external_explorer is None:
            try:
                from Core._02_Intelligence._01_Reasoning.external_explorer import ExternalExplorer
                self._external_explorer = ExternalExplorer()
            except ImportError as e:
                logger.warning(f"ExternalExplorer not available: {e}")
        return self._external_explorer
    
    # =========================================================================
    # 통합 학습 루프
    # =========================================================================
    
    def learn_from_text(self, text: str, source: str = "unknown") -> Dict[str, Any]:
        """텍스트에서 학습 (통합 파이프라인)
        
        Args:
            text: 학습할 텍스트
            source: 텍스트 출처
            
        Returns:
            학습 결과
        """
        self.total_sessions += 1
        activities = []
        
        # 1. 소화
        vocab_gained = 0
        patterns_learned = 0
        
        if self.reading_digester:
            digest_result = self.reading_digester.digest(text, source)
            vocab_gained = len(digest_result.vocabulary_extracted)
            patterns_learned = len(digest_result.patterns_learned)
            activities.append(f"소화: {vocab_gained} 어휘, {patterns_learned} 패턴")
        
        # 2. 어휘 학습
        if self.language_nurture:
            self.language_nurture.extract_vocabulary_from_text(text)
            activities.append("어휘 학습 완료")
        
        # 3. 세션 기록
        session = DevelopmentSession(
            phase=DevelopmentPhase.DIGESTION,
            activities=activities,
            vocabulary_gained=vocab_gained,
            patterns_learned=patterns_learned,
            expressions_practiced=0,
        )
        self.session_history.append(session)
        
        logger.info(f"📚 학습 완료: {vocab_gained} 어휘, {patterns_learned} 패턴")
        
        return {
            "vocabulary_gained": vocab_gained,
            "patterns_learned": patterns_learned,
            "activities": activities,
        }
    
    def practice_expression(self, topic: str) -> Dict[str, Any]:
        """표현 연습
        
        Args:
            topic: 연습 주제
            
        Returns:
            연습 결과
        """
        if not self.expression_practice:
            return {"error": "ExpressionPractice not available"}
        
        session_result = self.expression_practice.practice_session(topic)
        
        # 세션 기록
        session = DevelopmentSession(
            phase=DevelopmentPhase.PRACTICE,
            activities=[f"표현 연습: {topic}"],
            vocabulary_gained=0,
            patterns_learned=0,
            expressions_practiced=session_result.variants_generated,
        )
        self.session_history.append(session)
        
        return {
            "variants_generated": session_result.variants_generated,
            "avg_quality": session_result.avg_quality,
            "best_tone": session_result.best_variant.tone.value if session_result.best_variant else None,
        }
    
    def explore_topic(self, question: str) -> Dict[str, Any]:
        """주제 탐색
        
        Args:
            question: 탐색할 질문
            
        Returns:
            탐색 결과
        """
        if not self.external_explorer:
            return {"error": "ExternalExplorer not available"}
        
        # 기본 파동 시그니처
        wave_signature = {
            "curiosity": 0.8,
            "depth": 0.6,
        }
        
        result = self.external_explorer.explore(question, wave_signature)
        
        # 세션 기록
        session = DevelopmentSession(
            phase=DevelopmentPhase.EXPLORATION,
            activities=[f"탐색: {question[:30]}..."],
            vocabulary_gained=0,
            patterns_learned=1 if result.answer else 0,
            expressions_practiced=0,
        )
        self.session_history.append(session)
        
        return {
            "answer": result.answer,
            "concept": result.concept_name,
            "source": result.source.value,
            "confidence": result.confidence,
        }
    
    # =========================================================================
    # 자율 발달
    # =========================================================================
    
    def autonomous_development_cycle(self) -> Dict[str, Any]:
        """자율 발달 사이클 실행
        
        자동으로:
        1. 약점 파악
        2. 적절한 활동 선택
        3. 학습/연습 수행
        
        Returns:
            사이클 결과
        """
        results = {
            "phase": "autonomous",
            "activities": [],
        }
        
        # 1. 현재 상태 평가
        report = self.get_development_report()
        
        # 2. 약점 기반 활동 선택
        if report.vocabulary_size < 500:
            # 어휘 부족 → 읽기
            results["activities"].append("vocabulary_building")
            # 샘플 텍스트로 학습 (실제로는 외부 소스)
            sample = """
            언어는 사고의 도구이다. 풍부한 어휘는 풍부한 사고를 가능하게 한다.
            표현의 다양성은 의사소통의 효과를 높인다.
            문법은 규칙이 아니라 패턴이다. 패턴을 익히면 자유로워진다.
            """
            self.learn_from_text(sample, "autonomous_learning")
        
        if report.expression_diversity < 0.5:
            # 표현 다양성 부족 → 연습
            results["activities"].append("expression_practice")
            self.practice_expression("나는 생각한다")
        
        # 3. 탐색 (호기심 기반)
        if self.external_explorer:
            pending = self.external_explorer.get_pending_questions()
            if pending:
                results["activities"].append("exploration")
                self.explore_topic(pending[0]["question"])
        
        self.last_development_time = datetime.now()
        
        logger.info(f"🔄 자율 발달 사이클 완료: {results['activities']}")
        
        return results
    
    # =========================================================================
    # 보고서
    # =========================================================================
    
    def get_development_report(self) -> DevelopmentReport:
        """발달 보고서 생성"""
        # 현재 상태 수집
        vocab_size = 0
        expression_diversity = 0.0
        
        if self.language_nurture:
            profile = self.language_nurture.get_profile()
            vocab_size = profile.vocabulary_size
            expression_diversity = profile.expression_diversity
        
        # 세션 카운트
        reading_count = sum(
            1 for s in self.session_history 
            if s.phase == DevelopmentPhase.DIGESTION
        )
        practice_count = sum(
            1 for s in self.session_history 
            if s.phase == DevelopmentPhase.PRACTICE
        )
        
        # 강점/약점 분석
        strengths = []
        weaknesses = []
        recommendations = []
        
        if vocab_size >= 500:
            strengths.append("풍부한 어휘")
        else:
            weaknesses.append("어휘 부족")
            recommendations.append("다양한 텍스트 읽기")
        
        if expression_diversity >= 0.5:
            strengths.append("다양한 표현력")
        else:
            weaknesses.append("표현 다양성 부족")
            recommendations.append("톤 변환 연습")
        
        # 레벨 결정
        if self.language_nurture:
            level = self.language_nurture.get_profile().level.value
        else:
            level = "unknown"
        
        return DevelopmentReport(
            overall_level=level,
            vocabulary_size=vocab_size,
            expression_diversity=expression_diversity,
            reading_count=reading_count,
            practice_count=practice_count,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )
    
    def get_status(self) -> Dict[str, Any]:
        """상태 조회"""
        report = self.get_development_report()
        return {
            "level": report.overall_level,
            "vocabulary_size": report.vocabulary_size,
            "expression_diversity": report.expression_diversity,
            "total_sessions": self.total_sessions,
            "reading_sessions": report.reading_count,
            "practice_sessions": report.practice_count,
            "strengths": report.strengths,
            "weaknesses": report.weaknesses,
            "recommendations": report.recommendations[:3],
            "auto_development": self.auto_development_enabled,
        }


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🌱 LanguageDevelopmentHub Demo")
    print("   \"언어 발달 통합 허브\"")
    print("=" * 60)
    
    hub = LanguageDevelopmentHub()
    
    # 1. 텍스트 학습
    print("\n[1] 텍스트 학습:")
    learn_result = hub.learn_from_text(
        """
        파동은 에너지의 전파 방식이다. 
        만약 두 파동이 만나면, 간섭 현상이 발생한다.
        공명은 같은 주파수의 파동이 증폭되는 현상이다.
        이 원리는 소리뿐 아니라 빛, 물, 심지어 감정에도 적용된다.
        """,
        source="physics_basics.md"
    )
    print(f"   어휘: {learn_result['vocabulary_gained']}")
    print(f"   패턴: {learn_result['patterns_learned']}")
    
    # 2. 표현 연습
    print("\n[2] 표현 연습:")
    practice_result = hub.practice_expression("나는 배우고 있다")
    print(f"   변형: {practice_result['variants_generated']}개")
    print(f"   최고 톤: {practice_result['best_tone']}")
    
    # 3. 탐색
    print("\n[3] 주제 탐색:")
    explore_result = hub.explore_topic("공명은 무엇인가?")
    print(f"   답: {explore_result.get('answer', '없음')}")
    print(f"   개념: {explore_result.get('concept', '없음')}")
    
    # 4. 자율 발달
    print("\n[4] 자율 발달 사이클:")
    auto_result = hub.autonomous_development_cycle()
    print(f"   활동: {auto_result['activities']}")
    
    # 5. 보고서
    print("\n[5] 발달 보고서:")
    status = hub.get_status()
    print(f"   레벨: {status['level']}")
    print(f"   어휘: {status['vocabulary_size']}개")
    print(f"   강점: {status['strengths']}")
    print(f"   약점: {status['weaknesses']}")
    print(f"   권장: {status['recommendations']}")
    
    print("\n✅ LanguageDevelopmentHub Demo complete!")

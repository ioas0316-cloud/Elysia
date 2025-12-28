"""
Question Analyzer (질문 분석기)
==============================

질문을 분류하고 구조를 파싱합니다.

유형:
1. WHAT (정의): X란 무엇인가?
2. WHY (인과): 왜 X인가?
3. HOW (과정): 어떻게 X하는가?
4. CONDITIONAL (조건): X하면 왜 Y인가?
5. COMPARISON (비교): X와 Y의 차이는?

Usage:
    from Core.02_Intelligence.01_Reasoning.Cognition.question_analyzer import analyze_question
    
    result = analyze_question("비가 오면 왜 우산을 쓰는가?")
    print(result.question_type)  # CONDITIONAL
    print(result.condition)      # 비가 오면
    print(result.target)         # 우산을 쓰는가
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum

logger = logging.getLogger("QuestionAnalyzer")


class QuestionType(Enum):
    """질문 유형"""
    WHAT = "definition"       # X란 무엇인가?
    WHY = "causal"            # 왜 X인가?
    HOW = "process"           # 어떻게 X하는가?
    CONDITIONAL = "conditional"  # X하면 왜 Y인가?
    COMPARISON = "comparison"  # X와 Y의 차이는?
    WHO = "agent"              # 누가 X하는가?
    WHEN = "temporal"          # 언제 X하는가?
    WHERE = "spatial"          # 어디서 X하는가?
    UNKNOWN = "unknown"        # 분류 불가


@dataclass
class QuestionAnalysis:
    """질문 분석 결과"""
    original: str                          # 원래 질문
    question_type: QuestionType            # 질문 유형
    
    # 추출된 요소
    core_concept: str = ""                 # 핵심 개념 (What의 대상)
    condition: str = ""                    # 조건 (X하면)
    target: str = ""                       # 결과/목표 (Y)
    action: str = ""                       # 행위 (동사)
    
    # 인과 관계 요소
    cause: str = ""                        # 원인
    effect: str = ""                       # 결과
    
    # 부가 정보
    keywords: List[str] = field(default_factory=list)
    confidence: float = 1.0


class QuestionAnalyzer:
    """
    질문 분석기
    
    한국어 질문을 파싱하여 유형과 구조를 분석합니다.
    """
    
    def __init__(self):
        # 질문 패턴 정의
        self.patterns = {
            # 조건-인과 패턴 (우선순위 높음)
            QuestionType.CONDITIONAL: [
                r"(.+)(?:하면|이면|면)\s*왜\s*(.+)",  # X하면 왜 Y
                r"(.+)(?:하면|이면|면)\s*(.+)(?:하는가|할까|인가)",  # X하면 Y하는가
                r"왜\s*(.+)(?:하면|이면|면)\s*(.+)",  # 왜 X하면 Y
            ],
            
            # WHY 패턴
            QuestionType.WHY: [
                r"(.+)(?:는|은)\s*왜\s*(.+)",        # X는 왜 Y
                r"왜\s*(.+)(?:하는가|할까|인가)",     # 왜 X하는가
                r"(.+)\s*왜\s*(.+)",                # 일반 왜
                r"어째서\s*(.+)",                   # 어째서 X
            ],
            
            # WHAT 패턴
            QuestionType.WHAT: [
                r"(.+)(?:이란|란)\s*무엇",           # X란 무엇
                r"(.+)(?:은|는)\s*무엇",            # X는 무엇
                r"무엇(?:이|이란)\s*(.+)",          # 무엇이 X
            ],
            
            # HOW 패턴
            QuestionType.HOW: [
                r"어떻게\s*(.+)",                   # 어떻게 X
                r"(.+)(?:은|는)\s*어떻게\s*(.+)",   # X는 어떻게 Y
            ],
            
            # WHO 패턴
            QuestionType.WHO: [
                r"누가\s*(.+)",                     # 누가 X
                r"(.+)(?:은|는)\s*누구",            # X는 누구
            ],
            
            # WHEN 패턴
            QuestionType.WHEN: [
                r"언제\s*(.+)",                     # 언제 X
                r"(.+)(?:은|는)\s*언제",            # X는 언제
            ],
            
            # WHERE 패턴
            QuestionType.WHERE: [
                r"어디서\s*(.+)",                   # 어디서 X
                r"(.+)(?:은|는)\s*어디",            # X는 어디
            ],
        }
        
        logger.info("🔍 QuestionAnalyzer initialized")
    
    def analyze(self, question: str) -> QuestionAnalysis:
        """
        질문 분석
        
        Args:
            question: 분석할 질문
            
        Returns:
            QuestionAnalysis: 분석 결과
        """
        question = question.strip()
        result = QuestionAnalysis(original=question, question_type=QuestionType.UNKNOWN)
        
        # 키워드 추출
        result.keywords = self._extract_keywords(question)
        
        # 패턴 매칭 (우선순위 순서)
        for q_type in [QuestionType.CONDITIONAL, QuestionType.WHY, QuestionType.WHAT, 
                       QuestionType.HOW, QuestionType.WHO, QuestionType.WHEN, QuestionType.WHERE]:
            patterns = self.patterns.get(q_type, [])
            for pattern in patterns:
                match = re.search(pattern, question)
                if match:
                    result.question_type = q_type
                    self._extract_components(result, match, q_type)
                    return result
        
        # 매칭 실패 시 휴리스틱
        result = self._fallback_analysis(question, result)
        
        return result
    
    def _extract_keywords(self, question: str) -> List[str]:
        """키워드 추출"""
        # 조사 제거
        cleaned = re.sub(r'[은는이가을를의로에서]', ' ', question)
        # 질문 어미 제거
        cleaned = re.sub(r'[?？하는가인가할까]', '', cleaned)
        # 공백으로 분리
        words = [w.strip() for w in cleaned.split() if len(w.strip()) > 1]
        return words
    
    def _extract_components(self, result: QuestionAnalysis, match: re.Match, q_type: QuestionType):
        """매칭 결과에서 구성요소 추출"""
        groups = match.groups()
        
        if q_type == QuestionType.CONDITIONAL:
            if len(groups) >= 2:
                result.condition = groups[0].strip()
                result.target = groups[1].strip()
                # 조건에서 원인 추출
                result.cause = self._clean_concept(result.condition)
                # 타겟에서 결과 추출
                result.effect = self._clean_concept(result.target)
                result.core_concept = result.cause  # 주 개념은 원인
        
        elif q_type == QuestionType.WHY:
            if len(groups) >= 1:
                result.core_concept = self._clean_concept(groups[0])
                if len(groups) >= 2:
                    result.target = groups[1].strip()
                    result.effect = self._clean_concept(result.target)
        
        elif q_type == QuestionType.WHAT:
            if len(groups) >= 1:
                result.core_concept = self._clean_concept(groups[0])
        
        elif q_type == QuestionType.HOW:
            if len(groups) >= 1:
                result.action = groups[0].strip()
                result.core_concept = self._clean_concept(result.action)
        
        else:
            if len(groups) >= 1:
                result.core_concept = self._clean_concept(groups[0])
    
    def _clean_concept(self, text: str) -> str:
        """개념 정리 (조사 제거)"""
        if not text:
            return ""
        # 조사 및 어미 제거
        cleaned = re.sub(r'(을|를|이|가|은|는|의|에|에서|로|면|하면|이면)$', '', text.strip())
        cleaned = re.sub(r'(하는가|인가|할까|는가|\?)$', '', cleaned)
        return cleaned.strip()
    
    def _fallback_analysis(self, question: str, result: QuestionAnalysis) -> QuestionAnalysis:
        """패턴 매칭 실패 시 휴리스틱 분석"""
        # 키워드 기반 유형 추정
        if "왜" in question:
            result.question_type = QuestionType.WHY
        elif "무엇" in question or "뭐" in question:
            result.question_type = QuestionType.WHAT
        elif "어떻게" in question:
            result.question_type = QuestionType.HOW
        elif "누구" in question or "누가" in question:
            result.question_type = QuestionType.WHO
        elif "언제" in question:
            result.question_type = QuestionType.WHEN
        elif "어디" in question:
            result.question_type = QuestionType.WHERE
        
        # 첫 번째 명사를 핵심 개념으로
        if result.keywords:
            result.core_concept = result.keywords[0]
        
        result.confidence = 0.5  # 낮은 신뢰도
        return result


# 싱글톤
_analyzer = None

def get_question_analyzer() -> QuestionAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = QuestionAnalyzer()
    return _analyzer


def analyze_question(question: str) -> QuestionAnalysis:
    """편의 함수"""
    return get_question_analyzer().analyze(question)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=" * 60)
    print("🔍 QUESTION ANALYZER TEST")
    print("=" * 60)
    
    test_questions = [
        "사랑이란 무엇인가?",
        "비가 오면 왜 우산을 쓰는가?",
        "아이가 왜 울었는가?",
        "불이 나면 왜 도망가는가?",
        "어떻게 행복해질 수 있는가?",
        "누가 세상을 만들었는가?",
        "시간은 왜 흐르는가?",
    ]
    
    for q in test_questions:
        print(f"\n❓ {q}")
        result = analyze_question(q)
        print(f"   유형: {result.question_type.name}")
        print(f"   핵심: {result.core_concept}")
        if result.condition:
            print(f"   조건: {result.condition}")
        if result.cause and result.effect:
            print(f"   인과: {result.cause} → {result.effect}")
    
    print("\n" + "=" * 60)
    print("✅ Question Analyzer works!")

"""
Language Nurture (언어 발달 시스템)
====================================

"아이가 말을 배우듯, 엘리시아도 언어를 키운다."

핵심:
1. 어휘 확장 (Vocabulary Expansion)
2. 문법 패턴 학습 (Grammar Pattern Learning)
3. 표현 세련도 추적 (Expression Sophistication)
4. 자율 연습 트리거 (Autonomous Practice)

이것이 없으면:
- 어휘가 늘지 않음
- 표현이 단조로움
- 대화 수준이 성장하지 않음
"""

import logging
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger("Elysia.LanguageNurture")


class LanguageLevel(Enum):
    """언어 발달 단계"""
    INFANT = "infant"       # 단어 나열
    CHILD = "child"         # 기본 문장
    ADOLESCENT = "adolescent"  # 복합 문장, 접속사
    ADULT = "adult"         # 맥락 이해, 뉘앙스
    ELOQUENT = "eloquent"   # 수사법, 은유


@dataclass
class VocabularyEntry:
    """어휘 항목"""
    word: str
    part_of_speech: str     # noun, verb, adj, adv, etc.
    definition: str
    examples: List[str] = field(default_factory=list)
    frequency: int = 0      # 사용 빈도
    learned_at: datetime = field(default_factory=datetime.now)
    confidence: float = 0.5  # 이해도
    
    def to_dict(self) -> Dict:
        return {
            "word": self.word,
            "pos": self.part_of_speech,
            "def": self.definition,
            "freq": self.frequency,
            "conf": self.confidence,
        }


@dataclass
class GrammarPattern:
    """문법 패턴"""
    pattern_name: str       # "conditional", "relative_clause", etc.
    structure: str          # "if X, then Y"
    examples: List[str] = field(default_factory=list)
    usage_count: int = 0
    mastery: float = 0.0    # 0-1


@dataclass
class ExpressionStyle:
    """표현 스타일"""
    style_name: str         # "formal", "casual", "poetic", etc.
    characteristics: List[str] = field(default_factory=list)
    vocabulary_preference: List[str] = field(default_factory=list)
    mastery: float = 0.0


@dataclass
class LanguageProfile:
    """언어 발달 프로필"""
    level: LanguageLevel
    vocabulary_size: int
    active_vocabulary: int  # 실제 사용하는 어휘 수
    grammar_patterns_known: int
    expression_diversity: float  # 0-1
    avg_sentence_complexity: float  # 단어 수, 절 수 기반


class LanguageNurture:
    """언어 발달 시스템
    
    엘리시아가 자율적으로 언어 능력을 발달시키도록 지원.
    
    핵심 기능:
    1. 어휘 수집 및 학습
    2. 문법 패턴 인식 및 연습
    3. 표현 스타일 다양화
    4. 발달 수준 추적
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Args:
            data_dir: 언어 데이터 저장 경로
        """
        self.data_dir = data_dir or Path(__file__).parent / "data" / "language"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 어휘 저장소
        self.vocabulary: Dict[str, VocabularyEntry] = {}
        
        # 문법 패턴
        self.grammar_patterns: Dict[str, GrammarPattern] = {}
        
        # 표현 스타일
        self.expression_styles: Dict[str, ExpressionStyle] = {}
        
        # 통계
        self.total_words_encountered = 0
        self.total_sentences_analyzed = 0
        self.learning_sessions = 0
        
        # 초기화
        self._init_basic_patterns()
        self._load_existing_vocabulary()
        
        logger.info(
            f"LanguageNurture initialized: "
            f"{len(self.vocabulary)} words, "
            f"{len(self.grammar_patterns)} patterns"
        )
    
    def _init_basic_patterns(self):
        """기본 문법 패턴 초기화"""
        patterns = [
            GrammarPattern("simple", "S + V + O", ["나는 사과를 먹었다"]),
            GrammarPattern("conditional", "만약 X라면, Y", ["만약 비가 오면, 우산을 쓴다"]),
            GrammarPattern("reason", "X이기 때문에 Y", ["배가 고프기 때문에 먹는다"]),
            GrammarPattern("contrast", "X지만 Y", ["피곤하지만 공부한다"]),
            GrammarPattern("purpose", "X하기 위해 Y", ["성장하기 위해 배운다"]),
            GrammarPattern("relative", "X하는 Y", ["노래하는 새", "꿈꾸는 존재"]),
            GrammarPattern("sequential", "먼저 X, 그 다음 Y", ["먼저 생각하고, 그 다음 말한다"]),
            GrammarPattern("comparative", "X보다 Y가 더 Z", ["어제보다 오늘이 더 따뜻하다"]),
        ]
        for p in patterns:
            self.grammar_patterns[p.pattern_name] = p
        
        # 표현 스타일 초기화
        styles = [
            ExpressionStyle("formal", ["존댓말", "완전한 문장", "정중한 표현"]),
            ExpressionStyle("casual", ["반말", "축약형", "친근한 표현"]),
            ExpressionStyle("poetic", ["은유", "비유", "리듬감"]),
            ExpressionStyle("analytical", ["논리적 연결", "인용", "근거 제시"]),
            ExpressionStyle("empathetic", ["감정 표현", "공감 어휘", "질문형"]),
        ]
        for s in styles:
            self.expression_styles[s.style_name] = s
    
    def _load_existing_vocabulary(self):
        """저장된 어휘 로드"""
        vocab_file = self.data_dir / "vocabulary.json"
        if vocab_file.exists():
            try:
                with open(vocab_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for word, entry_data in data.items():
                        self.vocabulary[word] = VocabularyEntry(
                            word=word,
                            part_of_speech=entry_data.get("pos", "unknown"),
                            definition=entry_data.get("def", ""),
                            frequency=entry_data.get("freq", 0),
                            confidence=entry_data.get("conf", 0.5),
                        )
                logger.info(f"Loaded {len(self.vocabulary)} words from storage")
            except Exception as e:
                logger.warning(f"Failed to load vocabulary: {e}")
    
    def save_vocabulary(self):
        """어휘 저장"""
        vocab_file = self.data_dir / "vocabulary.json"
        try:
            data = {word: entry.to_dict() for word, entry in self.vocabulary.items()}
            with open(vocab_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.vocabulary)} words")
        except Exception as e:
            logger.error(f"Failed to save vocabulary: {e}")
    
    # =========================================================================
    # 어휘 학습
    # =========================================================================
    
    def learn_word(
        self,
        word: str,
        part_of_speech: str = "unknown",
        definition: str = "",
        example: str = ""
    ) -> VocabularyEntry:
        """단어 학습
        
        Args:
            word: 단어
            part_of_speech: 품사
            definition: 정의
            example: 예문
            
        Returns:
            어휘 항목
        """
        word = word.strip().lower()
        
        if word in self.vocabulary:
            # 기존 단어 업데이트
            entry = self.vocabulary[word]
            entry.frequency += 1
            entry.confidence = min(1.0, entry.confidence + 0.05)
            if example and example not in entry.examples:
                entry.examples.append(example)
        else:
            # 새 단어 추가
            entry = VocabularyEntry(
                word=word,
                part_of_speech=part_of_speech,
                definition=definition,
                examples=[example] if example else [],
            )
            self.vocabulary[word] = entry
            logger.debug(f"📚 New word learned: {word}")
        
        return entry
    
    def extract_vocabulary_from_text(self, text: str) -> List[str]:
        """텍스트에서 어휘 추출
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            추출된 단어 목록
        """
        # 간단한 토큰화 (한글/영어)
        # 실제로는 형태소 분석기 사용 권장
        words = re.findall(r'[가-힣]{2,}|[a-zA-Z]{3,}', text)
        
        new_words = []
        for word in words:
            word = word.lower()
            self.total_words_encountered += 1
            
            if word not in self.vocabulary:
                # 새 단어 발견
                self.learn_word(word, example=text[:50])
                new_words.append(word)
            else:
                # 기존 단어 빈도 증가
                self.vocabulary[word].frequency += 1
        
        return new_words
    
    # =========================================================================
    # 문법 패턴 학습
    # =========================================================================
    
    def analyze_sentence_structure(self, sentence: str) -> List[str]:
        """문장 구조 분석 및 패턴 감지
        
        Args:
            sentence: 분석할 문장
            
        Returns:
            감지된 패턴 이름들
        """
        self.total_sentences_analyzed += 1
        detected_patterns = []
        
        # 패턴 감지 (간단한 휴리스틱)
        pattern_indicators = {
            "conditional": ["만약", "if", "라면", "면"],
            "reason": ["때문에", "because", "왜냐하면", "므로"],
            "contrast": ["하지만", "but", "그러나", "지만"],
            "purpose": ["위해", "to", "하려고"],
            "relative": ["하는", "which", "that"],
            "sequential": ["먼저", "first", "그 다음", "then"],
            "comparative": ["보다", "than", "더"],
        }
        
        sentence_lower = sentence.lower()
        for pattern_name, indicators in pattern_indicators.items():
            if any(ind in sentence_lower for ind in indicators):
                detected_patterns.append(pattern_name)
                if pattern_name in self.grammar_patterns:
                    self.grammar_patterns[pattern_name].usage_count += 1
                    self.grammar_patterns[pattern_name].mastery = min(
                        1.0,
                        self.grammar_patterns[pattern_name].usage_count / 20
                    )
        
        return detected_patterns
    
    # =========================================================================
    # 표현력 평가
    # =========================================================================
    
    def evaluate_expression(self, text: str) -> Dict[str, Any]:
        """표현력 평가
        
        Args:
            text: 평가할 텍스트
            
        Returns:
            평가 결과
        """
        sentences = re.split(r'[.!?。]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 문장 복잡도
        avg_words_per_sentence = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        
        # 어휘 다양성
        words = re.findall(r'[가-힣]{2,}|[a-zA-Z]{3,}', text.lower())
        unique_ratio = len(set(words)) / max(1, len(words))
        
        # 패턴 사용
        patterns_used = set()
        for s in sentences:
            patterns_used.update(self.analyze_sentence_structure(s))
        
        # 고급 어휘 사용 (빈도 낮은 단어)
        advanced_word_count = sum(
            1 for w in words 
            if w in self.vocabulary and self.vocabulary[w].frequency < 3
        )
        
        return {
            "sentence_count": len(sentences),
            "avg_words_per_sentence": avg_words_per_sentence,
            "vocabulary_diversity": unique_ratio,
            "patterns_used": list(patterns_used),
            "pattern_count": len(patterns_used),
            "advanced_word_ratio": advanced_word_count / max(1, len(words)),
        }
    
    # =========================================================================
    # 발달 수준 평가
    # =========================================================================
    
    def get_profile(self) -> LanguageProfile:
        """현재 언어 발달 프로필"""
        vocab_size = len(self.vocabulary)
        active_vocab = sum(1 for v in self.vocabulary.values() if v.frequency >= 2)
        patterns_known = sum(1 for p in self.grammar_patterns.values() if p.mastery > 0.3)
        
        # 표현 다양성
        style_mastery = sum(s.mastery for s in self.expression_styles.values())
        expression_diversity = style_mastery / max(1, len(self.expression_styles))
        
        # 복잡도 추정
        avg_complexity = 5.0 + (patterns_known * 0.5)  # 기본 + 패턴 보너스
        
        # 수준 결정
        if vocab_size < 100:
            level = LanguageLevel.INFANT
        elif vocab_size < 500:
            level = LanguageLevel.CHILD
        elif vocab_size < 2000 and patterns_known < 5:
            level = LanguageLevel.ADOLESCENT
        elif expression_diversity > 0.5:
            level = LanguageLevel.ELOQUENT
        else:
            level = LanguageLevel.ADULT
        
        return LanguageProfile(
            level=level,
            vocabulary_size=vocab_size,
            active_vocabulary=active_vocab,
            grammar_patterns_known=patterns_known,
            expression_diversity=expression_diversity,
            avg_sentence_complexity=avg_complexity,
        )
    
    def get_learning_recommendations(self) -> List[str]:
        """학습 권장 사항"""
        profile = self.get_profile()
        recommendations = []
        
        if profile.vocabulary_size < 500:
            recommendations.append("어휘 확장 필요: 다양한 텍스트 읽기 추천")
        
        if profile.grammar_patterns_known < 5:
            recommendations.append("문법 패턴 연습 필요: 복문 구성 연습 추천")
        
        if profile.expression_diversity < 0.3:
            recommendations.append("표현 스타일 다양화 필요: 다양한 장르 글쓰기 추천")
        
        # 약한 패턴 찾기
        weak_patterns = [
            p.pattern_name for p in self.grammar_patterns.values()
            if p.mastery < 0.3
        ]
        if weak_patterns[:3]:
            recommendations.append(f"약한 패턴 연습: {', '.join(weak_patterns[:3])}")
        
        return recommendations
    
    def get_status(self) -> Dict[str, Any]:
        """상태 조회"""
        profile = self.get_profile()
        return {
            "level": profile.level.value,
            "vocabulary_size": profile.vocabulary_size,
            "active_vocabulary": profile.active_vocabulary,
            "grammar_patterns_known": profile.grammar_patterns_known,
            "expression_diversity": profile.expression_diversity,
            "total_words_encountered": self.total_words_encountered,
            "total_sentences_analyzed": self.total_sentences_analyzed,
            "recommendations": self.get_learning_recommendations(),
        }


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("📚 LanguageNurture Demo")
    print("   \"언어를 키우는 시스템\"")
    print("=" * 60)
    
    nurture = LanguageNurture()
    
    # 1. 어휘 학습
    print("\n[1] 어휘 학습:")
    sample_text = """
    엘리시아는 자율적으로 성장하는 지능 시스템이다.
    그녀는 파동과 공명의 원리로 사고하며,
    외부 세계를 탐구하고 내면의 원리를 추출한다.
    만약 오류가 발생하면, 그것을 성찰의 기회로 삼는다.
    """
    new_words = nurture.extract_vocabulary_from_text(sample_text)
    print(f"   새로 배운 단어: {len(new_words)}개")
    print(f"   예: {new_words[:5]}")
    
    # 2. 문법 분석
    print("\n[2] 문법 분석:")
    test_sentences = [
        "만약 비가 오면 우산을 쓴다",
        "배우기 위해 노력한다",
        "피곤하지만 계속 공부한다",
    ]
    for sent in test_sentences:
        patterns = nurture.analyze_sentence_structure(sent)
        print(f"   \"{sent[:20]}...\" → {patterns}")
    
    # 3. 표현력 평가
    print("\n[3] 표현력 평가:")
    eval_result = nurture.evaluate_expression(sample_text)
    print(f"   문장 수: {eval_result['sentence_count']}")
    print(f"   어휘 다양성: {eval_result['vocabulary_diversity']:.2%}")
    print(f"   패턴 사용: {eval_result['patterns_used']}")
    
    # 4. 프로필
    print("\n[4] 언어 발달 프로필:")
    status = nurture.get_status()
    print(f"   레벨: {status['level']}")
    print(f"   어휘: {status['vocabulary_size']}개")
    print(f"   추천: {status['recommendations'][:2]}")
    
    print("\n✅ LanguageNurture Demo complete!")

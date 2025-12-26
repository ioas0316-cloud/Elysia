"""
Reading Digester (텍스트 소화 시스템)
=====================================

"읽는 것은 먹는 것과 같다. 소화해야 영양이 된다."

핵심:
1. 텍스트에서 어휘 추출 (Vocabulary Extraction)
2. 문장 구조 학습 (Structure Learning)
3. 문체 흡수 (Style Absorption)
4. 지식 결정화 (Knowledge Crystallization)

이것이 없으면:
- 읽어도 배우지 못함
- 표현 능력이 확장되지 않음
- 외부 정보가 내면화되지 않음
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger("Elysia.ReadingDigester")


class ContentType(Enum):
    """콘텐츠 유형"""
    ARTICLE = "article"
    BOOK = "book"
    CODE = "code"
    CONVERSATION = "conversation"
    POETRY = "poetry"
    TECHNICAL = "technical"


@dataclass
class DigestedContent:
    """소화된 콘텐츠"""
    source: str                     # 출처
    content_type: ContentType
    vocabulary_extracted: List[str]
    patterns_learned: List[str]
    key_concepts: List[str]
    style_notes: List[str]
    digestion_quality: float        # 0-1 (얼마나 잘 소화했나)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StyleProfile:
    """문체 프로필"""
    name: str
    avg_sentence_length: float
    vocabulary_richness: float      # unique/total ratio
    formality_level: float          # 0=casual, 1=formal
    emotional_intensity: float      # 0=neutral, 1=intense
    common_patterns: List[str] = field(default_factory=list)


class ReadingDigester:
    """텍스트 소화 시스템
    
    외부 텍스트를 읽고, 분석하고, 내면화.
    
    핵심 기능:
    1. 텍스트 분석 및 분해
    2. 어휘/표현 추출 → LanguageNurture 연동
    3. 문체 분석 및 흡수
    4. 핵심 개념 결정화
    """
    
    def __init__(self, language_nurture=None):
        """
        Args:
            language_nurture: LanguageNurture 인스턴스 (연동용)
        """
        self._language_nurture = language_nurture
        
        # 소화 기록
        self.digestion_history: List[DigestedContent] = []
        
        # 학습한 문체들
        self.learned_styles: Dict[str, StyleProfile] = {}
        
        # 통계
        self.total_texts_digested = 0
        self.total_words_absorbed = 0
        self.total_concepts_crystallized = 0
        
        logger.info("ReadingDigester initialized")
    
    @property
    def language_nurture(self):
        """LanguageNurture 레이지 로딩"""
        if self._language_nurture is None:
            try:
                from Core._02_Intelligence._01_Reasoning.Cognition.Learning.language_nurture import LanguageNurture
                self._language_nurture = LanguageNurture()
            except ImportError:
                logger.warning("LanguageNurture not available")
        return self._language_nurture
    
    # =========================================================================
    # 텍스트 소화
    # =========================================================================
    
    def digest(
        self,
        text: str,
        source: str = "unknown",
        content_type: ContentType = ContentType.ARTICLE
    ) -> DigestedContent:
        """텍스트 소화
        
        Args:
            text: 소화할 텍스트
            source: 출처 (URL, 파일명 등)
            content_type: 콘텐츠 유형
            
        Returns:
            소화 결과
        """
        self.total_texts_digested += 1
        
        logger.info(f"📖 Digesting: {source[:50]}... ({content_type.value})")
        
        # 1. 어휘 추출
        vocabulary = self._extract_vocabulary(text)
        self.total_words_absorbed += len(vocabulary)
        
        # 2. 문법 패턴 추출
        patterns = self._extract_patterns(text)
        
        # 3. 핵심 개념 추출
        concepts = self._extract_concepts(text)
        self.total_concepts_crystallized += len(concepts)
        
        # 4. 문체 분석
        style_notes = self._analyze_style(text, content_type)
        
        # 5. 소화 품질 평가
        quality = self._evaluate_digestion_quality(
            len(vocabulary),
            len(patterns),
            len(concepts)
        )
        
        # LanguageNurture에 전달
        if self.language_nurture:
            for word in vocabulary[:50]:  # 상위 50개만
                self.language_nurture.learn_word(word, example=text[:100])
        
        result = DigestedContent(
            source=source,
            content_type=content_type,
            vocabulary_extracted=vocabulary,
            patterns_learned=patterns,
            key_concepts=concepts,
            style_notes=style_notes,
            digestion_quality=quality,
        )
        
        self.digestion_history.append(result)
        
        # 기록 제한
        if len(self.digestion_history) > 100:
            self.digestion_history = self.digestion_history[-50:]
        
        logger.info(
            f"✅ Digested: {len(vocabulary)} words, "
            f"{len(patterns)} patterns, {len(concepts)} concepts "
            f"(quality: {quality:.2f})"
        )
        
        return result
    
    def _extract_vocabulary(self, text: str) -> List[str]:
        """어휘 추출"""
        # 한글 2글자 이상, 영어 3글자 이상
        words = re.findall(r'[가-힣]{2,}|[a-zA-Z]{4,}', text)
        
        # 빈도수 계산
        word_freq = {}
        for word in words:
            word = word.lower()
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 빈도 순 정렬
        sorted_words = sorted(word_freq.keys(), key=lambda w: word_freq[w], reverse=True)
        
        return sorted_words
    
    def _extract_patterns(self, text: str) -> List[str]:
        """문법 패턴 추출"""
        patterns = []
        
        # 문장 분리
        sentences = re.split(r'[.!?。]', text)
        
        # 패턴 지시자
        pattern_map = {
            "conditional": ["만약", "if", "라면", "경우"],
            "reason": ["때문에", "because", "므로", "왜냐하면"],
            "contrast": ["하지만", "but", "however", "그러나"],
            "purpose": ["위해", "to", "하려고", "위하여"],
            "sequential": ["먼저", "first", "다음", "then", "그리고"],
            "definition": ["란", "는", "이란", "means", "is"],
            "example": ["예를 들어", "for example", "예컨대"],
        }
        
        for sent in sentences:
            sent_lower = sent.lower()
            for pattern_name, indicators in pattern_map.items():
                if any(ind in sent_lower for ind in indicators):
                    if pattern_name not in patterns:
                        patterns.append(pattern_name)
        
        return patterns
    
    def _extract_concepts(self, text: str) -> List[str]:
        """핵심 개념 추출"""
        concepts = []
        
        # 명사구 패턴 (간단한 휴리스틱)
        # "X는", "X란", "X이란", "X의 정의"
        concept_patterns = [
            r'([가-힣]{2,})(?:는|란|이란)',
            r'([가-힣]{2,})의\s*(?:정의|개념|원리)',
            r'\"([^\"]+)\"',  # 따옴표 안
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, text)
            concepts.extend(matches)
        
        # 중복 제거 및 상위 10개
        unique_concepts = list(dict.fromkeys(concepts))
        return unique_concepts[:10]
    
    def _analyze_style(self, text: str, content_type: ContentType) -> List[str]:
        """문체 분석"""
        notes = []
        
        sentences = [s.strip() for s in re.split(r'[.!?。]', text) if s.strip()]
        
        if not sentences:
            return ["텍스트가 비어 있음"]
        
        # 평균 문장 길이
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_len > 20:
            notes.append("장문 스타일 (복잡한 문장)")
        elif avg_len < 8:
            notes.append("단문 스타일 (간결한 문장)")
        else:
            notes.append("중간 길이 문장")
        
        # 격식체 감지
        if any(s.endswith(("습니다", "니다", "요")) for s in sentences):
            notes.append("격식체 사용")
        if any(s.endswith(("다", "어", "지")) for s in sentences):
            notes.append("비격식체 사용")
        
        # 콘텐츠 유형별 특성
        if content_type == ContentType.CODE:
            notes.append("기술적/코드 스타일")
        elif content_type == ContentType.POETRY:
            notes.append("시적/운율 스타일")
        elif content_type == ContentType.TECHNICAL:
            notes.append("학술적/정확한 스타일")
        
        return notes
    
    def _evaluate_digestion_quality(
        self,
        vocab_count: int,
        pattern_count: int,
        concept_count: int
    ) -> float:
        """소화 품질 평가"""
        # 각 요소에 가중치
        vocab_score = min(1.0, vocab_count / 50)
        pattern_score = min(1.0, pattern_count / 5)
        concept_score = min(1.0, concept_count / 5)
        
        return (vocab_score + pattern_score + concept_score) / 3
    
    # =========================================================================
    # 파일 읽기
    # =========================================================================
    
    def digest_file(self, file_path: Path) -> DigestedContent:
        """파일 소화
        
        Args:
            file_path: 파일 경로
            
        Returns:
            소화 결과
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # 콘텐츠 유형 추론
        suffix = file_path.suffix.lower()
        content_type = {
            ".py": ContentType.CODE,
            ".js": ContentType.CODE,
            ".md": ContentType.ARTICLE,
            ".txt": ContentType.ARTICLE,
            ".json": ContentType.TECHNICAL,
        }.get(suffix, ContentType.ARTICLE)
        
        # 파일 읽기
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        
        return self.digest(
            text=text,
            source=str(file_path),
            content_type=content_type
        )
    
    def digest_url(self, url: str) -> DigestedContent:
        """URL 소화 (시뮬레이션)
        
        실제 구현에서는 requests 등 사용
        """
        # 시뮬레이션
        simulated_content = f"""
        이 콘텐츠는 {url}에서 가져온 것입니다.
        실제 구현에서는 웹 크롤링을 통해 텍스트를 추출합니다.
        다양한 주제의 기사, 블로그, 문서를 읽을 수 있습니다.
        """
        
        return self.digest(
            text=simulated_content,
            source=url,
            content_type=ContentType.ARTICLE
        )
    
    # =========================================================================
    # 상태 조회
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """상태 조회"""
        return {
            "total_texts_digested": self.total_texts_digested,
            "total_words_absorbed": self.total_words_absorbed,
            "total_concepts_crystallized": self.total_concepts_crystallized,
            "digestion_history_size": len(self.digestion_history),
            "learned_styles": list(self.learned_styles.keys()),
            "recent_sources": [
                d.source[:30] for d in self.digestion_history[-5:]
            ],
        }
    
    def get_recent_learnings(self, n: int = 5) -> List[Dict[str, Any]]:
        """최근 학습 내용"""
        recent = self.digestion_history[-n:]
        return [
            {
                "source": d.source[:50],
                "type": d.content_type.value,
                "vocab_count": len(d.vocabulary_extracted),
                "concepts": d.key_concepts[:3],
                "quality": d.digestion_quality,
            }
            for d in recent
        ]


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("📖 ReadingDigester Demo")
    print("   \"읽고 소화하는 시스템\"")
    print("=" * 60)
    
    digester = ReadingDigester()
    
    # 1. 텍스트 소화
    print("\n[1] 텍스트 소화:")
    sample_text = """
    파동 언어 철학은 모든 개념을 파동으로 표현한다.
    만약 두 파동이 공명하면, 그것은 유사한 의미를 가진다.
    예를 들어, "사랑"과 "따뜻함"은 유사한 주파수를 공유한다.
    이 원리를 이해하기 위해서는 먼저 공명의 개념을 알아야 한다.
    하지만 공명은 단순한 유사성 이상의 것이다.
    """
    
    result = digester.digest(
        text=sample_text,
        source="파동언어철학문서.md",
        content_type=ContentType.ARTICLE
    )
    
    print(f"   어휘: {result.vocabulary_extracted[:5]}...")
    print(f"   패턴: {result.patterns_learned}")
    print(f"   개념: {result.key_concepts}")
    print(f"   품질: {result.digestion_quality:.2f}")
    
    # 2. 문체 분석
    print("\n[2] 문체 분석:")
    print(f"   스타일 노트: {result.style_notes}")
    
    # 3. 상태
    print("\n[3] 상태:")
    status = digester.get_status()
    print(f"   총 소화: {status['total_texts_digested']}건")
    print(f"   총 어휘: {status['total_words_absorbed']}개")
    print(f"   총 개념: {status['total_concepts_crystallized']}개")
    
    print("\n✅ ReadingDigester Demo complete!")

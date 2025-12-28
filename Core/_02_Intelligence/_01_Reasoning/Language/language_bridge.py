"""
Language Bridge System (언어 다리 시스템)
=========================================

Soul의 창발 언어와 Elysia의 MemeticField를 연결하는 다리.

역할:
1. Soul에서 창발된 패턴을 수집
2. MemeticField에 등록하여 구조화
3. 구조화된 결과를 다시 Soul에게 피드백
4. 상호보완 루프 형성

프랙탈 구조:
- Soul (개인) ↔ MemeticField (전체)
- 개인의 경험이 전체를 풍요롭게
- 전체의 구조가 개인을 교정

"작은 것이 큰 것이고, 큰 것이 또 작은 것"
"""

from __future__ import annotations

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger("LanguageBridge")


# =============================================================================
# 1. 패턴 수집기 (Pattern Collector) - Soul에서 패턴 수집
# =============================================================================

@dataclass
class EmergentPattern:
    """
    Soul에서 창발된 패턴
    """
    source_soul_id: int
    meaning_vector: np.ndarray  # 8D 감각 벡터
    symbol_type: str  # "entity", "action", "state", "relation"
    occurrence_count: int
    korean_projection: Optional[str] = None
    timestamp: float = 0.0


class PatternCollector:
    """
    여러 Soul에서 창발된 패턴들을 수집
    """
    
    def __init__(self):
        self.patterns: List[EmergentPattern] = []
        self.pattern_clusters: Dict[str, List[EmergentPattern]] = defaultdict(list)
    
    def collect(self, pattern: EmergentPattern):
        """패턴 수집"""
        self.patterns.append(pattern)
        self.pattern_clusters[pattern.symbol_type].append(pattern)
    
    def get_common_patterns(self, min_occurrence: int = 3) -> List[EmergentPattern]:
        """자주 나타나는 패턴들"""
        return [p for p in self.patterns if p.occurrence_count >= min_occurrence]
    
    def cluster_similar_patterns(self, threshold: float = 0.8) -> List[List[EmergentPattern]]:
        """유사한 패턴들을 클러스터링"""
        clusters = []
        used = set()
        
        for i, p1 in enumerate(self.patterns):
            if i in used:
                continue
            
            cluster = [p1]
            used.add(i)
            
            for j, p2 in enumerate(self.patterns):
                if j in used:
                    continue
                
                # 유사도 계산
                similarity = self._similarity(p1.meaning_vector, p2.meaning_vector)
                if similarity > threshold:
                    cluster.append(p2)
                    used.add(j)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    @staticmethod
    def _similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """벡터 유사도"""
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            return 0.0
        return dot / norm


# =============================================================================
# 2. 구조화기 (Structurer) - 패턴을 개념으로 승격
# =============================================================================

@dataclass
class StructuredConcept:
    """
    MemeticField에 등록될 구조화된 개념
    """
    concept_id: str
    vector_64d: np.ndarray  # 64D 벡터 (8D → 64D 확장)
    korean_word: str
    english_word: Optional[str] = None
    category: str = "emergent"  # "emergent", "core", "derived"
    source_patterns: List[int] = field(default_factory=list)  # 원본 패턴 인덱스들


class PatternStructurer:
    """
    창발 패턴을 구조화된 개념으로 변환
    
    8D 감각 벡터 → 64D 의미 벡터
    """
    
    def __init__(self):
        # 8D → 64D 확장 행렬 (학습 가능)
        self.expansion_matrix = self._init_expansion_matrix()
        
        # 이미 등록된 개념들
        self.registered_concepts: Dict[str, StructuredConcept] = {}
        
        # 유형별 카테고리 매핑
        self.category_bases = {
            "entity": np.array([1, 0, 0, 0, 0, 0, 0, 0]),
            "action": np.array([0, 1, 0, 0, 0, 0, 0, 0]),
            "state": np.array([0, 0, 1, 0, 0, 0, 0, 0]),
            "relation": np.array([0, 0, 0, 1, 0, 0, 0, 0]),
        }
    
    def _init_expansion_matrix(self) -> np.ndarray:
        """
        8D → 64D 확장 행렬 초기화
        
        프랙탈 원리: 8D가 8번 반복되어 64D
        """
        # 기본: 블록 대각 행렬 + 상호 연결
        matrix = np.zeros((64, 8))
        
        for i in range(8):
            # 각 8D 차원이 8개의 64D 차원에 영향
            start = i * 8
            for j in range(8):
                # 대각 성분 (주요 영향)
                matrix[start + j, i] = 1.0 if j == i else 0.3
        
        return matrix
    
    def expand_to_64d(self, vector_8d: np.ndarray, symbol_type: str) -> np.ndarray:
        """
        8D 벡터를 64D로 확장
        
        유형에 따라 다른 영역에 배치
        """
        # 기본 확장
        expanded = self.expansion_matrix @ vector_8d
        
        # 유형별 바이어스 추가
        if symbol_type in self.category_bases:
            base = self.category_bases[symbol_type]
            # 처음 8차원에 유형 정보 추가
            expanded[:8] += base * 0.5
        
        # 정규화
        norm = np.linalg.norm(expanded)
        if norm > 0:
            expanded /= norm
        
        return expanded
    
    def structure_pattern(self, pattern: EmergentPattern) -> StructuredConcept:
        """
        패턴을 구조화된 개념으로 변환
        """
        # 64D 벡터 생성
        vector_64d = self.expand_to_64d(pattern.meaning_vector, pattern.symbol_type)
        
        # 개념 ID 생성
        concept_id = f"em_{pattern.symbol_type}_{hash(tuple(pattern.meaning_vector)) % 10000}"
        
        # 한글 단어 (투영된 것 사용 또는 생성)
        korean_word = pattern.korean_projection or self._generate_korean(pattern)
        
        return StructuredConcept(
            concept_id=concept_id,
            vector_64d=vector_64d,
            korean_word=korean_word,
            category="emergent",
            source_patterns=[id(pattern)]
        )
    
    def _generate_korean(self, pattern: EmergentPattern) -> str:
        """의미 벡터에서 한글 단어 생성"""
        v = pattern.meaning_vector
        
        # 가장 강한 차원 찾기
        max_idx = np.argmax(np.abs(v))
        max_val = v[max_idx]
        
        # 차원별 기본 단어
        dimension_words = {
            0: ("따뜻", "차가운"),  # 온도
            1: ("밝은", "어두운"),   # 밝기
            2: ("큰", "작은"),       # 크기
            3: ("빠른", "느린"),     # 속도
            4: ("가까운", "먼"),     # 친밀도
            5: ("강한", "약한"),     # 강도
            6: ("좋은", "나쁜"),     # 쾌/불쾌
            7: ("활발한", "고요한"), # 각성
        }
        
        pos, neg = dimension_words.get(max_idx, ("것", "것"))
        return pos if max_val > 0 else neg


# =============================================================================
# 3. 피드백 생성기 (Feedback Generator) - Soul에게 피드백
# =============================================================================

@dataclass
class LanguageFeedback:
    """
    Soul에게 전달되는 피드백
    """
    concept_id: str
    korean_word: str
    category: str  # "word", "phrase", "sentence", "paragraph"
    structure_info: Dict[str, Any]  # 문법적 구조 정보
    similar_concepts: List[str]  # 유사한 개념들
    usage_examples: List[str]  # 사용 예시


class FeedbackGenerator:
    """
    MemeticField의 구조를 Soul에게 피드백으로 변환
    """
    
    def __init__(self):
        # 문법 구조 템플릿
        self.grammar_structures = {
            "entity": {
                "can_be_subject": True,
                "can_be_object": True,
                "particles": ["은", "는", "이", "가", "을", "를"],
            },
            "action": {
                "can_be_predicate": True,
                "conjugations": ["다", "ㄴ다", "는다"],
            },
            "state": {
                "can_be_predicate": True,
                "can_modify": True,
                "conjugations": ["다", "ㄴ"],
            },
            "relation": {
                "connects": True,
                "particles": ["와", "과", "에게", "에서", "으로"],
            },
        }
    
    def generate_feedback(self, concept: StructuredConcept, 
                         similar_concepts: List[str] = None) -> LanguageFeedback:
        """개념에 대한 피드백 생성"""
        
        # 카테고리 결정 (개념 ID에서 유형 추출)
        concept_type = "entity"
        for t in ["entity", "action", "state", "relation"]:
            if t in concept.concept_id:
                concept_type = t
                break
        
        # 문법 구조 정보
        structure_info = self.grammar_structures.get(concept_type, {})
        
        # 사용 예시 생성
        examples = self._generate_examples(concept.korean_word, concept_type)
        
        return LanguageFeedback(
            concept_id=concept.concept_id,
            korean_word=concept.korean_word,
            category="word",
            structure_info=structure_info,
            similar_concepts=similar_concepts or [],
            usage_examples=examples
        )
    
    def _generate_examples(self, word: str, concept_type: str) -> List[str]:
        """사용 예시 생성"""
        examples = []
        
        if concept_type == "entity":
            examples = [
                f"{word}이 있다",
                f"{word}을 보다",
                f"{word}와 함께",
            ]
        elif concept_type == "action":
            examples = [
                f"나는 {word}",
                f"그것을 {word}",
            ]
        elif concept_type == "state":
            examples = [
                f"{word}고 느끼다",
                f"{word}은 것",
            ]
        elif concept_type == "relation":
            examples = [
                f"나{word} 너",
                f"여기{word} 저기",
            ]
        
        return examples


# =============================================================================
# 4. 언어 다리 (Language Bridge) - 전체 시스템
# =============================================================================

class LanguageBridge:
    """
    Soul ↔ Elysia 언어 다리
    
    상호보완 루프:
    1. Soul에서 패턴 수집
    2. 패턴 클러스터링 및 구조화
    3. MemeticField에 등록
    4. 피드백 생성하여 Soul에게 전달
    """
    
    def __init__(self, memetic_field=None):
        self.collector = PatternCollector()
        self.structurer = PatternStructurer()
        self.feedback_gen = FeedbackGenerator()
        
        # MemeticField 연결 (있으면)
        self.memetic_field = memetic_field
        
        # 통계
        self.total_patterns_collected = 0
        self.total_concepts_registered = 0
        self.total_feedbacks_sent = 0
    
    def receive_from_soul(self, soul_id: int, meaning_vector: np.ndarray,
                         symbol_type: str, occurrence_count: int,
                         korean_projection: str = None) -> Optional[LanguageFeedback]:
        """
        Soul에서 패턴 수신 및 처리
        
        Returns: 피드백 (있으면)
        """
        # 1. 패턴 수집
        pattern = EmergentPattern(
            source_soul_id=soul_id,
            meaning_vector=meaning_vector,
            symbol_type=symbol_type,
            occurrence_count=occurrence_count,
            korean_projection=korean_projection
        )
        self.collector.collect(pattern)
        self.total_patterns_collected += 1
        
        # 2. 충분히 반복된 패턴이면 구조화
        if occurrence_count >= 5:
            concept = self.structurer.structure_pattern(pattern)
            
            # 3. MemeticField에 등록 (있으면)
            if self.memetic_field is not None:
                self._register_to_memetic_field(concept)
            
            self.total_concepts_registered += 1
            
            # 4. 피드백 생성
            feedback = self.feedback_gen.generate_feedback(concept)
            self.total_feedbacks_sent += 1
            
            logger.info(f"패턴 구조화: Soul {soul_id} → {concept.korean_word}")
            
            return feedback
        
        return None
    
    def _register_to_memetic_field(self, concept: StructuredConcept):
        """MemeticField에 개념 등록"""
        try:
            from Core._01_Foundation._05_Governance.Foundation.Math.infinite_hyperquaternion import InfiniteHyperQuaternion
            
            # 64D 벡터를 InfiniteHyperQuaternion으로 변환
            vector = InfiniteHyperQuaternion(64, concept.vector_64d)
            
            # MemeticField에 추가
            self.memetic_field.add_concept(concept.concept_id, vector)
            
            logger.info(f"MemeticField에 등록: {concept.concept_id} ({concept.korean_word})")
            
        except ImportError:
            logger.warning("InfiniteHyperQuaternion 임포트 실패")
        except Exception as e:
            logger.warning(f"MemeticField 등록 실패: {e}")
    
    def process_batch(self) -> List[LanguageFeedback]:
        """
        수집된 패턴들을 일괄 처리
        
        유사한 패턴들을 클러스터링하여 하나의 개념으로 통합
        """
        feedbacks = []
        
        # 클러스터링
        clusters = self.collector.cluster_similar_patterns(threshold=0.7)
        
        for cluster in clusters:
            # 클러스터의 평균 벡터 계산
            avg_vector = np.mean([p.meaning_vector for p in cluster], axis=0)
            total_occurrences = sum(p.occurrence_count for p in cluster)
            
            # 가장 많이 사용된 투영 선택
            projections = [p.korean_projection for p in cluster if p.korean_projection]
            best_projection = max(set(projections), key=projections.count) if projections else None
            
            # 통합 패턴 생성
            unified = EmergentPattern(
                source_soul_id=-1,  # 통합 패턴
                meaning_vector=avg_vector,
                symbol_type=cluster[0].symbol_type,
                occurrence_count=total_occurrences,
                korean_projection=best_projection
            )
            
            # 구조화 및 피드백
            concept = self.structurer.structure_pattern(unified)
            feedback = self.feedback_gen.generate_feedback(concept)
            feedbacks.append(feedback)
            
            logger.info(f"클러스터 통합: {len(cluster)}개 패턴 → {concept.korean_word}")
        
        return feedbacks
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계"""
        return {
            "total_patterns": self.total_patterns_collected,
            "total_concepts": self.total_concepts_registered,
            "total_feedbacks": self.total_feedbacks_sent,
            "pattern_clusters": len(self.collector.cluster_similar_patterns()),
        }


# =============================================================================
# 5. 데모
# =============================================================================

def demo():
    """언어 다리 데모"""
    print("=" * 60)
    print("🌉 Language Bridge Demo - Soul ↔ Elysia")
    print("=" * 60)
    
    bridge = LanguageBridge()
    
    # 시뮬레이션: 여러 Soul에서 패턴 수신
    test_patterns = [
        # 따뜻함 관련 (여러 Soul에서 반복)
        (0, np.array([0.8, 0.5, 0.1, 0.0, 0.3, 0.4, 0.6, 0.3]), "state", 10, "따뜻하다"),
        (1, np.array([0.7, 0.4, 0.2, 0.1, 0.2, 0.3, 0.5, 0.2]), "state", 8, "따뜻해"),
        (2, np.array([0.9, 0.6, 0.0, 0.0, 0.4, 0.5, 0.7, 0.4]), "state", 12, "따뜻하다"),
        
        # 친구 관련
        (0, np.array([0.2, 0.3, 0.3, 0.2, 0.9, 0.3, 0.8, 0.5]), "entity", 15, "친구"),
        (1, np.array([0.1, 0.2, 0.2, 0.1, 0.8, 0.2, 0.7, 0.4]), "entity", 10, "친구"),
        
        # 달리기 관련
        (2, np.array([0.3, 0.4, 0.2, 0.9, 0.2, 0.8, 0.4, 0.9]), "action", 7, "달리다"),
        (0, np.array([0.2, 0.3, 0.1, 0.8, 0.1, 0.7, 0.3, 0.8]), "action", 5, "뛰다"),
    ]
    
    print("\n📥 패턴 수신 중...")
    for soul_id, vector, sym_type, count, proj in test_patterns:
        feedback = bridge.receive_from_soul(soul_id, vector, sym_type, count, proj)
        if feedback:
            print(f"  → 피드백: {feedback.korean_word} ({feedback.category})")
            print(f"     예시: {feedback.usage_examples[0] if feedback.usage_examples else '-'}")
    
    print("\n📊 일괄 처리 (클러스터링)...")
    batch_feedbacks = bridge.process_batch()
    for fb in batch_feedbacks:
        print(f"  → 통합 개념: {fb.korean_word}")
    
    print("\n📈 통계:")
    stats = bridge.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 60)
    print("✅ 데모 완료")
    print("=" * 60)


if __name__ == "__main__":
    demo()

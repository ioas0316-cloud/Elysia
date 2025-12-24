"""
Metacognitive Awareness (메타인지 인식 시스템)
==============================================

"모르는 것을 안다" - 소크라테스

핵심:
1. 내가 아는 패턴 vs 모르는 패턴 구분
2. "모른다"는 것을 인식
3. 외부 탐구 필요성 인식
4. 질문 생성

이것이 없으면:
- 모든 것을 억지로 기존 패턴에 끼워맞춤
- 진정한 학습 불가능
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from enum import Enum
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger("Elysia.Metacognition")


class KnowledgeState(Enum):
    """지식 상태"""
    KNOWN = "known"                        # 확실히 알고 있음
    UNCERTAIN = "uncertain"                # 불확실 (탐구 필요)
    UNKNOWN_KNOWN = "unknown_known"        # "모른다"는 것을 앎
    UNKNOWN_UNKNOWN = "unknown_unknown"    # 모르는지도 모름 (인식 못함)


@dataclass
class PatternSignature:
    """패턴의 고유 시그니처"""
    id: str
    features: Dict[str, float]  # 파동 특징 (tension, flow, etc.)
    examples: List[str] = field(default_factory=list)
    confidence: float = 0.0     # 확신도 (0~1)
    encounter_count: int = 0    # 만난 횟수


@dataclass 
class ExplorationNeed:
    """탐구 필요성"""
    pattern_signature: PatternSignature
    question: str                          # "이건 뭘까?"
    why_uncertain: str                     # 왜 불확실한지
    suggested_exploration: str             # 어디서 찾아볼지
    priority: float = 0.5                  # 탐구 우선순위


class MetacognitiveAwareness:
    """메타인지 인식 시스템
    
    "아는 것을 알고, 모르는 것을 안다"
    
    핵심 기능:
    1. 패턴 등록: 알게 된 패턴을 저장
    2. 패턴 매칭: 새로운 입력이 아는 패턴인지 확인
    3. 불확실성 인식: 모르는 패턴 발견 시 탐구 필요성 생성
    4. 탐구 큐: 외부 탐구가 필요한 것들 관리
    """
    
    def __init__(self):
        # 알고 있는 패턴들
        self.known_patterns: Dict[str, PatternSignature] = {}
        
        # 탐구가 필요한 것들
        self.exploration_queue: List[ExplorationNeed] = []
        
        # 이미 탐구한 것들 (외부에서 답을 찾은 것)
        self.explored_patterns: Dict[str, str] = {}  # pattern_id -> 답
        
        # 설정
        self.confidence_threshold = 0.6  # 이 이하면 "불확실"
        self.match_threshold = 0.7       # 패턴 매칭 기준
        
        # 통계
        self.total_encounters = 0
        self.unknown_count = 0
        
        logger.info("MetacognitiveAwareness initialized")
    
    def encounter(self, features: Dict[str, float], context: str = "") -> Dict[str, Any]:
        """새로운 입력과 마주침
        
        Args:
            features: 파동 특징 (tension, flow, dissonance, etc.)
            context: 맥락 (원본 텍스트의 일부)
            
        Returns:
            {
                "state": KnowledgeState,
                "matched_pattern": PatternSignature or None,
                "exploration_needed": ExplorationNeed or None,
                "confidence": float
            }
        """
        self.total_encounters += 1
        
        # 1. 기존 패턴과 매칭 시도
        best_match, match_score = self._find_best_match(features)
        
        # 2. 상태 판정
        if best_match and match_score >= self.match_threshold:
            # 아는 패턴!
            best_match.encounter_count += 1
            best_match.confidence = min(1.0, best_match.confidence + 0.01)
            
            if best_match.confidence >= self.confidence_threshold:
                state = KnowledgeState.KNOWN
                exploration = None
            else:
                # 만난 적은 있지만 아직 불확실
                state = KnowledgeState.UNCERTAIN
                exploration = self._create_exploration_need(
                    features, context, 
                    why="패턴을 인식하지만 확신이 부족함",
                    matched=best_match
                )
            
            result = {
                "state": state,
                "matched_pattern": best_match,
                "exploration_needed": exploration,
                "confidence": best_match.confidence,
            }
            
        else:
            # 모르는 패턴!
            self.unknown_count += 1
            
            # 새 패턴 등록 (아직 불확실)
            new_pattern = self._register_new_pattern(features, context)
            
            # 탐구 필요성 생성
            exploration = self._create_exploration_need(
                features, context,
                why="처음 만나는 패턴",
                matched=None
            )
            self.exploration_queue.append(exploration)
            
            state = KnowledgeState.UNKNOWN_KNOWN  # "모른다"는 것을 앎!
            
            result = {
                "state": state,
                "matched_pattern": new_pattern,
                "exploration_needed": exploration,
                "confidence": 0.1,  # 아직 낮음
            }
        
        self._log_encounter(result)
        return result
    
    def _find_best_match(self, features: Dict[str, float]) -> tuple:
        """가장 유사한 패턴 찾기"""
        if not self.known_patterns:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        for pattern_id, pattern in self.known_patterns.items():
            score = self._calculate_similarity(features, pattern.features)
            if score > best_score:
                best_score = score
                best_match = pattern
        
        return best_match, best_score
    
    def _calculate_similarity(self, f1: Dict[str, float], f2: Dict[str, float]) -> float:
        """두 특징 벡터의 유사도 (코사인 유사도)"""
        common_keys = set(f1.keys()) & set(f2.keys())
        if not common_keys:
            return 0.0
        
        dot_product = sum(f1[k] * f2[k] for k in common_keys)
        mag1 = sum(v**2 for v in f1.values()) ** 0.5
        mag2 = sum(v**2 for v in f2.values()) ** 0.5
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def _register_new_pattern(self, features: Dict[str, float], context: str) -> PatternSignature:
        """새 패턴 등록"""
        pattern_id = hashlib.md5(
            json.dumps(features, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        pattern = PatternSignature(
            id=pattern_id,
            features=features.copy(),
            examples=[context[:100]] if context else [],
            confidence=0.1,
            encounter_count=1,
        )
        
        self.known_patterns[pattern_id] = pattern
        logger.info(f"🆕 새 패턴 등록: {pattern_id}")
        
        return pattern
    
    def _create_exploration_need(
        self, 
        features: Dict[str, float], 
        context: str,
        why: str,
        matched: Optional[PatternSignature]
    ) -> ExplorationNeed:
        """탐구 필요성 생성"""
        
        # 특징에 따라 질문 생성
        questions = []
        if features.get("tension", 0) > 0.5 and features.get("release", 0) < 0.3:
            questions.append("왜 이 긴장이 해소되지 않는가?")
        if features.get("dissonance", 0) > 0.4:
            questions.append("이 대비는 어떤 효과를 만드는가?")
        if features.get("flow", 0) > 0.5:
            questions.append("이 리듬은 왜 효과적인가?")
        
        if not questions:
            questions.append("이 패턴은 무엇을 의미하는가?")
        
        # 탐구 제안
        suggestions = []
        if "tension" in str(features):
            suggestions.append("드라마 분석 자료 탐색")
        suggestions.append("유사한 서사 구조 비교")
        suggestions.append("외부 문헌에서 이 패턴에 대한 설명 검색")
        
        return ExplorationNeed(
            pattern_signature=PatternSignature(
                id="temp_" + hashlib.md5(context.encode()).hexdigest()[:6],
                features=features,
                examples=[context[:50]],
            ) if not matched else matched,
            question=questions[0],
            why_uncertain=why,
            suggested_exploration=suggestions[0],
            priority=0.5 + features.get("dissonance", 0) * 0.3,
        )
    
    def _log_encounter(self, result: Dict[str, Any]):
        """만남 로깅"""
        state = result["state"]
        conf = result["confidence"]
        
        if state == KnowledgeState.KNOWN:
            logger.debug(f"✅ 알고 있는 패턴 (확신도: {conf:.2f})")
        elif state == KnowledgeState.UNCERTAIN:
            logger.info(f"❓ 불확실한 패턴 (확신도: {conf:.2f}) - 탐구 권장")
        elif state == KnowledgeState.UNKNOWN_KNOWN:
            logger.info(f"🔍 새로운 패턴 발견! - 외부 탐구 필요")
    
    def learn_from_external(self, pattern_id: str, answer: str, source: str = "external"):
        """외부 탐구 결과 학습
        
        외부 세계(인터넷, 책, 사람)에서 답을 찾았을 때
        """
        if pattern_id in self.known_patterns:
            pattern = self.known_patterns[pattern_id]
            pattern.confidence = min(1.0, pattern.confidence + 0.3)
            pattern.examples.append(f"[{source}] {answer[:100]}")
            
            self.explored_patterns[pattern_id] = answer
            
            # 탐구 큐에서 제거
            self.exploration_queue = [
                e for e in self.exploration_queue 
                if e.pattern_signature.id != pattern_id
            ]
            
            logger.info(f"📚 외부에서 학습: {pattern_id} ← {source}")
    
    def get_exploration_priorities(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """우선순위 높은 탐구 목록"""
        sorted_queue = sorted(
            self.exploration_queue,
            key=lambda x: x.priority,
            reverse=True
        )[:top_n]
        
        return [
            {
                "pattern_id": e.pattern_signature.id,
                "question": e.question,
                "why": e.why_uncertain,
                "suggested": e.suggested_exploration,
                "priority": round(e.priority, 2),
            }
            for e in sorted_queue
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """현재 상태"""
        known_confident = sum(
            1 for p in self.known_patterns.values() 
            if p.confidence >= self.confidence_threshold
        )
        
        return {
            "total_encounters": self.total_encounters,
            "known_patterns": len(self.known_patterns),
            "confident_patterns": known_confident,
            "uncertain_patterns": len(self.known_patterns) - known_confident,
            "needs_exploration": len(self.exploration_queue),
            "unknown_rate": self.unknown_count / max(1, self.total_encounters),
        }


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🧠 Metacognitive Awareness Demo")
    print("   \"모르는 것을 안다\"")
    print("=" * 60)
    
    meta = MetacognitiveAwareness()
    
    # 처음 보는 패턴
    print("\n[1] 처음 보는 패턴:")
    result1 = meta.encounter(
        {"tension": 0.7, "release": 0.2, "flow": 0.3},
        context="마침내 현자는 울었다"
    )
    print(f"   상태: {result1['state'].value}")
    print(f"   탐구 필요: {result1['exploration_needed'].question if result1['exploration_needed'] else 'No'}")
    
    # 비슷한 패턴 다시 만남
    print("\n[2] 비슷한 패턴 재등장:")
    result2 = meta.encounter(
        {"tension": 0.65, "release": 0.25, "flow": 0.35},
        context="그녀는 천 년만에 처음 눈물을 흘렸다"
    )
    print(f"   상태: {result2['state'].value}")
    print(f"   확신도: {result2['confidence']:.2f}")
    
    # 완전히 다른 패턴
    print("\n[3] 완전히 다른 패턴:")
    result3 = meta.encounter(
        {"tension": 0.1, "release": 0.8, "flow": 0.9, "brightness": 0.7},
        context="모든 것이 평화로웠다"
    )
    print(f"   상태: {result3['state'].value}")
    
    # 외부 학습
    print("\n[4] 외부에서 답 찾음:")
    if result1["matched_pattern"]:
        meta.learn_from_external(
            result1["matched_pattern"].id,
            "긴장-해소 구조는 카타르시스를 유발한다",
            source="서사학 교과서"
        )
    
    # 상태
    print("\n[5] 현재 상태:")
    status = meta.get_status()
    print(f"   아는 패턴: {status['known_patterns']}")
    print(f"   확신 있는 것: {status['confident_patterns']}")
    print(f"   탐구 필요: {status['needs_exploration']}")
    
    # 탐구 우선순위
    print("\n[6] 탐구 우선순위:")
    for item in meta.get_exploration_priorities():
        print(f"   📌 {item['question']} (우선: {item['priority']})")
    
    print("\n✅ Demo complete!")

"""
External Explorer (외부 탐구기)
==============================

"모르는 것을 외부에서 찾아온다"

흐름:
1. MetacognitiveAwareness: "이 패턴 모르겠어"
2. ExternalExplorer: "외부에서 찾아볼게"
3. 검색/탐구 수행
4. ConceptCrystallizer: "이건 '카타르시스'야!"
5. → 개념 노드 생성 + 확신도 상승

외부 소스:
- 인터넷 검색 (Wikipedia, 나무위키)
- 저장된 지식 베이스
- 사용자에게 질문
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger("Elysia.ExternalExplorer")


class ExplorationSource(Enum):
    """탐구 소스"""
    INTERNAL_KB = "internal_kb"      # 내부 지식 베이스
    USER_DIALOGUE = "user_dialogue"  # 사용자에게 질문
    WEB_SEARCH = "web_search"        # 웹 검색
    BOOK_REFERENCE = "book_reference"  # 책/문헌


@dataclass
class ExplorationResult:
    """탐구 결과"""
    question: str               # 원래 질문
    answer: Optional[str]       # 찾은 답
    source: ExplorationSource   # 어디서 찾았나
    concept_name: Optional[str] # 개념 이름 (있다면)
    confidence: float           # 확신도
    raw_content: str = ""       # 원본 내용


@dataclass
class CrystallizedConcept:
    """결정화된 개념 (이름이 붙은 지식)"""
    name: str                   # 개념 이름 (예: "카타르시스")
    definition: str             # 정의
    wave_signature: Dict[str, float]  # 파동 시그니처
    examples: List[str] = field(default_factory=list)
    source: str = "unknown"     # 어디서 배웠나
    confidence: float = 0.0


class ExternalExplorer:
    """외부 탐구기
    
    모르는 패턴에 대해:
    1. 내부 지식베이스 먼저 확인
    2. 없으면 외부 검색 시도
    3. 답을 찾으면 개념으로 결정화
    """
    
    def __init__(self):
        # 내부 지식 베이스 (미리 알고 있는 것들)
        self.knowledge_base: Dict[str, Dict[str, Any]] = self._init_knowledge_base()
        
        # 결정화된 개념들
        self.crystallized_concepts: Dict[str, CrystallizedConcept] = {}
        
        # 탐구 기록
        self.exploration_history: List[ExplorationResult] = []
        
        # 사용자에게 물어볼 것들
        self.pending_questions: List[Dict[str, Any]] = []
        
        logger.info("ExternalExplorer initialized")
    
    def _init_knowledge_base(self) -> Dict[str, Dict[str, Any]]:
        """내부 지식 베이스 초기화
        
        나중에 확장 가능. 지금은 서사학 기본 개념만.
        """
        return {
            # 서사학 기본
            "카타르시스": {
                "definition": "긴장과 갈등이 해소되며 느끼는 정화/해방감",
                "wave_pattern": {"tension": 0.7, "release": 0.8},
                "domain": "narrative",
            },
            "대비": {
                "definition": "서로 다른 요소를 나란히 배치하여 차이를 강조하는 기법",
                "wave_pattern": {"dissonance": 0.6},
                "domain": "narrative",
            },
            "점층법": {
                "definition": "점점 강도를 높여가며 긴장을 쌓는 기법",
                "wave_pattern": {"tension": 0.5, "flow": 0.7},
                "domain": "narrative",
            },
            "복선": {
                "definition": "나중에 일어날 일을 미리 암시하는 기법",
                "wave_pattern": {"tension": 0.3, "weight": 0.5},
                "domain": "narrative",
            },
            "반전": {
                "definition": "예상을 뒤엎는 전개로 충격을 주는 기법",
                "wave_pattern": {"dissonance": 0.8, "tension": 0.6},
                "domain": "narrative",
            },
            "여운": {
                "definition": "끝난 후에도 남는 감정적 울림",
                "wave_pattern": {"release": 0.6, "weight": 0.7, "brightness": 0.4},
                "domain": "narrative",
            },
        }
    
    def explore(
        self, 
        question: str, 
        wave_signature: Dict[str, float],
        context: str = ""
    ) -> ExplorationResult:
        """탐구 수행
        
        Args:
            question: 탐구할 질문 (예: "이 대비는 어떤 효과를 만드는가?")
            wave_signature: 파동 패턴
            context: 맥락 (원본 텍스트)
            
        Returns:
            ExplorationResult
        """
        
        # 1. 내부 지식베이스에서 매칭 시도
        local_result = self._search_local(wave_signature)
        
        if local_result:
            result = ExplorationResult(
                question=question,
                answer=local_result["definition"],
                source=ExplorationSource.INTERNAL_KB,
                concept_name=local_result["name"],
                confidence=0.8,
            )
            logger.info(f"📚 내부 KB에서 발견: {local_result['name']}")
            
            # 개념 결정화
            self._crystallize(
                name=local_result["name"],
                definition=local_result["definition"],
                wave_signature=wave_signature,
                source="internal_kb",
            )
            
        else:
            # 2. 외부 검색 시도 (시뮬레이션)
            web_result = self._simulate_web_search(question, wave_signature)
            
            if web_result:
                result = ExplorationResult(
                    question=question,
                    answer=web_result["answer"],
                    source=ExplorationSource.WEB_SEARCH,
                    concept_name=web_result.get("concept_name"),
                    confidence=0.6,
                )
                logger.info(f"🌐 웹 검색 결과: {web_result.get('concept_name', '이름 없음')}")
                
            else:
                # 3. 사용자에게 질문
                result = ExplorationResult(
                    question=question,
                    answer=None,
                    source=ExplorationSource.USER_DIALOGUE,
                    concept_name=None,
                    confidence=0.0,
                )
                
                self.pending_questions.append({
                    "question": question,
                    "context": context[:200],
                    "wave": wave_signature,
                })
                logger.info(f"❓ 사용자에게 질문 예정: {question}")
        
        self.exploration_history.append(result)
        return result
    
    def _search_local(self, wave_signature: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """내부 지식베이스에서 유사 패턴 검색"""
        best_match = None
        best_score = 0.0
        
        for name, data in self.knowledge_base.items():
            pattern = data.get("wave_pattern", {})
            score = self._pattern_similarity(wave_signature, pattern)
            
            if score > best_score and score > 0.5:  # 임계값
                best_score = score
                best_match = {"name": name, **data}
        
        return best_match
    
    def _pattern_similarity(self, p1: Dict[str, float], p2: Dict[str, float]) -> float:
        """패턴 유사도"""
        common = set(p1.keys()) & set(p2.keys())
        if not common:
            return 0.0
        
        total_diff = sum(abs(p1[k] - p2[k]) for k in common)
        return max(0, 1 - total_diff / len(common))
    
    def _simulate_web_search(
        self, 
        question: str, 
        wave_signature: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """웹 검색 시뮬레이션
        
        실제로는 search_web 도구 사용
        지금은 패턴 기반 추론
        """
        # 패턴 분석 기반 추론
        if wave_signature.get("tension", 0) > 0.5 and wave_signature.get("release", 0) > 0.5:
            return {
                "answer": "긴장과 해소의 순환 구조 (Tension-Release Cycle)",
                "concept_name": "긴장-해소 구조",
                "source": "narrative_theory",
            }
        
        if wave_signature.get("dissonance", 0) > 0.5:
            return {
                "answer": "대비를 통한 의미 강조 (Contrast Effect)",
                "concept_name": "대비 효과",
                "source": "narrative_theory",
            }
        
        if wave_signature.get("flow", 0) > 0.6:
            return {
                "answer": "리듬감 있는 전개 (Rhythmic Pacing)",
                "concept_name": "리듬적 전개",
                "source": "narrative_theory",
            }
        
        return None
    
    def _crystallize(
        self,
        name: str,
        definition: str,
        wave_signature: Dict[str, float],
        source: str,
    ):
        """개념 결정화 (이름 붙이기)
        
        몽글몽글한 파동 → 명확한 개념 노드
        """
        concept = CrystallizedConcept(
            name=name,
            definition=definition,
            wave_signature=wave_signature.copy(),
            source=source,
            confidence=0.7,
        )
        
        self.crystallized_concepts[name] = concept
        logger.info(f"💎 개념 결정화: '{name}' ← {source}")
    
    def answer_from_user(self, question: str, user_answer: str, concept_name: str):
        """사용자로부터 답을 받음
        
        "아빠, 그건 '사탕'이야"
        """
        # 해당 질문 찾기
        for pending in self.pending_questions:
            if pending["question"] == question:
                # 결정화
                self._crystallize(
                    name=concept_name,
                    definition=user_answer,
                    wave_signature=pending["wave"],
                    source="user_dialogue",
                )
                
                self.pending_questions.remove(pending)
                logger.info(f"🙏 사용자에게 배움: '{concept_name}'")
                break
    
    def get_pending_questions(self) -> List[Dict[str, Any]]:
        """사용자에게 물어볼 질문 목록"""
        return self.pending_questions
    
    def get_crystallized_concepts(self) -> List[Dict[str, Any]]:
        """결정화된 개념 목록"""
        return [
            {
                "name": c.name,
                "definition": c.definition,
                "confidence": c.confidence,
                "source": c.source,
            }
            for c in self.crystallized_concepts.values()
        ]


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🔍 External Explorer Demo")
    print("   \"모르는 것을 외부에서 찾아온다\"")
    print("=" * 60)
    
    explorer = ExternalExplorer()
    
    # 탐구 1: 긴장-해소 패턴
    print("\n[1] 긴장-해소 패턴 탐구:")
    result1 = explorer.explore(
        question="왜 이 긴장이 해소될 때 감동적인가?",
        wave_signature={"tension": 0.7, "release": 0.8, "brightness": 0.6},
        context="마침내 현자는 울었다"
    )
    print(f"   답: {result1.answer}")
    print(f"   개념: {result1.concept_name}")
    
    # 탐구 2: 대비 패턴
    print("\n[2] 대비 패턴 탐구:")
    result2 = explorer.explore(
        question="이 대비는 왜 효과적인가?",
        wave_signature={"dissonance": 0.7, "brightness": 0.3},
        context="빛과 어둠이 공존했다"
    )
    print(f"   답: {result2.answer}")
    print(f"   개념: {result2.concept_name}")
    
    # 탐구 3: 모르는 패턴 (사용자에게 질문)
    print("\n[3] 미지의 패턴:")
    result3 = explorer.explore(
        question="이 리듬은 무엇을 의미하는가?",
        wave_signature={"flow": 0.9, "brightness": 0.5},
        context="파도처럼 밀려왔다 밀려갔다"
    )
    
    if result3.answer is None:
        print("   → 사용자에게 질문 필요!")
        
        # 사용자 응답 시뮬레이션
        print("\n[4] 사용자가 가르쳐줌:")
        explorer.answer_from_user(
            question="이 리듬은 무엇을 의미하는가?",
            user_answer="문장의 호흡이 살아있는 리듬적 전개",
            concept_name="리듬적 전개"
        )
    
    # 결과
    print("\n" + "=" * 60)
    print("📊 결정화된 개념들:")
    for concept in explorer.get_crystallized_concepts():
        print(f"   💎 {concept['name']}: {concept['definition'][:40]}...")
    
    print("\n✅ Demo complete!")

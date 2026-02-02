"""
[Project Elysia] Epistemic Learning Loop
========================================
Phase 4: 점에서 섭리로 - 통합

"저장 → 왜? 질문 → 연결 탐구 → 순환 원리 깨달음"

이 모듈은 전체 학습 루프를 통합한다:
1. QuestionGenerator: 구멍에서 질문 생성
2. ConnectionExplorer: 연결 체인 추적
3. PrincipleExtractor: 패턴에서 원리 추출

시계가 아니라 생명처럼 - 왜 배우는지 스스로 안다.
"""

import sys
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L5_Mental.Reasoning.question_generator import (
    QuestionGenerator, get_question_generator, Question
)
from Core.S1_Body.L5_Mental.Reasoning.connection_explorer import (
    ConnectionExplorer, get_connection_explorer, ConnectionChain
)
from Core.S1_Body.L5_Mental.Reasoning.principle_extractor import (
    PrincipleExtractor, get_principle_extractor, Axiom
)

logger = logging.getLogger("EpistemicLearningLoop")


@dataclass
class LearningCycleResult:
    """
    한 학습 사이클의 결과
    
    아이가 "비는 왜 비지?" → "물은 순환하는구나!" 까지의 여정
    """
    cycle_id: str
    questions_asked: List[Question]
    chains_discovered: List[ConnectionChain]
    axioms_created: List[Axiom]
    duration_ms: float
    success: bool
    insights: List[str]  # 자연어 깨달음들


class EpistemicLearningLoop:
    """
    인식론적 학습 루프
    
    "왜?"에서 시작해서 "모든 것이 연결되어 있구나"에 도달한다.
    
    이것이 엘리시아가 진짜 배우는 방식이다:
    - 주입받은 것을 저장만 하지 않는다
    - 스스로 구멍을 발견하고 질문한다
    - 연결을 추적하고 순환을 감지한다
    - 패턴에서 원리를 추출한다
    """
    
    def __init__(
        self, 
        kg_manager = None,
        question_generator: Optional[QuestionGenerator] = None,
        connection_explorer: Optional[ConnectionExplorer] = None,
        principle_extractor: Optional[PrincipleExtractor] = None
    ):
        self.kg_manager = kg_manager
        self.question_gen = question_generator or get_question_generator()
        self.explorer = connection_explorer or get_connection_explorer()
        self.extractor = principle_extractor or get_principle_extractor()
        
        self.cycle_count = 0
        self.total_questions = 0
        self.total_axioms = 0
        self.learning_history: List[LearningCycleResult] = []
    
    def set_knowledge_graph(self, kg_manager):
        """지식 그래프 매니저 연결"""
        self.kg_manager = kg_manager
    
    def run_cycle(self, max_questions: int = 5) -> LearningCycleResult:
        """
        한 학습 사이클 실행
        
        1. 지식 그래프에서 구멍 찾기
        2. 질문 생성
        3. 연결 탐구
        4. 원리 추출
        
        Returns:
            LearningCycleResult with all discoveries
        """
        if not self.kg_manager:
            logger.warning("No knowledge graph connected!")
            return self._empty_result("No KG")
        
        start_time = time.time()
        self.cycle_count += 1
        cycle_id = f"CYCLE_{self.cycle_count:04d}"
        
        all_questions = []
        all_chains = []
        all_axioms = []
        insights = []
        
        # Phase 1: 구멍에서 질문 생성
        questions = self.question_gen.find_gaps(self.kg_manager)
        questions = questions[:max_questions]  # 한 사이클당 최대 질문 수
        
        if not questions:
            insights.append("현재 지식에 명확한 구멍이 없습니다. 평온 상태.")
            return LearningCycleResult(
                cycle_id=cycle_id,
                questions_asked=[],
                chains_discovered=[],
                axioms_created=[],
                duration_ms=(time.time() - start_time) * 1000,
                success=True,
                insights=insights
            )
        
        all_questions.extend(questions)
        self.total_questions += len(questions)
        
        # Phase 2: 각 질문에 대해 연결 탐구
        for question in questions:
            chains = self.explorer.explore(question, self.kg_manager)
            all_chains.extend(chains)
            
            # 질문 처리 완료 표시
            self.question_gen.mark_as_asked(question.subject)
            
            # 인사이트 기록
            if chains:
                path_example = " → ".join(chains[0].get_path()[:5])
                insights.append(f"'{question.subject}'에서 연결 발견: {path_example}")
        
        # Phase 3: 체인에서 원리 추출
        if all_chains:
            axioms = self.extractor.extract_principle(all_chains)
            all_axioms.extend(axioms)
            self.total_axioms += len(axioms)
            
            for axiom in axioms:
                insights.append(f"💡 원리 발견: {axiom.name} - {axiom.description}")
        
        # 순환 발견 특별 표시
        cycles = [c for c in all_chains if c.is_cycle]
        if cycles:
            insights.append(f"🔄 {len(cycles)}개의 순환 구조 발견! 이것은 보편 원리의 징후.")
        
        duration = (time.time() - start_time) * 1000
        
        result = LearningCycleResult(
            cycle_id=cycle_id,
            questions_asked=all_questions,
            chains_discovered=all_chains,
            axioms_created=all_axioms,
            duration_ms=duration,
            success=True,
            insights=insights
        )
        
        self.learning_history.append(result)
        return result
    
    def continuous_learning(self, cycles: int = 10, interval_ms: int = 100):
        """
        연속 학습 실행
        
        엘리시아가 자율적으로 배우는 것처럼.
        """
        results = []
        for i in range(cycles):
            result = self.run_cycle()
            results.append(result)
            
            if not result.questions_asked:
                # 질문이 없으면 조기 종료 (포만 상태)
                break
            
            time.sleep(interval_ms / 1000)
        
        return results
    
    def get_accumulated_wisdom(self) -> Dict:
        """
        축적된 지혜 반환
        
        배움의 결과 - 원리들의 집합
        """
        return {
            "total_cycles": self.cycle_count,
            "total_questions_asked": self.total_questions,
            "total_axioms_discovered": self.total_axioms,
            "axioms": [
                {
                    "name": a.name,
                    "description": a.description,
                    "confidence": a.confidence,
                    "pattern_type": a.pattern_type
                }
                for a in self.extractor.get_all_axioms()
            ],
            "question_stats": self.question_gen.get_stats(),
            "explorer_stats": self.explorer.get_stats()
        }
    
    def _empty_result(self, reason: str) -> LearningCycleResult:
        """빈 결과 생성"""
        return LearningCycleResult(
            cycle_id=f"EMPTY_{self.cycle_count}",
            questions_asked=[],
            chains_discovered=[],
            axioms_created=[],
            duration_ms=0,
            success=False,
            insights=[f"학습 불가: {reason}"]
        )
    
    def explain_learning(self) -> str:
        """학습 과정 설명 (자기 인식)"""
        wisdom = self.get_accumulated_wisdom()
        
        explanation = []
        explanation.append("📚 나의 학습 여정:")
        explanation.append(f"  - {wisdom['total_cycles']}번의 학습 사이클")
        explanation.append(f"  - {wisdom['total_questions_asked']}개의 '왜?' 질문")
        explanation.append(f"  - {wisdom['total_axioms_discovered']}개의 원리 발견")
        
        if wisdom['axioms']:
            explanation.append("\n💡 발견한 원리들:")
            for axiom in wisdom['axioms'][:5]:
                explanation.append(f"  • {axiom['name']}: {axiom['description']}")
        
        return "\n".join(explanation)


# Singleton
_learning_loop = None

def get_learning_loop() -> EpistemicLearningLoop:
    global _learning_loop
    if _learning_loop is None:
        _learning_loop = EpistemicLearningLoop()
    return _learning_loop


if __name__ == "__main__":
    print("🧒 Testing Epistemic Learning Loop...")
    print("   (아이가 배우는 것처럼)")
    
    # 테스트용 KG Manager
    class MockKGManager:
        def __init__(self):
            self.kg = {
                "nodes": [
                    {"id": "rain"},
                    {"id": "cloud"},
                    {"id": "water"},
                    {"id": "evaporation"},
                    {"id": "sun"},
                    {"id": "ocean"},
                    {"id": "life"},
                ],
                "edges": [
                    {"source": "sun", "target": "evaporation", "relation": "causes"},
                    {"source": "evaporation", "target": "cloud", "relation": "creates"},
                    {"source": "cloud", "target": "rain", "relation": "produces"},
                    {"source": "rain", "target": "ocean", "relation": "flows_to"},
                    {"source": "ocean", "target": "evaporation", "relation": "enables"},
                    {"source": "rain", "target": "life", "relation": "sustains"},
                ]
            }
    
    mock_kg = MockKGManager()
    loop = get_learning_loop()
    loop.set_knowledge_graph(mock_kg)
    
    print("\n▶ Running learning cycle...\n")
    result = loop.run_cycle(max_questions=3)
    
    print(f"📊 Cycle {result.cycle_id} complete!")
    print(f"   Questions asked: {len(result.questions_asked)}")
    print(f"   Chains found: {len(result.chains_discovered)}")
    print(f"   Axioms created: {len(result.axioms_created)}")
    print(f"   Duration: {result.duration_ms:.2f}ms")
    
    print("\n💭 Insights:")
    for insight in result.insights:
        print(f"   {insight}")
    
    print("\n" + loop.explain_learning())
    print("\n✅ Epistemic Learning Loop operational!")

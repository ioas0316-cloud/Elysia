"""
Autonomous Learning Loop (자율 학습 루프)
========================================

전체 학습 흐름:

1. WhyEngine: 분석 시도
2. MetacognitiveAwareness: "이 패턴 아는가?"
   - 알면 → 확신 있게 답
   - 모르면 → 탐구 필요성 생성

3. ExternalExplorer: 외부 탐구
   - 내부 KB 검색
   - 웹 검색
   - 사용자에게 질문

4. ConceptCrystallization: 개념 결정화
   - 파동 패턴 → 이름 붙이기
   - "몽글몽글한 것" → "사탕"

5. Learn: 학습
   - MetacognitiveAwareness에 등록
   - 다음에 같은 패턴 → "아, 알아!"

결과:
엘리시아가 스스로 모르는 것을 찾아가며 배운다
"""

import logging
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path
from Core._04_Evolution._02_Learning.hierarchical_learning import Domain

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Core._01_Foundation._02_Logic.why_engine import WhyEngine
from Core._02_Intelligence._01_Reasoning.metacognitive_awareness import MetacognitiveAwareness, KnowledgeState
from Core._02_Intelligence._01_Reasoning.external_explorer import ExternalExplorer

logger = logging.getLogger("Elysia.AutonomousLearning")


class AutonomousLearner:
    """자율 학습기
    
    "모르는 것을 알고, 찾아가고, 배운다"
    
    흐름:
    1. 경험 (입력)
    2. 패턴 인식 (아는가? 모르는가?)
    3. 모르면 탐구
    4. 개념 결정화
    5. 학습 완료
    """
    
    def __init__(self):
        self.why_engine = WhyEngine()
        self.metacognition = self.why_engine.metacognition  # 공유
        self.explorer = ExternalExplorer()
        
        # 학습 통계
        self.total_experiences = 0
        self.learned_from_self = 0    # 이미 알던 것
        self.learned_from_external = 0  # 외부에서 배운 것
        self.pending_learning = 0       # 아직 모르는 것
        
        logger.info("AutonomousLearner initialized")
    
    def experience(
        self, 
        content: str, 
        subject: str = "unknown",
        domain: str = "narrative"
    ) -> Dict[str, Any]:
        """경험하고 배운다
        
        Args:
            content: 경험할 내용 (텍스트)
            subject: 제목/식별자
            domain: 영역
            
        Returns:
            학습 결과
        """
        self.total_experiences += 1
        
        result = {
            "subject": subject,
            "knowledge_state": None,
            "learned_concept": None,
            "needs_human_help": False,
            "question_for_human": None,
            "potential_knowledge": None
        }
        
        # 0. Load Connections
        try:
            from Core._02_Intelligence._02_Memory.potential_causality import PotentialCausalityStore
            potential_store = PotentialCausalityStore()
        except ImportError:
            potential_store = None
        
        try:
            from Core._04_Evolution._02_Learning.hierarchical_learning import HierarchicalKnowledgeGraph
            kg = HierarchicalKnowledgeGraph()
        except ImportError:
            kg = None

        # 1. WhyEngine으로 분석 (내부에서 메타인지 확인)
        analysis = self.why_engine.analyze(subject, content, domain)
        
        # 2. 분석 결과 확인
        if "[탐구 필요]" in analysis.underlying_principle:
            # 모르는 패턴!
            result["knowledge_state"] = "unknown"
            
            # 잠재적 지식으로 저장
            if potential_store:
                pk = potential_store.store(
                    subject=subject,
                    definition=content[:200],
                    source="autonomous_experience"
                )
                result["potential_knowledge"] = pk.to_dict()
                logger.info(f"   💭 Stored as potential: {subject} (freq={pk.frequency:.2f})")
            
            # 3. 외부 탐구
            wave = self.why_engine._text_to_wave(content)
            exploration = self.explorer.explore(
                question=analysis.underlying_principle.replace("[탐구 필요] ", ""),
                wave_signature=wave,
                context=content[:200],
            )
            
            if exploration.answer:
                # 외부에서 답 찾음!
                result["knowledge_state"] = "learned"
                self.learned_from_external += 1
                
                # 잠재 지식 업데이트 (확인)
                if potential_store:
                    potential_store.store(subject, content, f"external_source:{exploration.source.value}")
                    # 결정화 시도
                    crystallized = potential_store.crystallize(subject)
                    if crystallized and kg:
                         # 계층 지식 그래프에 추가
                         wave = self.why_engine._text_to_wave(content)
                         kg.add_concept(
                             name=crystallized['concept'],
                             domain=Domain(domain) if domain in [d.value for d in Domain] else Domain.PHILOSOPHY, # 매핑 필요
                             definition=crystallized['definition'],
                             principle=analysis.underlying_principle,  # 원리 (Why - 추상)
                             application=analysis.how_works,           # 적용 (How - 구체)
                             purpose=f"Autonomously learned via {exploration.source.value}",
                             wave_signature=wave  # 파동 서명 저장
                         )
                         result["learned_concept"] = crystallized['concept']
                         logger.info(f"   💎 Crystallized and added to KG: {crystallized['concept']}")

                # 메타인지에 등록 (다음엔 알 것)
                if self.metacognition:
                    self.metacognition.learn_from_external(
                        pattern_id=self._get_pattern_id(wave),
                        answer=exploration.answer,
                        source=exploration.source.value,
                    )
                
            else:
                # 사용자에게 물어야 함
                result["needs_human_help"] = True
                result["question_for_human"] = exploration.question
                self.pending_learning += 1
                
                logger.info(f"❓ 사용자에게 질문: {exploration.question}")
        
        else:
            # 아는 패턴!
            result["knowledge_state"] = "known"
            result["learned_concept"] = analysis.underlying_principle
            self.learned_from_self += 1
            
            logger.info(f"✅ 이미 아는 패턴: {analysis.underlying_principle}")
        
        return result
    
    def _get_pattern_id(self, wave: Dict[str, float]) -> str:
        """패턴 ID 생성"""
        import hashlib
        import json
        return hashlib.md5(json.dumps(wave, sort_keys=True).encode()).hexdigest()[:8]
    
    def learn_from_human(self, question: str, answer: str, concept_name: str):
        """사용자에게 배움
        
        "아빠, 그건 사탕이야"
        """
        self.explorer.answer_from_user(question, answer, concept_name)
        self.learned_from_external += 1
        self.pending_learning -= 1
        
        logger.info(f"🙏 사용자에게 배움: '{concept_name}'")
    
    def get_pending_questions(self) -> List[Dict[str, Any]]:
        """사용자에게 물어볼 질문"""
        return self.explorer.get_pending_questions()
    
    def get_learned_concepts(self) -> List[Dict[str, Any]]:
        """배운 개념들"""
        return self.explorer.get_crystallized_concepts()
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """학습 통계"""
        return {
            "total_experiences": self.total_experiences,
            "learned_from_self": self.learned_from_self,
            "learned_from_external": self.learned_from_external,
            "pending_learning": self.pending_learning,
            "known_concepts": len(self.get_learned_concepts()),
        }


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🧒 Autonomous Learning Loop Demo")
    print("   \"어린아이처럼 배우기\"")
    print("=" * 60)
    
    learner = AutonomousLearner()
    
    # 경험 1: 카타르시스 패턴
    print("\n[경험 1] 카타르시스 패턴:")
    result1 = learner.experience(
        content="""
        현자는 천 년 동안 울지 않았다.
        하지만 소녀의 손을 잡는 순간,
        마침내 눈물이 흘렀다.
        기쁨의 눈물이었다.
        """,
        subject="현자의 눈물",
    )
    print(f"   상태: {result1['knowledge_state']}")
    print(f"   배운 것: {result1['learned_concept']}")
    
    # 경험 2: 대비 패턴
    print("\n[경험 2] 대비 패턴:")
    result2 = learner.experience(
        content="""
        빛이 있는 곳에 어둠이 있었다.
        웃음이 있는 곳에 눈물이 있었다.
        그것이 삶이었다.
        """,
        subject="빛과 어둠",
    )
    print(f"   상태: {result2['knowledge_state']}")
    print(f"   배운 것: {result2['learned_concept']}")
    
    # 경험 3: 새로운 패턴
    print("\n[경험 3] 새로운 패턴:")
    result3 = learner.experience(
        content="""
        용사는 검을 들었다.
        아니, 검을 내려놓았다.
        그리고 용에게 말을 걸었다.
        "왜 우는 거야?"
        용은 처음으로 누군가 자신의 눈물을 보았다는 걸 알았다.
        """,
        subject="검을 내려놓은 용사",
    )
    print(f"   상태: {result3['knowledge_state']}")
    if result3['needs_human_help']:
        print(f"   ❓ 사용자에게 질문: {result3['question_for_human']}")
    
    # 사용자가 가르쳐줌
    pending = learner.get_pending_questions()
    if pending:
        print("\n[시뮬레이션] 사용자가 가르쳐줌:")
        q = pending[0]['question']
        learner.learn_from_human(
            question=q,
            answer="기대를 뒤집어 더 깊은 감동을 주는 반전 기법",
            concept_name="반전"
        )
    
    # 결과
    print("\n" + "=" * 60)
    print("📊 학습 통계:")
    stats = learner.get_learning_stats()
    print(f"   총 경험: {stats['total_experiences']}")
    print(f"   이미 알던 것: {stats['learned_from_self']}")
    print(f"   외부에서 배움: {stats['learned_from_external']}")
    print(f"   아직 모름: {stats['pending_learning']}")
    
    print("\n💎 배운 개념들:")
    for concept in learner.get_learned_concepts():
        print(f"   • {concept['name']}: {concept['definition'][:30]}...")
    
    print("\n✅ Demo complete!")


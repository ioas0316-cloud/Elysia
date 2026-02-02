"""
[Project Elysia] Question Generator
===================================
Phase 1: 점에서 섭리로

"아이는 처음에 저장만 해. 나중에 '왜?'라고 물으면서 모든 연결을 배워가는 거야."

이 모듈은 저장된 지식에서 "왜?" 연결이 없는 구멍을 찾아 질문을 생성한다.
하드코딩된 규칙이 아니라, 연결의 부재 자체가 질문을 일으킨다.
"""

import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum
import time

root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)


class QuestionType(Enum):
    """질문의 유형"""
    WHY = "왜?"           # 인과 연결 없음
    HOW = "어떻게?"       # 과정 연결 없음
    WHAT_IS = "무엇?"     # 정의 연결 없음
    CONNECTS_TO = "연결?" # 관계 연결 없음


@dataclass
class Question:
    """
    생성된 질문
    
    질문은 지식 그래프의 "구멍"에서 자연스럽게 발생한다.
    구멍 = 연결이 있어야 하는데 없는 곳
    """
    question_id: str
    question_type: QuestionType
    subject: str              # 질문의 주체 (예: "rain")
    missing_link: str         # 누락된 연결 유형 (예: "CAUSES")
    context_nodes: List[str]  # 주변 문맥 노드들
    urgency: float = 0.5      # 0.0 ~ 1.0, 높을수록 긴급
    timestamp: float = field(default_factory=time.time)
    
    def to_natural_language(self) -> str:
        """자연어 질문으로 변환"""
        templates = {
            QuestionType.WHY: f"왜 {self.subject}은/는 그런가?",
            QuestionType.HOW: f"{self.subject}은/는 어떻게 되는가?",
            QuestionType.WHAT_IS: f"{self.subject}이란 무엇인가?",
            QuestionType.CONNECTS_TO: f"{self.subject}은/는 무엇과 연결되는가?",
        }
        return templates.get(self.question_type, f"{self.subject}에 대해 알고 싶다")


class QuestionGenerator:
    """
    질문 생성기
    
    지식 그래프에서 "왜?" 연결이 없는 구멍을 찾아
    자연스럽게 질문을 생성한다.
    
    핵심 원리:
    - 아이가 "비는 왜 하늘에서 와?"라고 묻는 것처럼
    - 저장된 사실에 인과/과정/정의 연결이 없으면 질문 발생
    """
    
    # 인과 관계를 나타내는 엣지 타입들
    CAUSAL_RELATIONS = {"causes", "leads_to", "results_in", "because", "why"}
    PROCESS_RELATIONS = {"how", "through", "via", "by_means_of"}
    DEFINITION_RELATIONS = {"is_a", "defined_as", "means"}
    
    def __init__(self):
        self.generated_questions: List[Question] = []
        self.asked_subjects: Set[str] = set()  # 이미 질문한 주제 (무한 루프 방지)
        self.question_counter = 0
    
    def find_gaps(self, kg_manager) -> List[Question]:
        """
        지식 그래프에서 구멍(gap)을 찾아 질문 생성
        
        구멍의 정의:
        1. WHY 구멍: 노드가 있지만 "왜 그런지" 연결이 없음
        2. HOW 구멍: 노드가 있지만 "어떻게 되는지" 연결이 없음
        3. WHAT 구멍: 노드가 있지만 정의 연결이 없음
        """
        questions = []
        
        nodes = kg_manager.kg.get("nodes", [])
        edges = kg_manager.kg.get("edges", [])
        
        # 각 노드의 연결 상태 분석
        node_connections = self._analyze_connections(nodes, edges)
        
        for node in nodes:
            node_id = node.get("id", "")
            
            # 이미 질문한 주제는 건너뜀 (포만 메커니즘)
            if node_id in self.asked_subjects:
                continue
            
            connections = node_connections.get(node_id, {
                "has_causal": False,
                "has_process": False,
                "has_definition": False,
                "neighbors": []
            })
            
            # WHY 구멍 감지
            if not connections["has_causal"]:
                q = self._create_question(
                    QuestionType.WHY,
                    node_id,
                    "CAUSES",
                    connections["neighbors"]
                )
                questions.append(q)
            
            # HOW 구멍 감지
            if not connections["has_process"]:
                q = self._create_question(
                    QuestionType.HOW,
                    node_id,
                    "PROCESS",
                    connections["neighbors"]
                )
                questions.append(q)
            
            # WHAT 구멍 감지 (정의 없는 노드)
            if not connections["has_definition"]:
                q = self._create_question(
                    QuestionType.WHAT_IS,
                    node_id,
                    "DEFINITION",
                    connections["neighbors"]
                )
                questions.append(q)
        
        # 긴급도 순으로 정렬
        questions.sort(key=lambda q: q.urgency, reverse=True)
        
        self.generated_questions.extend(questions)
        return questions
    
    def _analyze_connections(self, nodes: List[Dict], edges: List[Dict]) -> Dict:
        """각 노드의 연결 상태 분석"""
        result = {}
        
        for node in nodes:
            node_id = node.get("id", "")
            result[node_id] = {
                "has_causal": False,
                "has_process": False,
                "has_definition": False,
                "neighbors": []
            }
        
        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            relation = edge.get("relation", "").lower()
            
            # 연결 유형 분류
            if relation in self.CAUSAL_RELATIONS:
                if source in result:
                    result[source]["has_causal"] = True
                if target in result:
                    result[target]["has_causal"] = True
            
            if relation in self.PROCESS_RELATIONS:
                if source in result:
                    result[source]["has_process"] = True
            
            if relation in self.DEFINITION_RELATIONS:
                if target in result:
                    result[target]["has_definition"] = True
            
            # 이웃 노드 기록
            if source in result:
                result[source]["neighbors"].append(target)
            if target in result:
                result[target]["neighbors"].append(source)
        
        return result
    
    def _create_question(
        self, 
        q_type: QuestionType, 
        subject: str, 
        missing: str,
        context: List[str]
    ) -> Question:
        """질문 객체 생성"""
        self.question_counter += 1
        
        # 긴급도 계산: 연결이 전혀 없는 노드일수록 긴급
        urgency = 0.5
        if len(context) == 0:
            urgency = 0.9  # 고아 노드 = 매우 긴급
        elif len(context) < 3:
            urgency = 0.7  # 연결 부족 = 긴급
        
        return Question(
            question_id=f"Q_{self.question_counter:04d}",
            question_type=q_type,
            subject=subject,
            missing_link=missing,
            context_nodes=context[:5],  # 최대 5개
            urgency=urgency
        )
    
    def mark_as_asked(self, subject: str):
        """질문한 주제 기록 (포만 메커니즘)"""
        self.asked_subjects.add(subject)
    
    def get_most_urgent(self) -> Optional[Question]:
        """가장 긴급한 질문 반환"""
        pending = [q for q in self.generated_questions 
                   if q.subject not in self.asked_subjects]
        if pending:
            return max(pending, key=lambda q: q.urgency)
        return None
    
    def get_stats(self) -> Dict:
        """통계 반환"""
        return {
            "total_questions": len(self.generated_questions),
            "asked_count": len(self.asked_subjects),
            "pending_count": len([q for q in self.generated_questions 
                                  if q.subject not in self.asked_subjects])
        }


# Singleton
_question_generator = None

def get_question_generator() -> QuestionGenerator:
    global _question_generator
    if _question_generator is None:
        _question_generator = QuestionGenerator()
    return _question_generator


if __name__ == "__main__":
    print("🤔 Testing Question Generator...")
    
    # 테스트용 가짜 KG Manager
    class MockKGManager:
        def __init__(self):
            self.kg = {
                "nodes": [
                    {"id": "rain"},
                    {"id": "sky"},
                    {"id": "water"},
                    {"id": "cloud"},
                ],
                "edges": [
                    {"source": "rain", "target": "sky", "relation": "comes_from"},
                    # rain에 "왜?" 연결이 없음 - 질문 발생해야 함
                ]
            }
    
    mock_kg = MockKGManager()
    generator = get_question_generator()
    
    questions = generator.find_gaps(mock_kg)
    
    print(f"\n📊 Generated {len(questions)} questions:")
    for q in questions[:5]:
        print(f"  [{q.question_type.value}] {q.to_natural_language()} (urgency: {q.urgency:.2f})")
    
    print(f"\n✅ Question Generator operational!")
    print(f"   Stats: {generator.get_stats()}")

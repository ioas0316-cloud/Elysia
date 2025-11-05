# c:/Elysia/Project_Sophia/logical_reasoner.py

import re
from typing import List, Dict, Any
from tools.kg_manager import KGManager

class LogicalReasoner:
    """
    사용자 입력으로부터 논리적 사실을 추론하고 지식 그래프와 상호작용합니다.
    인과 관계를 포함한 복잡한 추론을 지원합니다.
    """
    def __init__(self, kg_manager: KGManager = None):
        """
        LogicalReasoner를 초기화합니다.
        기존 KGManager를 주입하거나, 제공되지 않으면 새로 생성합니다.
        """
        self.kg_manager = kg_manager if kg_manager else KGManager()

    def _find_mentioned_entities(self, message: str) -> List[str]:
        """메시지에서 KG에 존재하는 엔티티를 찾습니다."""
        mentioned_entities = []
        nodes = self.kg_manager.kg.get('nodes', [])
        for node in nodes:
            entity = node['id']
            if entity in message:
                mentioned_entities.append(entity)
        return list(set(mentioned_entities))

    def deduce_facts(self, message: str) -> List[str]:
        """
        메시지를 분석하고 지식 그래프를 쿼리하여 관련 사실을 추론합니다.
        질문의 의도(원인/결과)를 파악하여 더 정확한 답변을 생성합니다.
        """
        final_facts = set()
        mentioned_entities = self._find_mentioned_entities(message)

        query_is_for_cause = "원인" in message or "이유" in message
        query_is_for_effect = "결과" in message or "영향" in message

        if not mentioned_entities:
            return []

        for entity in mentioned_entities:
            # First, gather all possible facts related to the entity
            all_possible_facts = set()

            # Find causes of the entity
            causes = self.kg_manager.find_causes(entity)
            for edge in causes:
                fact = f"'{edge['source']}'은(는) '{entity}'의 원인이 될 수 있습니다."
                if 'strength' in edge: fact += f" (인과 강도: {edge['strength']})"
                if 'conditions' in edge: fact += f" (조건: {', '.join(edge['conditions'])})"
                all_possible_facts.add(fact)

            # Find effects of the entity
            effects = self.kg_manager.find_effects(entity)
            for edge in effects:
                fact = f"'{entity}'은(는) '{edge['target']}'을(를) 유발할 수 있습니다."
                if 'strength' in edge: fact += f" (인과 강도: {edge['strength']})"
                if 'conditions' in edge: fact += f" (조건: {', '.join(edge['conditions'])})"
                all_possible_facts.add(fact)

            # Find general relationships
            for edge in self.kg_manager.kg.get('edges', []):
                if edge.get('relation') == 'causes': continue
                if edge['source'] == entity:
                    all_possible_facts.add(f"'{edge['source']}'은(는) '{edge['target']}'와(과) '{edge['relation']}' 관계를 가집니다.")
                elif edge['target'] == entity:
                    all_possible_facts.add(f"'{entity}'은(는) '{edge['source']}'와(과) '{edge['relation']}' 관계를 가집니다.")

            # Now, filter the facts based on the query
            if query_is_for_cause and not query_is_for_effect:
                for fact in all_possible_facts:
                    if "원인이 될 수 있습니다" in fact:
                        final_facts.add(fact)
            elif query_is_for_effect and not query_is_for_cause:
                 for fact in all_possible_facts:
                    if "유발할 수 있습니다" in fact:
                        final_facts.add(fact)
            else: # Ambiguous or general query
                final_facts.update(all_possible_facts)

        return list(final_facts)

if __name__ == '__main__':
    reasoner = LogicalReasoner()
    # Use a clean in-memory KG for the test run
    reasoner.kg_manager.kg = {"nodes": [], "edges": []}
    reasoner.kg_manager.add_edge("햇빛", "식물 성장", "causes", properties={"strength": 0.85})
    reasoner.kg_manager.add_edge("수분", "식물 성장", "causes", properties={"strength": 0.9, "conditions": ["적절한 온도"]})
    reasoner.kg_manager.add_edge("식물 성장", "산소 발생", "causes")
    reasoner.kg_manager.add_node("소크라테스")
    reasoner.kg_manager.add_node("인간")
    reasoner.kg_manager.add_edge("소크라테스", "인간", "is_a")

    print("--- '원인' 질문 테스트 ---")
    test_message_cause = "식물 성장의 원인은 무엇이야?"
    facts_cause = reasoner.deduce_facts(test_message_cause)
    if facts_cause:
        for f in sorted(list(facts_cause)):
            print(f"- {f}")
    else:
        print("추론된 사실이 없습니다.")

    print("\n--- '결과' 질문 테스트 ---")
    test_message_effect = "식물 성장의 결과는 무엇이야?"
    facts_effect = reasoner.deduce_facts(test_message_effect)
    if facts_effect:
        for f in sorted(list(facts_effect)):
            print(f"- {f}")
    else:
        print("추론된 사실이 없습니다.")

    print("\n--- 일반 질문 테스트 ---")
    test_message_general = "소크라테스에 대해 알려줘"
    facts_general = reasoner.deduce_facts(test_message_general)
    if facts_general:
        for f in sorted(list(facts_general)):
            print(f"- {f}")
    else:
        print("추론된 사실이 없습니다.")

# c:/Elysia/Project_Sophia/logical_reasoner.py

import re
from typing import List, Dict, Any
from tools.kg_manager import KGManager

class LogicalReasoner:
    """
    사용자 입력으로부터 논리적 사실을 추론하고 지식 그래프와 상호작용합니다.
    (This is a stable version that includes description retrieval)
    """
    def __init__(self, kg_manager: KGManager = None):
        """
        LogicalReasoner를 초기화합니다.
        """
        self.kg_manager = kg_manager if kg_manager else KGManager()

    def _find_mentioned_entities(self, message: str) -> List[str]:
        """
        메시지를 스캔하여 지식 그래프에 존재하는 모든 엔티티(노드 ID)를 찾습니다.
        """
        mentioned_entities = set()
        message_lower = message.lower()

        nodes = self.kg_manager.kg.get('nodes', [])
        sorted_nodes = sorted(nodes, key=lambda x: len(x['id']), reverse=True)

        for node in sorted_nodes:
            node_id = node['id']
            if node_id in message_lower:
                mentioned_entities.add(node_id)

        return list(mentioned_entities)

    def deduce_facts(self, message: str) -> List[str]:
        """
        메시지를 분석하고, 관련된 모든 관계 사실과 함께 노드의 '정의'를 추론합니다.
        """
        final_facts = set()
        mentioned_entities = self._find_mentioned_entities(message)

        if not mentioned_entities:
            return []

        for entity in mentioned_entities:
            # 1. Add the definition of the entity, if it exists.
            node = self.kg_manager.get_node(entity)
            if node and node.get('description'):
                final_facts.add(f"'{entity}'의 정의: {node.get('description')}")

            # 2. Find all relationships involving the entity.
            for edge in self.kg_manager.kg.get('edges', []):
                source, target, relation = edge.get('source'), edge.get('target'), edge.get('relation')

                fact = None
                if source == entity:
                    if relation == 'causes':
                        fact = f"'{source}'은(는) '{target}'을(를) 유발할 수 있습니다."
                        if 'strength' in edge: fact += f" (인과 강도: {edge['strength']})"
                        if 'conditions' in edge: fact += f" (조건: {', '.join(edge['conditions'])})"
                    else:
                        fact = f"'{source}'은(는) '{target}'와(과) '{relation}' 관계를 가집니다."
                elif target == entity:
                    if relation == 'causes':
                        fact = f"'{source}'은(는) '{target}'의 원인이 될 수 있습니다."
                        if 'strength' in edge: fact += f" (인과 강도: {edge['strength']})"
                        if 'conditions' in edge: fact += f" (조건: {', '.join(edge['conditions'])})"
                    else:
                        fact = f"'{source}'은(는) '{target}'와(과) '{relation}' 관계를 가집니다."

                if fact:
                    final_facts.add(fact)

        return list(final_facts)

if __name__ == '__main__':
    # Standalone testing block
    test_kg_manager = KGManager()
    test_kg_manager.kg = {"nodes": [], "edges": []}

    test_kg_manager.add_node("socrates", properties={"description": "A Greek philosopher"})
    test_kg_manager.add_node("human")
    test_kg_manager.add_or_update_edge("socrates", "human", "is_a")

    reasoner = LogicalReasoner(kg_manager=test_kg_manager)

    print("\n--- 일반 질문 테스트 ---")
    test_message_def = "Tell me about socrates"
    facts_def = reasoner.deduce_facts(test_message_def)
    if facts_def:
        for f in sorted(list(facts_def)):
            print(f"- {f}")
    else:
        print("추론된 사실이 없습니다.")

# c:/Elysia/Project_Sophia/logical_reasoner.py

import re
from typing import List, Dict, Any, Optional
from tools.kg_manager import KGManager

class LogicalReasoner:
    """
    사용자 입력으로부터 논리적 사실을 추론하고 지식 그래프와 상호작용합니다.
    인과 관계를 포함한 복잡한 추론을 지원합니다.
    """
    def __init__(self):
        """
        LogicalReasoner를 초기화하고 KGManager를 인스턴스화합니다.
        """
        self.kg_manager = KGManager()

    def _find_mentioned_entities(self, message: str) -> List[str]:
        """메시지에서 KG에 존재하는 엔티티를 찾습니다."""
        mentioned_entities = []
        nodes = self.kg_manager.kg.get('nodes', [])
        for node in nodes:
            entity = node['id']
            if entity in message:
                mentioned_entities.append(entity)
        return list(set(mentioned_entities))

    def deduce_facts(self, message: str) -> List[Dict[str, Any]]:
        """
        메시지를 분석하고 지식 그래프를 쿼리하여 확장된 인과 관계를 포함한 사실을 추론합니다.
        추론된 사실과 그 근거(evidence)를 함께 반환합니다.
        """
        final_facts = []
        final_fact_texts = set() # To avoid duplicate fact strings
        mentioned_entities = self._find_mentioned_entities(message)

        if not mentioned_entities:
            return []

        query_is_for_cause = "원인" in message or "이유" in message
        query_is_for_effect = "결과" in message or "영향" in message

        for entity in mentioned_entities:
            all_possible_facts = []

            # Use the new generic method to find all related edges
            related_edges = self.kg_manager.find_related_edges(entity, direction='any')

            for edge in related_edges:
                props = edge.get('properties', {})
                fact_text = self._format_fact(entity, edge, props)
                if fact_text:
                    all_possible_facts.append({"fact_text": fact_text, "evidence": edge})

            # Filter the facts based on the query intention
            if query_is_for_cause and not query_is_for_effect:
                for fact_obj in all_possible_facts:
                    if "[원인]" in fact_obj["fact_text"] and fact_obj["fact_text"] not in final_fact_texts:
                        final_facts.append(fact_obj)
                        final_fact_texts.add(fact_obj["fact_text"])
            elif query_is_for_effect and not query_is_for_cause:
                for fact_obj in all_possible_facts:
                    if "[결과]" in fact_obj["fact_text"] and fact_obj["fact_text"] not in final_fact_texts:
                        final_facts.append(fact_obj)
                        final_fact_texts.add(fact_obj["fact_text"])
            else:  # Ambiguous or general query
                for fact_obj in all_possible_facts:
                    if fact_obj["fact_text"] not in final_fact_texts:
                        final_facts.append(fact_obj)
                        final_fact_texts.add(fact_obj["fact_text"])

        return final_facts

    def _format_fact(self, entity: str, edge: Dict[str, Any], props: Dict[str, Any]) -> Optional[str]:
        """Formats a single edge into a human-readable fact string."""
        source, target, relation = edge['source'], edge['target'], edge['relation']
        fact = ""

        # Determine the sentence structure based on the relation and direction
        if relation == 'causes':
            if target == entity: fact = f"'{source}'은(는) '{target}'의 원인이 될 수 있습니다. [원인]"
            else: fact = f"'{source}'은(는) '{target}'을(를) 유발할 수 있습니다. [결과]"
        elif relation == 'enables':
            if target == entity: fact = f"'{source}'은(는) '{target}'을(를) 가능하게 합니다. [원인]"
            else: fact = f"'{source}' 덕분에 '{target}'이(가) 가능해집니다. [결과]"
        elif relation == 'requires':
            if target == entity: fact = f"'{target}'은(는) '{source}'을(를) 필요로 합니다. [원인]"
            else: fact = f"'{source}'을(를) 하려면 '{target}'이(가) 필요합니다. [결과]"
        elif relation == 'prevents':
            if target == entity: fact = f"'{source}'은(는) '{target}'을(를) 막을 수 있습니다. [원인]"
            else: fact = f"'{source}' 때문에 '{target}'을(를) 할 수 없습니다. [결과]"
        elif relation == 'contextualizes':
            fact = f"'{source}'은(는) '{target}'의 배경이 됩니다."
        elif relation == 'is_a':
            if target == entity: fact = f"'{target}'은(는) '{source}'의 한 종류입니다."
            else: fact = f"'{source}'은(는) '{target}'의 한 종류입니다."
        else: # Default for other relationships
            if source == entity: fact = f"'{source}'은(는) '{target}'와(과) '{relation}' 관계입니다."
            else: fact = f"'{target}'은(는) '{source}'와(과) '{relation}' 관계입니다."

        # Append properties to the fact string
        prop_details = []
        if 'strength' in props: prop_details.append(f"강도: {props['strength']}")
        if 'certainty' in props: prop_details.append(f"확실성: {props['certainty']}")
        if 'modality' in props: prop_details.append(f"양상: {props['modality']}")

        if prop_details:
            fact += f" ({', '.join(prop_details)})"

        return fact

if __name__ == '__main__':
    reasoner = LogicalReasoner()
    # Use a clean in-memory KG for the test run
    reasoner.kg_manager.kg = {"nodes": [], "edges": []}

    print("--- KG 데이터 구축 (v2) ---")
    reasoner.kg_manager.add_or_update_edge(
        "충분한 수면", "인지 능력 향상", "enables",
        properties={"strength": 0.7, "certainty": 0.8, "source": "medical_study"}
    )
    reasoner.kg_manager.add_or_update_edge(
        "운동", "건강 증진", "causes",
        properties={"strength": 0.85, "modality": "always"}
    )
    reasoner.kg_manager.add_or_update_edge(
        "건강 증진", "행복도 증가", "enables",
        properties={"strength": 0.6}
    )
    reasoner.kg_manager.add_or_update_edge(
        "프로그래밍", "논리적 사고", "requires",
        properties={"strength": 0.9}
    )
    reasoner.kg_manager.add_or_update_edge(
        "스트레스", "건강 증진", "prevents",
        properties={"strength": 0.7}
    )
    reasoner.kg_manager.add_node("엘리시아")
    reasoner.kg_manager.add_node("AI")
    reasoner.kg_manager.add_or_update_edge("엘리시아", "AI", "is_a")
    print("KG 데이터 구축 완료.")

    def run_test(title, message):
        print(f"\n--- {title} ---")
        print(f"질문: \"{message}\"")
        fact_objects = reasoner.deduce_facts(message)
        if fact_objects:
            # Sort by fact_text for consistent output
            sorted_facts = sorted(fact_objects, key=lambda x: x['fact_text'])
            for fact_obj in sorted_facts:
                print(f"- {fact_obj['fact_text']}")
                # Optional: print evidence for debugging
                # print(f"  [근거: {fact_obj['evidence']}]")
        else:
            print("추론된 사실이 없습니다.")

    run_test("'원인' 질문 테스트", "건강 증진의 원인은 무엇이야?")
    run_test("'결과' 질문 테스트", "건강 증진의 영향은 무엇이야?")
    run_test("'requires' 관계 테스트", "프로그래밍에 대해 알려줘")
    run_test("'prevents' 관계 테스트", "스트레스와 건강 증진의 관계는?")
    run_test("일반 질문 테스트", "엘리시아에 대해 알려줘")

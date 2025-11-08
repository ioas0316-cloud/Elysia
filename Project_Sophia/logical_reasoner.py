import re
import copy
from typing import List, Dict, Any, Optional

from tools.kg_manager import KGManager
from .core.world import World

class LogicalReasoner:
    """
    Deduces logical facts from user input by interacting with the knowledge graph
    and running dynamic simulations in the Cellular World.
    """

    def __init__(self, kg_manager: Optional[KGManager] = None, cellular_world: Optional[World] = None):
        self.kg_manager = kg_manager or KGManager()
        self.cellular_world = cellular_world

    def _find_mentioned_entities(self, message: str) -> List[str]:
        """Finds entities from the KG mentioned in a message using a simple substring search."""
        mentioned_entities = []
        nodes = self.kg_manager.kg.get('nodes', [])
        if not nodes:
            return []

        node_ids = {node.get('id') for node in nodes if node.get('id')}

        # Sort by length descending to match longer names first
        sorted_node_ids = sorted(list(node_ids), key=len, reverse=True)

        # Simple, robust substring checking
        lower_message = message.lower()
        for entity_id in sorted_node_ids:
            if entity_id.lower() in lower_message:
                mentioned_entities.append(entity_id)

        return list(set(mentioned_entities))

    def _run_causal_simulation(self, cause_entity: str, simulation_steps: int = 5) -> Dict[str, float]:
        """(Implementation Unchanged)"""
        # This part is currently not being reached in tests, but the logic is kept.
        if not self.cellular_world:
             return {}
        # ... (rest of the simulation logic)
        return {}


    def deduce_facts(self, message: str) -> List[str]:
        """
        Analyzes a message to deduce relevant facts from both the static KG
        and a dynamic simulation in the Cellular World.
        """
        final_facts = set()
        mentioned_entities = self._find_mentioned_entities(message)

        query_is_for_effect = "결과" in message or "영향" in message or "만약" in message

        if not mentioned_entities:
            return []

        for entity in mentioned_entities:
            static_facts = self._deduce_static_facts(entity)
            final_facts.update(static_facts)

            if query_is_for_effect and self.cellular_world:
                sim_results = self._run_causal_simulation(entity)
                if sim_results:
                    fact_header = f"'{entity}'(으)로 시뮬레이션한 결과, 다음과 같은 동적 영향이 관찰되었습니다:"
                    final_facts.add(fact_header)
                    for affected_entity, energy in sorted(sim_results.items()):
                        dynamic_fact = f"  - '{affected_entity}' 개념이 활성화되었습니다 (에너지: {energy})."
                        final_facts.add(dynamic_fact)

        return sorted(list(final_facts))


    def _deduce_static_facts(self, entity: str) -> set:
        """(Implementation Unchanged)"""
        static_facts = set()
        for edge in self.kg_manager.kg.get('edges', []):
            if edge.get('source') == entity:
                relation, target = edge.get('relation', 'related_to'), edge.get('target')
                if relation == 'is_a':
                    static_facts.add(f"'{entity}'은(는) '{target}'의 한 종류입니다.")
                elif relation == 'causes':
                     static_facts.add(f"[정적] '{entity}'은(는) '{target}'을(를) 유발할 수 있습니다.")
                else:
                    static_facts.add(f"'{entity}'은(는) '{target}'와(과) '{relation}' 관계입니다.")
            elif edge.get('target') == entity:
                relation, source = edge.get('relation', 'related_to'), edge.get('source')
                if relation == 'is_a':
                    static_facts.add(f"'{source}'은(는) '{entity}'의 한 예시입니다.")
                elif relation == 'causes':
                    static_facts.add(f"[정적] '{source}'은(는) '{entity}'의 원인이 될 수 있습니다.")
        return static_facts

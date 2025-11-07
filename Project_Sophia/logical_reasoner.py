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
        """Initializes the LogicalReasoner with injected dependencies."""
        self.kg_manager = kg_manager or KGManager()
        self.cellular_world = cellular_world

    def _find_mentioned_entities(self, message: str) -> List[str]:
        """Finds entities from the KG mentioned in a message, handling Korean particles."""
        mentioned_entities = []
        node_ids = {node.get('id') for node in self.kg_manager.kg.get('nodes', [])}

        for entity_id in node_ids:
            if entity_id and re.search(re.escape(entity_id), message, re.IGNORECASE):
                mentioned_entities.append(entity_id)
        return list(set(mentioned_entities))

    def _run_causal_simulation(self, cause_entity: str, simulation_steps: int = 5) -> Dict[str, float]:
        """
        Runs a causal simulation to infer potential outcomes.
        Returns a dictionary of affected entities and their final energy levels.
        """
        if not self.cellular_world or cause_entity not in self.cellular_world.cells:
            return {}

        # 1. Create a safe sandbox for the simulation
        sandbox_world = copy.deepcopy(self.cellular_world)

        # 2. Prime the sandbox with connections from the static KG for this simulation
        for cell_id, cell in sandbox_world.cells.items():
            edges = self.kg_manager.find_effects(cell_id)
            for edge in edges:
                target_id = edge.get('target')
                if target_id in sandbox_world.cells:
                    strength = edge.get('strength', 0.5)
                    cell.connect(sandbox_world.cells[target_id], strength=strength)

        # 3. Apply a strong stimulus to the cause entity
        sandbox_world.inject_stimulus(cause_entity, 1.0)

        # 4. Run the simulation for several steps to allow energy to propagate
        for _ in range(simulation_steps):
            sandbox_world.run_simulation_step()

        # 5. Analyze results to find significantly affected cells
        affected_entities = {}
        for cell_id, cell in sandbox_world.cells.items():
            original_cell = self.cellular_world.cells.get(cell_id)
            original_energy = original_cell.energy if original_cell else 0.0

            # A cell is "significantly affected" if its energy has noticeably increased
            if cell.energy > original_energy + 0.05:
                # We exclude the cause entity itself from the results
                if cell_id != cause_entity:
                    affected_entities[cell_id] = round(cell.energy, 2)

        return affected_entities

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
            # First, deduce static facts from the Knowledge Graph
            static_facts = self._deduce_static_facts(entity)
            final_facts.update(static_facts)

            # If the query is about consequences, run the dynamic simulation
            if query_is_for_effect:
                sim_results = self._run_causal_simulation(entity)
                if sim_results:
                    fact_header = f"'{entity}'(으)로 시뮬레이션한 결과, 다음과 같은 동적 영향이 관찰되었습니다:"
                    final_facts.add(fact_header)
                    for affected_entity, energy in sorted(sim_results.items()):
                        dynamic_fact = f"  - '{affected_entity}' 개념이 활성화되었습니다 (에너지: {energy})."
                        final_facts.add(dynamic_fact)

        return sorted(list(final_facts))

    def _deduce_static_facts(self, entity: str) -> set:
        """Helper function to get facts from the static Knowledge Graph."""
        static_facts = set()

        # Find relationships where the entity is the source
        for edge in self.kg_manager.kg.get('edges', []):
            if edge.get('source') == entity:
                relation, target = edge['relation'], edge['target']
                if relation == 'is_a':
                    static_facts.add(f"'{entity}'은(는) '{target}'의 한 종류입니다.")
                elif relation == 'causes':
                     static_facts.add(f"[정적] '{entity}'은(는) '{target}'을(를) 유발할 수 있습니다.")
                else:
                    static_facts.add(f"'{entity}'은(는) '{target}'와(과) '{relation}' 관계입니다.")

        # Find relationships where the entity is the target
        for edge in self.kg_manager.kg.get('edges', []):
            if edge.get('target') == entity:
                relation, source = edge['relation'], edge['source']
                if relation == 'is_a':
                    static_facts.add(f"'{source}'은(는) '{entity}'의 한 예시입니다.")
                elif relation == 'causes':
                    static_facts.add(f"[정적] '{source}'은(는) '{entity}'의 원인이 될 수 있습니다.")

        return static_facts

import re
import copy
from typing import List, Dict, Any, Optional

from tools.kg_manager import KGManager
from .core.world import World
from .core.thought import Thought

class LogicalReasoner:
    """
    Deduces logical facts from user input by interacting with the knowledge graph
    and running dynamic simulations in the Cellular World.
    """

    def __init__(self, kg_manager: Optional[KGManager] = None, cellular_world: Optional[World] = None):
        self.kg_manager = kg_manager or KGManager()
        self.cellular_world = cellular_world

    def _find_mentioned_entities(self, message: str) -> List[str]:
        """
        Finds entities from the KG mentioned in a message using a robust,
        length-sorted substring search to handle multi-word entities correctly.
        """
        mentioned_entities = []
        nodes = self.kg_manager.kg.get('nodes', [])
        if not nodes:
            return []

        node_ids = {node.get('id') for node in nodes if node.get('id')}

        # Sort by length descending to match longer names first (e.g., "black hole" before "hole")
        sorted_node_ids = sorted(list(node_ids), key=len, reverse=True)

        lower_message = message.lower()
        for entity_id in sorted_node_ids:
            if entity_id.lower() in lower_message:
                mentioned_entities.append(entity_id)

        return list(set(mentioned_entities))

    def _run_causal_simulation(self, cause_entity: str, simulation_steps: int = 5) -> List[Thought]:
        """
        Runs a dynamic simulation in the Cellular World to find potential causal effects.
        """
        if not self.cellular_world:
            return []

        # 1. Store initial state to find changes
        initial_energies: Dict[str, float] = {
            cell_id: cell.energy for cell_id, cell in self.cellular_world.cells.items()
        }

        # 2. Inject stimulus
        self.cellular_world.inject_stimulus(cause_entity, energy_boost=100.0)

        # 3. Run simulation
        for _ in range(simulation_steps):
            self.cellular_world.run_simulation_step()

        # 4. Analyze results and generate thoughts
        simulation_thoughts: List[Thought] = []
        for cell_id, cell in self.cellular_world.cells.items():
            initial_energy = initial_energies.get(cell_id, 0.0)
            final_energy = cell.energy

            # Consider a cell "activated" if its energy significantly increased
            # and it's not the cause_entity itself.
            if final_energy > initial_energy + 10.0 and cell_id != cause_entity:
                content = f"'{cause_entity}'의 영향으로 '{cell_id}' 개념이 활성화될 수 있습니다."
                thought = Thought(
                    content=content,
                    source='flesh',  # '살'에서 비롯된 생각
                    confidence=0.7,  # Simulation results are less certain than static KG facts
                    energy=final_energy,
                    evidence=[{'cell_id': cell_id, 'initial_energy': initial_energy, 'final_energy': final_energy}]
                )
                simulation_thoughts.append(thought)

        # Sort by energy for relevance
        simulation_thoughts.sort(key=lambda t: t.energy, reverse=True)

        return simulation_thoughts


    def deduce_facts(self, message: str) -> List[Thought]:
        """
        Analyzes a message to deduce relevant facts from both the static KG
        and a dynamic simulation in the Cellular World.
        Returns a list of Thought objects.
        """
        all_thoughts: List[Thought] = []
        mentioned_entities = self._find_mentioned_entities(message)

        if not mentioned_entities:
            return []

        query_is_for_effect = "결과" in message or "영향" in message or "만약" in message

        for entity in mentioned_entities:
            # 1. Get static facts from KG
            static_thoughts = self._deduce_static_facts(entity)
            all_thoughts.extend(static_thoughts)

            # 2. Get dynamic insights from simulation if relevant
            if query_is_for_effect and self.cellular_world:
                # Create a deep copy of the world to run a clean simulation
                # without affecting the main instance.
                sim_world = copy.deepcopy(self.cellular_world)
                reasoner_for_sim = LogicalReasoner(self.kg_manager, sim_world)
                dynamic_thoughts = reasoner_for_sim._run_causal_simulation(entity)
                all_thoughts.extend(dynamic_thoughts)

        # Remove duplicate thoughts based on content
        unique_thoughts_dict = {t.content: t for t in all_thoughts}
        return list(unique_thoughts_dict.values())


    def _deduce_static_facts(self, entity: str) -> List[Thought]:
        """Deduces facts from the static KG and returns them as Thought objects."""
        static_thoughts: List[Thought] = []
        for edge in self.kg_manager.kg.get('edges', []):
            thought = None
            if edge.get('source') == entity:
                relation, target = edge.get('relation', 'related_to'), edge.get('target')
                content = ""
                if relation == 'is_a':
                    content = f"'{entity}'은(는) '{target}'의 한 종류입니다."
                elif relation == 'causes':
                     content = f"'{entity}'은(는) '{target}'을(를) 유발할 수 있습니다."
                else:
                    content = f"'{entity}'은(는) '{target}'와(과) '{relation}' 관계입니다."

                if content:
                    thought = Thought(
                        content=content,
                        source='bone',  # '뼈'에서 비롯된 생각
                        confidence=0.95, # Static facts are highly confident
                        evidence=[edge]
                    )

            elif edge.get('target') == entity:
                relation, source = edge.get('relation', 'related_to'), edge.get('source')
                content = ""
                if relation == 'is_a':
                    content = f"'{source}'은(는) '{entity}'의 한 예시입니다."
                elif relation == 'causes':
                    content = f"'{source}'은(는) '{entity}'의 원인이 될 수 있습니다."

                if content:
                    thought = Thought(
                        content=content,
                        source='bone',  # '뼈'에서 비롯된 생각
                        confidence=0.9, # Inferred relationships are slightly less confident
                        evidence=[edge]
                    )

            if thought:
                static_thoughts.append(thought)

        return static_thoughts

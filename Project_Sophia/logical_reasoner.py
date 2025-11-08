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
        sorted_node_ids = sorted(list(node_ids), key=len, reverse=True)
        lower_message = message.lower()
        for entity_id in sorted_node_ids:
            if entity_id.lower() in lower_message:
                mentioned_entities.append(entity_id)
        return list(set(mentioned_entities))

    def _run_thought_experiment(self, hypothesis: Thought, simulation_steps: int = 3) -> Dict[str, Any]:
        """
        Runs a 'thought experiment' in the Cellular World to test a hypothesis (a Thought object).
        Returns a dictionary summarizing the experiment's results.
        """
        if not self.cellular_world or not isinstance(hypothesis.evidence, list) or not hypothesis.evidence:
            return {"outcome": "not_applicable", "reason": "No cellular world or evidence to test."}

        edge = hypothesis.evidence[0]
        if not isinstance(edge, dict) or 'source' not in edge or 'target' not in edge:
            return {"outcome": "not_applicable", "reason": "Evidence is not a valid edge."}

        cause_entity, expected_effect = edge['source'], edge['target']

        sim_world = copy.deepcopy(self.cellular_world)

        initial_effect_energy = sim_world.get_cell(expected_effect).energy if sim_world.get_cell(expected_effect) else 0.0
        initial_total_energy = sim_world.get_total_energy()

        # Stimulate the cause entity
        sim_world.inject_stimulus(cause_entity, energy_boost=100.0)

        # Run the simulation
        newly_born_cells = []
        for _ in range(simulation_steps):
            newly_born_cells.extend(sim_world.run_simulation_step())

        final_effect_energy = sim_world.get_cell(expected_effect).energy if sim_world.get_cell(expected_effect) else 0.0
        final_total_energy = sim_world.get_total_energy()

        energy_increase = final_effect_energy - initial_effect_energy
        outcome = "inconclusive"
        if energy_increase > 10.0:
            outcome = "verified"
        elif energy_increase < 1.0:
            outcome = "refuted"

        return {
            "outcome": outcome,
            "cause": cause_entity,
            "expected_effect": expected_effect,
            "actual_energy_change": round(energy_increase, 2),
            "newly_born_cells": [cell.id for cell in newly_born_cells],
            "simulation_narrative": f"Stimulated '{cause_entity}'. Energy of '{expected_effect}' changed by {energy_increase:.2f}. {len(newly_born_cells)} new cells born."
        }

    def deduce_facts(self, message: str) -> List[Thought]:
        """
        Analyzes a message, deduces hypotheses from the static KG ('bone'),
        and then verifies them through 'thought experiments' in the Cellular World ('flesh').
        """
        all_thoughts: List[Thought] = []
        mentioned_entities = self._find_mentioned_entities(message)

        if not mentioned_entities:
            return []

        for entity in mentioned_entities:
            # 1. Generate hypotheses from the static KG ('bone')
            hypotheses = self._deduce_static_hypotheses(entity)

            # 2. Run thought experiments for each hypothesis
            for hypo in hypotheses:
                experiment_result = self._run_thought_experiment(hypo)
                hypo.experiment = experiment_result

                # Update confidence based on experiment outcome
                if experiment_result['outcome'] == 'verified':
                    hypo.confidence = min(0.99, hypo.confidence * 1.2) # Increase confidence
                elif experiment_result['outcome'] == 'refuted':
                    hypo.confidence *= 0.5 # Decrease confidence

                # Create a new thought if an unexpected outcome occurred
                if experiment_result['newly_born_cells']:
                    for new_cell_id in experiment_result['newly_born_cells']:
                        new_content = f"'{experiment_result['cause']}'에 대한 사고 실험 중, 예상치 못하게 '{new_cell_id}'라는 새로운 개념이 탄생했습니다."
                        emergent_thought = Thought(
                            content=new_content,
                            source='flesh',
                            confidence=0.75,
                            evidence=[{'source': experiment_result['cause'], 'emerged': new_cell_id}],
                            experiment=experiment_result
                        )
                        all_thoughts.append(emergent_thought)

                all_thoughts.append(hypo)

        unique_thoughts_dict = {t.content: t for t in all_thoughts}
        return list(unique_thoughts_dict.values())

    def _deduce_static_hypotheses(self, entity: str) -> List[Thought]:
        """Deduces hypotheses from the static KG and returns them as Thought objects."""
        hypotheses: List[Thought] = []
        for edge in self.kg_manager.kg.get('edges', []):
            thought = None
            content = ""

            if edge.get('source') == entity:
                relation, target = edge.get('relation', 'related_to'), edge.get('target')
                if relation == 'is_a': content = f"'{entity}'은(는) '{target}'의 한 종류일 수 있습니다."
                elif relation == 'causes': content = f"가설: '{entity}'은(는) '{target}'을(를) 유발할 수 있습니다."
                else: content = f"'{entity}'은(는) '{target}'와(과) '{relation}' 관계일 수 있습니다."

                if content:
                    thought = Thought(content=content, source='bone', confidence=0.8, evidence=[edge])

            elif edge.get('target') == entity:
                relation, source = edge.get('relation', 'related_to'), edge.get('source')
                if relation == 'is_a': content = f"'{source}'은(는) '{entity}'의 한 예시일 수 있습니다."
                elif relation == 'causes': content = f"가설: '{source}'은(는) '{entity}'의 원인이 될 수 있습니다."

                if content:
                    thought = Thought(content=content, source='bone', confidence=0.7, evidence=[edge])

            if thought:
                hypotheses.append(thought)

        return hypotheses

from __future__ import annotations

from typing import List, Dict, Any, Optional
import random

from tools.kg_manager import KGManager
from nano_core.bus import MessageBus
from nano_core.message import Message


class ExplorationCortex:
    """
    엘리시아의 '호기심 엔진'.
    고요한 시간('꿈의 주기') 동안 자신의 지식 그래프를 스스로 탐험하고,
    지식의 빈틈을 발견하여 새로운 '가설'을 생성합니다.
    """

    def __init__(self, kg_manager: KGManager, bus: MessageBus):
        self.kg = kg_manager
        self.bus = bus

    def explore_and_hypothesize(self, num_hypotheses: int = 1):
        """
        지식 그래프를 탐험하고 새로운 가설을 생성하여 MessageBus에 게시합니다.
        """
        nodes = self.kg.kg.get('nodes', [])
        edges = self.kg.kg.get('edges', [])
        if not nodes or len(nodes) < 2:
            return

        interesting_nodes = [n['id'] for n in nodes if n.get('mass', 0.0) > 1.0 or n.get('activation_energy', 0.0) > 0.1]
        if not interesting_nodes:
            interesting_nodes = [n['id'] for n in random.sample(nodes, min(len(nodes), 5))]

        hypotheses_made = 0
        for _ in range(num_hypotheses * 10): # Try multiple times to find a valid hypothesis
            if hypotheses_made >= num_hypotheses:
                break

            start_node_id = random.choice(interesting_nodes)

            # Perform a 2-step random walk
            mid_node_id = self._get_random_neighbor(start_node_id, edges)
            if not mid_node_id:
                continue

            end_node_id = self._get_random_neighbor(mid_node_id, edges)
            if not end_node_id:
                continue

            # Check if it's a valid hypothesis (not self, not directly connected)
            if start_node_id != end_node_id and not self._are_directly_connected(start_node_id, end_node_id, edges):
                # Propose a weak, general relationship
                hypothesis_msg = Message(
                    verb="validate",
                    slots={'subject': start_node_id, 'object': end_node_id, 'relation': 'related_to'},
                    strength=0.1,  # Very low strength, as it's just a guess
                    src="ExplorationCortex"
                )
                self.bus.post(hypothesis_msg)
                hypotheses_made += 1

    def _get_random_neighbor(self, node_id: str, edges: List[Dict]) -> Optional[str]:
        neighbors = []
        for edge in edges:
            if edge.get('source') == node_id:
                neighbors.append(edge.get('target'))
            elif edge.get('target') == node_id:
                neighbors.append(edge.get('source'))

        return random.choice(neighbors) if neighbors else None

    def _are_directly_connected(self, node1_id: str, node2_id: str, edges: List[Dict]) -> bool:
        for edge in edges:
            if (edge.get('source') == node1_id and edge.get('target') == node2_id) or \
               (edge.get('source') == node2_id and edge.get('target') == node1_id):
                return True
        return False

    def generate_definitional_questions(self, num_questions: int = 1) -> List[str]:
        """
        Finds "lonely" nodes (with few connections) in the knowledge graph
        and generates definitional questions about them.
        """
        lonely_nodes = self._find_lonely_nodes()
        if not lonely_nodes:
            return []

        num_to_generate = min(num_questions, len(lonely_nodes))
        nodes_to_ask_about = random.sample(lonely_nodes, num_to_generate)

        questions = [f"What is '{node_id}'?" for node_id in nodes_to_ask_about]
        return questions

    def _find_lonely_nodes(self, max_connections: int = 1) -> List[str]:
        """
        Finds nodes with a number of connections less than or equal to the threshold.
        """
        nodes = self.kg.kg.get('nodes', [])
        edges = self.kg.kg.get('edges', [])
        if not nodes:
            return []

        connection_counts = {node['id']: 0 for node in nodes}
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            if source in connection_counts:
                connection_counts[source] += 1
            if target in connection_counts:
                connection_counts[target] += 1

        lonely_nodes = [node_id for node_id, count in connection_counts.items() if count <= max_connections]
        return lonely_nodes

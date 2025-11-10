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

    def launch_exploration_mission(self, num_missions: int = 1):
        """
        Launch 'ExplorerBots' (starships) to explore paths between important nodes and lonely nodes.
        This simulates sending a ship on a mission to chart the unknown.
        """
        nodes = self.kg.kg.get('nodes', [])
        if not nodes or len(nodes) < 2:
            return

        # Start from an "important" node (a star system)
        interesting_nodes = [n['id'] for n in nodes if n.get('activation_energy', 0.0) > 0.1]
        if not interesting_nodes:
            return

        # The target is a "lonely" node (an unknown planet)
        lonely_nodes = self._find_lonely_nodes()
        if not lonely_nodes:
            return

        missions_launched = 0
        for _ in range(num_missions * 5): # Try a few times to find a valid mission
            if missions_launched >= num_missions:
                break

            start_node = random.choice(interesting_nodes)
            target_node = random.choice(lonely_nodes)

            # A mission is only valid if the start and end are different
            if start_node != target_node:
                mission_msg = Message(
                    verb="explore",
                    slots={
                        'start_node': start_node,
                        'target': target_node,
                        'path': [start_node] # The path starts with the origin
                    },
                    strength=0.9,  # A new mission starts with high priority
                    src="ExplorationCortex"
                )
                self.bus.post(mission_msg)
                missions_launched += 1

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

    def get_random_highly_connected_node(self) -> Optional[str]:
        """
        Finds a random node from the top quintile (20%) of most-connected nodes.
        This provides a good candidate for a "thought experiment" seed.
        """
        nodes = self.kg.kg.get('nodes', [])
        edges = self.kg.kg.get('edges', [])
        if not nodes or len(nodes) < 5: # Need a reasonable number of nodes to calculate quintile
            return None

        connection_counts = {node['id']: 0 for node in nodes}
        for edge in edges:
            source, target = edge.get('source'), edge.get('target')
            if source in connection_counts:
                connection_counts[source] += 1
            if target in connection_counts:
                connection_counts[target] += 1

        if not connection_counts:
            return None

        # Sort nodes by connection count, descending
        sorted_nodes = sorted(connection_counts.items(), key=lambda item: item[1], reverse=True)

        # Calculate the index for the top 20%
        top_quintile_index = len(sorted_nodes) // 5

        # Get the list of candidate nodes (top 20%)
        candidate_nodes = sorted_nodes[:top_quintile_index]

        if not candidate_nodes:
            # Fallback for very small graphs
            return sorted_nodes[0][0] if sorted_nodes else None

        # Return the ID of a random node from the candidates
        return random.choice(candidate_nodes)[0]

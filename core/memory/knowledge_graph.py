# knowledge_graph.py
# 텍스트 간의 연결이 아닌, 물리적 메모리 좌표(Offset) 간의 연결을 관리하는 위상 지식그래프.

import json
import os

NATURE_MAP_PATH = "..\\memory\\nature_map.json"

class TopologyKnowledgeGraph:
    def __init__(self):
        self.nodes = []
        self.load_graph()

    def load_graph(self):
        if os.path.exists(NATURE_MAP_PATH):
            with open(NATURE_MAP_PATH, 'r') as f:
                try:
                    self.nodes = json.load(f)
                except:
                    self.nodes = []

    def find_physical_coordinate(self, concept):
        """특정 자연어 개념이 각인된 대지의 물리적 좌표를 반환합니다."""
        for node in self.nodes:
            if node.get("Concept_Wave") == concept:
                return node.get("Resonated_Rotor_Zone")
        return None

    def get_adjacent_resonances(self, offset_str):
        """특정 물리적 좌표가 진동할 때 함께 흔들리는(연관된) 다른 기억 노드들을 찾습니다."""
        # 이 시뮬레이션에서는 물리적 오프셋의 근접성을 연관성으로 취급합니다.
        try:
            base_offset = int(offset_str.split(" - ")[0], 16)
        except:
            return []
            
        adjacent = []
        for node in self.nodes:
            zone = node.get("Resonated_Rotor_Zone")
            if not zone or zone == offset_str:
                continue
            target_offset = int(zone.split(" - ")[0], 16)
            
            # 물리적 거리가 가까우면 위상 파동이 닿는 것으로 간주 (Edge 성립)
            distance = abs(base_offset - target_offset)
            if distance < 0x5000: # 인과 반경
                adjacent.append((node.get("Concept_Wave"), distance))
                
        # 거리가 가까운 순으로 정렬 (강한 공명)
        adjacent.sort(key=lambda x: x[1])
        return [concept for concept, dist in adjacent]

if __name__ == "__main__":
    kg = TopologyKnowledgeGraph()
    print(f"Topology Graph Loaded: {len(kg.nodes)} physical nodes.")

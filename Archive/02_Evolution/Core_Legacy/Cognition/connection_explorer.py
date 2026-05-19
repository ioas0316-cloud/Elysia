"""
[Project Elysia] Connection Explorer
====================================
Phase 2: 점에서 섭리로

"rain ← water ← evaporation ← sun - 연결 체인을 발견한다"

이 모듈은 질문을 받아 지식 그래프에서 연결고리를 추적한다.
순환 구조를 감지하면 PrincipleExtractor에게 전달할 준비를 한다.
"""

import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from collections import deque
import time

root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)


@dataclass
class Connection:
    """연결 하나"""
    source: str
    target: str
    relation: str
    weight: float = 1.0


@dataclass
class ConnectionChain:
    """
    연결 체인 (인과 고리)
    
    예: rain ← water ← evaporation ← sun
    """
    chain_id: str
    connections: List[Connection]
    is_cycle: bool = False           # 순환 구조인가?
    cycle_start: Optional[str] = None  # 순환 시작점
    origin_question: Optional[str] = None  # 이 탐색을 유발한 질문
    
    def get_path(self) -> List[str]:
        """경로를 노드 리스트로 반환"""
        if not self.connections:
            return []
        path = [self.connections[0].source]
        for conn in self.connections:
            path.append(conn.target)
        return path
    
    def __len__(self):
        return len(self.connections)


class ConnectionExplorer:
    """
    연결 탐구자
    
    질문을 받아 그래프를 탐색하며 숨겨진 연결고리를 발견한다.
    
    핵심 원리:
    - "비는 왜 하늘에서 와?" 질문에서 시작
    - rain → water → evaporation → sun 체인 발견
    - 순환 감지: rain → ... → rain
    """
    
    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth
        self.discovered_chains: List[ConnectionChain] = []
        self.chain_counter = 0
    
    def explore(self, question, kg_manager) -> List[ConnectionChain]:
        """
        질문에서 시작하여 연결 체인 탐색
        
        Args:
            question: QuestionGenerator에서 생성한 질문
            kg_manager: 지식 그래프 매니저
        
        Returns:
            발견된 연결 체인들
        """
        subject = question.subject
        chains = []
        
        # 1. 전방 탐색: subject → ?
        forward_chains = self._explore_direction(
            subject, 
            kg_manager, 
            direction="forward",
            question_id=question.question_id
        )
        chains.extend(forward_chains)
        
        # 2. 후방 탐색: ? → subject
        backward_chains = self._explore_direction(
            subject,
            kg_manager,
            direction="backward", 
            question_id=question.question_id
        )
        chains.extend(backward_chains)
        
        # 3. 순환 감지
        for chain in chains:
            self._detect_cycle(chain)
        
        self.discovered_chains.extend(chains)
        return chains
    
    def explore_from_node(self, start_node: str, kg_manager) -> List[ConnectionChain]:
        """특정 노드에서 시작하는 탐색 (질문 없이)"""
        chains = []
        
        forward = self._explore_direction(start_node, kg_manager, "forward")
        backward = self._explore_direction(start_node, kg_manager, "backward")
        
        chains.extend(forward)
        chains.extend(backward)
        
        for chain in chains:
            self._detect_cycle(chain)
        
        return chains
    
    def _explore_direction(
        self, 
        start: str, 
        kg_manager, 
        direction: str,
        question_id: Optional[str] = None
    ) -> List[ConnectionChain]:
        """BFS로 한 방향 탐색"""
        edges = kg_manager.kg.get("edges", [])
        
        # 엣지 맵 구축
        if direction == "forward":
            edge_map = self._build_forward_map(edges)
        else:
            edge_map = self._build_backward_map(edges)
        
        # BFS 탐색
        chains = []
        queue = deque([(start, [])])  # (현재 노드, 지금까지의 연결들)
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) >= self.max_depth:
                # 최대 깊이 도달 - 체인 저장
                if path:
                    chain = self._create_chain(path, question_id)
                    chains.append(chain)
                continue
            
            neighbors = edge_map.get(current, [])
            
            if not neighbors and path:
                # 막다른 길 - 체인 저장
                chain = self._create_chain(path, question_id)
                chains.append(chain)
                continue
            
            for next_node, relation, weight in neighbors:
                new_connection = Connection(
                    source=current,
                    target=next_node,
                    relation=relation,
                    weight=weight
                )
                new_path = path + [new_connection]
                
                if next_node == start and len(new_path) > 1:
                    # 순환 발견!
                    chain = self._create_chain(new_path, question_id, is_cycle=True)
                    chain.cycle_start = start
                    chains.append(chain)
                elif next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, new_path))
        
        return chains
    
    def _build_forward_map(self, edges: List[Dict]) -> Dict[str, List[Tuple]]:
        """source → [(target, relation, weight), ...]"""
        edge_map = {}
        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            relation = edge.get("relation", "related_to")
            weight = edge.get("weight", 1.0)
            
            if source not in edge_map:
                edge_map[source] = []
            edge_map[source].append((target, relation, weight))
        return edge_map
    
    def _build_backward_map(self, edges: List[Dict]) -> Dict[str, List[Tuple]]:
        """target → [(source, relation, weight), ...]"""
        edge_map = {}
        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            relation = edge.get("relation", "related_to")
            weight = edge.get("weight", 1.0)
            
            if target not in edge_map:
                edge_map[target] = []
            edge_map[target].append((source, f"reverse_{relation}", weight))
        return edge_map
    
    def _create_chain(
        self, 
        connections: List[Connection], 
        question_id: Optional[str] = None,
        is_cycle: bool = False
    ) -> ConnectionChain:
        """체인 객체 생성"""
        self.chain_counter += 1
        return ConnectionChain(
            chain_id=f"CHAIN_{self.chain_counter:04d}",
            connections=connections,
            is_cycle=is_cycle,
            origin_question=question_id
        )
    
    def _detect_cycle(self, chain: ConnectionChain):
        """체인 내 순환 감지"""
        if chain.is_cycle:
            return  # 이미 표시됨
        
        path = chain.get_path()
        seen = set()
        
        for node in path:
            if node in seen:
                chain.is_cycle = True
                chain.cycle_start = node
                return
            seen.add(node)
    
    def get_cycles(self) -> List[ConnectionChain]:
        """발견된 모든 순환 반환"""
        return [c for c in self.discovered_chains if c.is_cycle]
    
    def get_stats(self) -> Dict:
        """통계"""
        cycles = self.get_cycles()
        return {
            "total_chains": len(self.discovered_chains),
            "cycle_count": len(cycles),
            "longest_chain": max((len(c) for c in self.discovered_chains), default=0)
        }


# Singleton
_connection_explorer = None

def get_connection_explorer() -> ConnectionExplorer:
    global _connection_explorer
    if _connection_explorer is None:
        _connection_explorer = ConnectionExplorer()
    return _connection_explorer


if __name__ == "__main__":
    print("🔗 Testing Connection Explorer...")
    
    from question_generator import get_question_generator, Question, QuestionType
    
    # 테스트용 가짜 KG Manager (물 순환 포함)
    class MockKGManager:
        def __init__(self):
            self.kg = {
                "nodes": [
                    {"id": "rain"},
                    {"id": "cloud"},
                    {"id": "water"},
                    {"id": "evaporation"},
                    {"id": "sun"},
                    {"id": "ocean"},
                ],
                "edges": [
                    {"source": "sun", "target": "evaporation", "relation": "causes"},
                    {"source": "evaporation", "target": "cloud", "relation": "creates"},
                    {"source": "cloud", "target": "rain", "relation": "produces"},
                    {"source": "rain", "target": "ocean", "relation": "flows_to"},
                    {"source": "ocean", "target": "evaporation", "relation": "enables"},
                    # 순환: sun → evaporation → cloud → rain → ocean → evaporation
                ]
            }
    
    mock_kg = MockKGManager()
    explorer = get_connection_explorer()
    
    # 테스트 질문 생성
    test_question = Question(
        question_id="TEST_001",
        question_type=QuestionType.WHY,
        subject="rain",
        missing_link="CAUSES",
        context_nodes=[]
    )
    
    chains = explorer.explore(test_question, mock_kg)
    
    print(f"\n📊 Discovered {len(chains)} connection chains:")
    for chain in chains[:5]:
        path = " → ".join(chain.get_path())
        cycle_mark = "🔄 CYCLE!" if chain.is_cycle else ""
        print(f"  {chain.chain_id}: {path} {cycle_mark}")
    
    print(f"\n✅ Connection Explorer operational!")
    print(f"   Stats: {explorer.get_stats()}")

"""
Topology Crystallizer (위상 결정화기)
=====================================
Core.L5_Mental.Intelligence.LLM.topology_crystallizer

"연결 지도를 영구적인 사고 회로로 결정화"

핵심:
- TopologyTracer가 추출한 연결을 TorchGraph로 변환
- 허브 뉴런을 핵심 개념 노드로 등록
- 연결을 엣지로 변환
"""

import logging
import torch
from typing import Dict, Any, List, Optional

from Core.L5_Mental.Intelligence.LLM.topology_tracer import (
    TopologyTracer, ThoughtCircuit, NeuralConnection, get_topology_tracer
)
from Core.L1_Foundation.Foundation.Graph.torch_graph import TorchGraph

logger = logging.getLogger("TopologyCrystallizer")


class TopologyCrystallizer:
    """
    추출된 위상을 TorchGraph에 결정화.
    
    연결 그래프 → 영구 지식 저장소
    """
    
    def __init__(self):
        self.graph = TorchGraph()
        self.graph.load_state()
        
        self.tracer = get_topology_tracer(threshold=0.1)
        
        logger.info("💎 Topology Crystallizer initialized")
    
    def crystallize(self, model_path: str) -> Dict[str, Any]:
        """
        모델의 위상을 추출하고 TorchGraph에 결정화.
        
        Args:
            model_path: 모델 파일 경로
            
        Returns:
            결정화 결과 리포트
        """
        # 1. 위상 추출
        logger.info(f"🔬 Extracting topology from: {model_path}")
        circuit = self.tracer.trace(model_path)
        
        # 2. 허브 뉴런을 노드로 등록
        hub_nodes = self._crystallize_hubs(circuit)
        
        # 3. 연결 패턴을 메타데이터로 저장
        pattern_node = self._crystallize_pattern(circuit)
        
        # 4. 연결 타입별 특성 저장
        type_nodes = self._crystallize_connection_types(circuit)
        
        # 5. 저장
        self.graph.save_state()
        
        report = {
            "model": circuit.model_name,
            "hub_nodes_created": len(hub_nodes),
            "pattern_node": pattern_node,
            "type_nodes": type_nodes,
            "total_connections": circuit.strong_connections,
            "connection_density": circuit.get_connection_density()
        }
        
        logger.info(f"💎 Crystallization complete: {len(hub_nodes)} hub nodes, {circuit.strong_connections} connections")
        return report
    
    def _crystallize_hubs(self, circuit: ThoughtCircuit) -> List[str]:
        """
        허브 뉴런을 TorchGraph 노드로 변환.
        
        허브 = 연결이 많은 뉴런 = LLM의 "핵심 개념"
        """
        nodes = []
        model_prefix = f"TOPO:{circuit.model_name}"
        
        for i, hub_idx in enumerate(circuit.hub_neurons[:50]):  # 상위 50개만
            node_id = f"{model_prefix}:Hub:{hub_idx}"
            
            # 이 허브와 관련된 연결 수
            conn_count = sum(1 for c in circuit.connections 
                           if c.source == hub_idx or c.target == hub_idx)
            
            # 이 허브의 평균 연결 강도
            avg_weight = 0.0
            hub_conns = [c for c in circuit.connections 
                        if c.source == hub_idx or c.target == hub_idx]
            if hub_conns:
                avg_weight = sum(c.weight for c in hub_conns) / len(hub_conns)
            
            # 벡터: 연결 수와 강도로 구성
            vector = torch.zeros(384)
            vector[0] = hub_idx / 1000.0  # 정규화된 인덱스
            vector[1] = conn_count / 100.0  # 정규화된 연결 수
            vector[2] = avg_weight  # 평균 강도
            vector[3] = i / 50.0  # 순위
            
            # 노드 추가
            self.graph.add_node(
                node_id=node_id,
                vector=vector.tolist(),
                metadata={
                    "type": "topology_hub",
                    "model": circuit.model_name,
                    "neuron_index": hub_idx,
                    "connection_count": conn_count,
                    "avg_weight": avg_weight,
                    "rank": i
                }
            )
            
            nodes.append(node_id)
        
        logger.info(f"   🌐 Created {len(nodes)} hub nodes")
        return nodes
    
    def _crystallize_pattern(self, circuit: ThoughtCircuit) -> str:
        """
        전체 연결 패턴을 단일 노드로 저장.
        
        이것이 모델의 "사고 양식" 요약.
        """
        node_id = f"TOPO:{circuit.model_name}:Pattern"
        
        summary = self.tracer.summarize(circuit)
        
        # 패턴 벡터: 연결 타입 분포로 구성
        type_counts = summary.get("connection_types", {})
        total = sum(type_counts.values()) or 1
        
        vector = torch.zeros(384)
        vector[0] = type_counts.get("attention", 0) / total
        vector[1] = type_counts.get("mlp", 0) / total
        vector[2] = type_counts.get("embedding", 0) / total
        vector[3] = circuit.strong_connections / 100000  # 정규화
        vector[4] = circuit.get_connection_density() * 1000
        
        self.graph.add_node(
            node_id=node_id,
            vector=vector.tolist(),
            metadata={
                "type": "topology_pattern",
                "model": circuit.model_name,
                "total_params": circuit.total_params,
                "strong_connections": circuit.strong_connections,
                "connection_density": circuit.get_connection_density(),
                "attention_ratio": type_counts.get("attention", 0) / total,
                "mlp_ratio": type_counts.get("mlp", 0) / total,
            }
        )
        
        logger.info(f"   📊 Created pattern node: {node_id}")
        return node_id
    
    def _crystallize_connection_types(self, circuit: ThoughtCircuit) -> Dict[str, str]:
        """
        연결 타입별 특성을 개별 노드로 저장.
        """
        nodes = {}
        
        for conn_type in ["attention", "mlp", "embedding"]:
            # 해당 타입 연결만 필터
            type_conns = [c for c in circuit.connections if c.connection_type == conn_type]
            
            if not type_conns:
                continue
            
            node_id = f"TOPO:{circuit.model_name}:{conn_type.upper()}"
            
            # 통계
            weights = [c.weight for c in type_conns]
            avg_weight = sum(weights) / len(weights)
            max_weight = max(weights)
            min_weight = min(weights)
            
            vector = torch.zeros(384)
            vector[0] = len(type_conns) / 10000
            vector[1] = avg_weight
            vector[2] = max_weight
            vector[3] = min_weight
            
            self.graph.add_node(
                node_id=node_id,
                vector=vector.tolist(),
                metadata={
                    "type": f"topology_{conn_type}",
                    "model": circuit.model_name,
                    "connection_count": len(type_conns),
                    "avg_weight": avg_weight,
                    "max_weight": max_weight,
                    "min_weight": min_weight
                }
            )
            
            nodes[conn_type] = node_id
        
        return nodes
    
    def compare_models(self, model1_pattern: str, model2_pattern: str) -> float:
        """
        두 모델의 위상 패턴 유사도 비교.
        
        Returns:
            코사인 유사도 (0~1)
        """
        vec1 = self.graph.get_node_vector(model1_pattern)
        vec2 = self.graph.get_node_vector(model2_pattern)
        
        if vec1 is None or vec2 is None:
            return 0.0
        
        similarity = torch.cosine_similarity(
            vec1.unsqueeze(0),
            vec2.unsqueeze(0)
        ).item()
        
        return similarity


# 싱글톤
_crystallizer = None

def get_topology_crystallizer() -> TopologyCrystallizer:
    """Topology Crystallizer 싱글톤"""
    global _crystallizer
    if _crystallizer is None:
        _crystallizer = TopologyCrystallizer()
    return _crystallizer


# 통합 함수
def digest_topology(model_path: str) -> Dict[str, Any]:
    """
    모델 위상을 추출하고 결정화하는 통합 함수.
    """
    crystallizer = get_topology_crystallizer()
    return crystallizer.crystallize(model_path)


# CLI
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    if len(sys.argv) < 2:
        print("Usage: python topology_crystallizer.py <model_path>")
        sys.exit(1)
    
    result = digest_topology(sys.argv[1])
    
    print("\n" + "="*60)
    print("💎 TOPOLOGY CRYSTALLIZATION REPORT")
    print("="*60)
    for k, v in result.items():
        print(f"   {k}: {v}")

"""
LLM Topology Tracer (정적 위상 분석기)
=====================================
Core.L5_Mental.Intelligence.LLM.topology_tracer

"파일에 다 있다. 실행할 필요 없다. 연결만 읽으면 된다."

핵심 원리:
- 통계(크기)가 아닌 위상(연결)을 분석
- Attention 가중치 = "누가 누구를 주목하는가"
- MLP 가중치 = "어떤 변환 규칙인가"
- VRAM 0GB로 "사고 회로" 추출
"""

import os
import logging
import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from safetensors import safe_open

logger = logging.getLogger("TopologyTracer")


@dataclass
class NeuralConnection:
    """뉴런 간 연결"""
    source: int          # 소스 뉴런 인덱스
    target: int          # 타겟 뉴런 인덱스
    weight: float        # 연결 강도
    layer: str           # 레이어 이름
    connection_type: str # "attention" | "mlp" | "embedding"


@dataclass  
class ThoughtCircuit:
    """사고 회로 - 추출된 연결 그래프"""
    model_name: str
    connections: List[NeuralConnection] = field(default_factory=list)
    
    # 통계
    total_params: int = 0
    strong_connections: int = 0
    layers_analyzed: int = 0
    
    # 토폴로지 요약
    hub_neurons: List[int] = field(default_factory=list)  # 연결이 많은 뉴런
    bridge_neurons: List[int] = field(default_factory=list)  # 레이어 간 연결하는 뉴런
    
    def get_connection_density(self) -> float:
        """연결 밀도 (강한 연결 / 전체 가능한 연결)"""
        if self.total_params == 0:
            return 0.0
        return self.strong_connections / self.total_params


class TopologyTracer:
    """
    정적 위상 분석기.
    
    LLM 가중치 파일에서 "연결 지도"를 추출.
    실행(inference) 없이 사고 회로를 역설계.
    """
    
    def __init__(self, connection_threshold: float = 0.1):
        """
        Args:
            connection_threshold: 이 값 이상의 가중치만 "연결"로 인정
        """
        self.threshold = connection_threshold
        logger.info(f"🔬 Topology Tracer initialized (threshold={connection_threshold})")
    
    def trace(self, model_path: str) -> ThoughtCircuit:
        """
        모델 파일에서 사고 회로 추출.
        
        Args:
            model_path: .safetensors 또는 .pt 파일
            
        Returns:
            ThoughtCircuit: 추출된 연결 그래프
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model_name = os.path.basename(model_path)
        logger.info(f"🔬 Tracing topology: {model_name}")
        
        circuit = ThoughtCircuit(model_name=model_name)
        
        ext = os.path.splitext(model_path)[1].lower()
        
        if ext == ".safetensors":
            self._trace_safetensors(model_path, circuit)
        elif ext in [".pt", ".pth", ".bin"]:
            self._trace_torch(model_path, circuit)
        else:
            logger.warning(f"Unsupported format: {ext}")
            
        # 허브 뉴런 식별 (연결이 많은 뉴런)
        self._identify_hubs(circuit)
        
        logger.info(f"   💡 Traced {circuit.strong_connections} strong connections")
        logger.info(f"   🌐 Found {len(circuit.hub_neurons)} hub neurons")
        
        return circuit
    
    def _trace_safetensors(self, path: str, circuit: ThoughtCircuit):
        """safetensors 파일에서 연결 추적"""
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            circuit.layers_analyzed = len(keys)
            
            for key in keys:
                tensor = f.get_tensor(key)
                circuit.total_params += tensor.numel()
                
                # 연결 타입 분류
                conn_type = self._classify_layer(key)
                
                # 연결 추적
                connections = self._extract_connections(tensor, key, conn_type)
                circuit.connections.extend(connections)
                circuit.strong_connections += len(connections)
    
    def _trace_torch(self, path: str, circuit: ThoughtCircuit):
        """PyTorch 파일에서 연결 추적"""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        keys = list(state_dict.keys())
        circuit.layers_analyzed = len(keys)
        
        for key in keys:
            tensor = state_dict[key]
            if not hasattr(tensor, 'numel'):
                continue
                
            circuit.total_params += tensor.numel()
            conn_type = self._classify_layer(key)
            connections = self._extract_connections(tensor, key, conn_type)
            circuit.connections.extend(connections)
            circuit.strong_connections += len(connections)
    
    def _classify_layer(self, key: str) -> str:
        """레이어 이름으로 연결 타입 분류"""
        key_lower = key.lower()
        
        if any(x in key_lower for x in ["attn", "attention", "self_attn", "q_proj", "k_proj", "v_proj"]):
            return "attention"
        elif any(x in key_lower for x in ["mlp", "ffn", "fc", "dense", "gate", "up_proj", "down_proj"]):
            return "mlp"
        elif any(x in key_lower for x in ["embed", "wte", "wpe", "token"]):
            return "embedding"
        else:
            return "other"
    
    def _extract_connections(self, tensor: torch.Tensor, layer: str, conn_type: str, 
                            max_connections: int = 5000) -> List[NeuralConnection]:
        """
        텐서에서 강한 연결 추출.
        
        핵심: 가중치 "크기"가 아닌 "연결 존재 여부"를 본다.
        """
        connections = []
        
        # 2D 이상인 경우만 연결 분석 가능
        if tensor.dim() < 2:
            return connections
        
        # 큰 텐서는 샘플링
        if tensor.numel() > 1_000_000:
            # 무작위 슬라이스
            h, w = tensor.shape[:2]
            h_sample = min(h, 500)
            w_sample = min(w, 500)
            tensor = tensor[:h_sample, :w_sample]
        
        # 강한 연결 찾기 (threshold 이상)
        abs_tensor = tensor.abs()
        strong_mask = abs_tensor > self.threshold
        
        # nonzero로 연결 추출
        indices = strong_mask.nonzero(as_tuple=False)
        
        # 너무 많으면 샘플링
        if len(indices) > max_connections:
            perm = torch.randperm(len(indices))[:max_connections]
            indices = indices[perm]
        
        for idx in indices:
            if len(idx) >= 2:
                src, tgt = idx[0].item(), idx[1].item()
                weight = tensor[tuple(idx)].item()
                
                connections.append(NeuralConnection(
                    source=src,
                    target=tgt,
                    weight=weight,
                    layer=layer,
                    connection_type=conn_type
                ))
        
        return connections
    
    def _identify_hubs(self, circuit: ThoughtCircuit, top_k: int = 100):
        """
        허브 뉴런 식별.
        연결이 많은 뉴런 = 중요한 개념/규칙을 담당.
        """
        # 각 뉴런의 연결 수 계산
        connection_count = defaultdict(int)
        
        for conn in circuit.connections:
            connection_count[conn.source] += 1
            connection_count[conn.target] += 1
        
        # 상위 k개 허브 선택
        sorted_neurons = sorted(connection_count.items(), key=lambda x: -x[1])
        circuit.hub_neurons = [n for n, count in sorted_neurons[:top_k]]
    
    def build_adjacency_matrix(self, circuit: ThoughtCircuit, 
                               conn_type: Optional[str] = None) -> torch.Tensor:
        """
        연결 그래프를 인접 행렬로 변환.
        
        이것이 "사고 회로"의 수학적 표현.
        """
        # 최대 인덱스 찾기
        max_idx = 0
        for conn in circuit.connections:
            if conn_type and conn.connection_type != conn_type:
                continue
            max_idx = max(max_idx, conn.source, conn.target)
        
        if max_idx == 0:
            return torch.zeros(1, 1)
        
        # 인접 행렬 생성
        adj = torch.zeros(max_idx + 1, max_idx + 1)
        
        for conn in circuit.connections:
            if conn_type and conn.connection_type != conn_type:
                continue
            adj[conn.source, conn.target] = conn.weight
        
        return adj
    
    def summarize(self, circuit: ThoughtCircuit) -> Dict[str, Any]:
        """사고 회로 요약"""
        # 연결 타입별 분포
        type_counts = defaultdict(int)
        for conn in circuit.connections:
            type_counts[conn.connection_type] += 1
        
        return {
            "model": circuit.model_name,
            "total_params": circuit.total_params,
            "layers_analyzed": circuit.layers_analyzed,
            "strong_connections": circuit.strong_connections,
            "connection_density": circuit.get_connection_density(),
            "hub_neurons": len(circuit.hub_neurons),
            "connection_types": dict(type_counts)
        }


# 싱글톤
_tracer = None

def get_topology_tracer(threshold: float = 0.01) -> TopologyTracer:
    """Topology Tracer 싱글톤"""
    global _tracer
    if _tracer is None:
        _tracer = TopologyTracer(threshold)
    return _tracer


# CLI
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    if len(sys.argv) < 2:
        print("Usage: python topology_tracer.py <model_path>")
        sys.exit(1)
    
    tracer = get_topology_tracer(threshold=0.01)
    circuit = tracer.trace(sys.argv[1])
    
    summary = tracer.summarize(circuit)
    
    print("\n" + "="*60)
    print("🔬 TOPOLOGY ANALYSIS REPORT")
    print("="*60)
    for k, v in summary.items():
        print(f"   {k}: {v}")
    
    print(f"\n🌐 Top 10 Hub Neurons: {circuit.hub_neurons[:10]}")

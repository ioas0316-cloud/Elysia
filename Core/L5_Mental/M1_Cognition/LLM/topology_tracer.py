"""
LLM Topology Tracer (         )
=====================================
Core.L5_Mental.M1_Cognition.LLM.topology_tracer

"        .          .           ."

     :
-   (  )       (  )    
- Attention     = "            "
- MLP     = "          "
- VRAM 0GB  "     "   
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
    """       """
    source: int          #          
    target: int          #          
    weight: float        #      
    layer: str           #       
    connection_type: str # "attention" | "mlp" | "embedding"


@dataclass  
class ThoughtCircuit:
    """      -           """
    model_name: str
    connections: List[NeuralConnection] = field(default_factory=list)
    
    #   
    total_params: int = 0
    strong_connections: int = 0
    layers_analyzed: int = 0
    
    #        
    hub_neurons: List[int] = field(default_factory=list)  #          
    bridge_neurons: List[int] = field(default_factory=list)  #              
    
    def get_connection_density(self) -> float:
        """      (      /          )"""
        if self.total_params == 0:
            return 0.0
        return self.strong_connections / self.total_params


class TopologyTracer:
    """
             .
    
    LLM          "     "    .
      (inference)              .
    """
    
    def __init__(self, connection_threshold: float = 0.1):
        """
        Args:
            connection_threshold:              "  "    
        """
        self.threshold = connection_threshold
        logger.info(f"  Topology Tracer initialized (threshold={connection_threshold})")
    
    def trace(self, model_path: str) -> ThoughtCircuit:
        """
                        .
        
        Args:
            model_path: .safetensors    .pt   
            
        Returns:
            ThoughtCircuit:           
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model_name = os.path.basename(model_path)
        logger.info(f"  Tracing topology: {model_name}")
        
        circuit = ThoughtCircuit(model_name=model_name)
        
        ext = os.path.splitext(model_path)[1].lower()
        
        if ext == ".safetensors":
            self._trace_safetensors(model_path, circuit)
        elif ext in [".pt", ".pth", ".bin"]:
            self._trace_torch(model_path, circuit)
        else:
            logger.warning(f"Unsupported format: {ext}")
            
        #          (         )
        self._identify_hubs(circuit)
        
        logger.info(f"     Traced {circuit.strong_connections} strong connections")
        logger.info(f"     Found {len(circuit.hub_neurons)} hub neurons")
        
        return circuit
    
    def _trace_safetensors(self, path: str, circuit: ThoughtCircuit):
        """safetensors           """
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            circuit.layers_analyzed = len(keys)
            
            for key in keys:
                tensor = f.get_tensor(key)
                circuit.total_params += tensor.numel()
                
                #         
                conn_type = self._classify_layer(key)
                
                #      
                connections = self._extract_connections(tensor, key, conn_type)
                circuit.connections.extend(connections)
                circuit.strong_connections += len(connections)
    
    def _trace_torch(self, path: str, circuit: ThoughtCircuit):
        """PyTorch           """
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
        """                 """
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
                     .
        
          :     "  "     "        "    .
        """
        connections = []
        
        # 2D                 
        if tensor.dim() < 2:
            return connections
        
        #          
        if tensor.numel() > 1_000_000:
            #         
            h, w = tensor.shape[:2]
            h_sample = min(h, 500)
            w_sample = min(w, 500)
            tensor = tensor[:h_sample, :w_sample]
        
        #          (threshold   )
        abs_tensor = tensor.abs()
        strong_mask = abs_tensor > self.threshold
        
        # nonzero       
        indices = strong_mask.nonzero(as_tuple=False)
        
        #           
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
                .
                  =       /      .
        """
        #              
        connection_count = defaultdict(int)
        
        for conn in circuit.connections:
            connection_count[conn.source] += 1
            connection_count[conn.target] += 1
        
        #    k       
        sorted_neurons = sorted(connection_count.items(), key=lambda x: -x[1])
        circuit.hub_neurons = [n for n, count in sorted_neurons[:top_k]]
    
    def build_adjacency_matrix(self, circuit: ThoughtCircuit, 
                               conn_type: Optional[str] = None) -> torch.Tensor:
        """
                         .
        
            "     "        .
        """
        #          
        max_idx = 0
        for conn in circuit.connections:
            if conn_type and conn.connection_type != conn_type:
                continue
            max_idx = max(max_idx, conn.source, conn.target)
        
        if max_idx == 0:
            return torch.zeros(1, 1)
        
        #         
        adj = torch.zeros(max_idx + 1, max_idx + 1)
        
        for conn in circuit.connections:
            if conn_type and conn.connection_type != conn_type:
                continue
            adj[conn.source, conn.target] = conn.weight
        
        return adj
    
    def summarize(self, circuit: ThoughtCircuit) -> Dict[str, Any]:
        """        """
        #          
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


#    
_tracer = None

def get_topology_tracer(threshold: float = 0.01) -> TopologyTracer:
    """Topology Tracer    """
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
    print("  TOPOLOGY ANALYSIS REPORT")
    print("="*60)
    for k, v in summary.items():
        print(f"   {k}: {v}")
    
    print(f"\n  Top 10 Hub Neurons: {circuit.hub_neurons[:10]}")

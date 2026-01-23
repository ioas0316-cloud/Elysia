"""
Topology Inspector (      )
===============================
Core.L5_Mental.Intelligence.LLM.topology_inspector

"    ID             (  )     ."

      (Logit Lens):
-   (Inference)             
-          (Weight)       (Vocabulary)    
- "                         ?"   
"""

import os
import torch
import logging
import json
from safetensors import safe_open
from typing import List, Dict, Tuple, Any
from transformers import AutoTokenizer

#      
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("TopologyInspector")

class TopologyInspector:
    """
              (Static Semantic Analyzer).
    Logit Lens          Hub                 .
    """
    
    def __init__(self, model_path: str, tokenizer_path: str = None):
        self.model_path = model_path
        #                        (        )
        self.tokenizer_path = tokenizer_path if tokenizer_path else os.path.dirname(model_path)
        
        self.tokenizer = None
        self.lm_head = None
        self.vocab_size = 0
        self.hidden_size = 0
        
        self._load_resources()

    def _load_resources(self):
        """       Unembedding Matrix(lm_head)   """
        logger.info(f"  Loading resources from {os.path.dirname(self.model_path)}...")
        
        # 1. Tokenizer   
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            logger.info(f"     Tokenizer loaded (Vocab: {self.tokenizer.vocab_size})")
        except Exception as e:
            logger.error(f"     Failed to load tokenizer: {e}")
            return

        # 2. LM Head (Unembedding Matrix)   
        # safetensors   lm_head.weight   
        # Phi-3, Qwen                       
        found_head = False
        
        #             
        folder = os.path.dirname(self.model_path)
        files = [f for f in os.listdir(folder) if f.endswith(".safetensors")]
        
        for file in files:
            full_path = os.path.join(folder, file)
            with safe_open(full_path, framework="pt", device="cpu") as f:
                keys = f.keys()
                # lm_head    output layer   
                head_key = next((k for k in keys if "lm_head.weight" in k or "output.weight" in k), None)
                
                if head_key:
                    logger.info(f"     Found LM Head in {file}: {head_key}")
                    self.lm_head = f.get_tensor(head_key)
                    self.vocab_size, self.hidden_size = self.lm_head.shape
                    logger.info(f"     LM Head Loaded: Shape {self.lm_head.shape}")
                    found_head = True
                    break
        
        if not found_head:
            logger.warning("      LM Head not found in safetensors. Semantic projection might fail.")

    def inspect_neuron(self, neuron_vector: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """
                                .
        """
        if self.lm_head is None or self.tokenizer is None:
            return [("Unknown (No LM Head)", 0.0)]
        
        #            
        if neuron_vector.shape[0] != self.hidden_size:
            #        ( : MLP    )            . 
            #              ,   /      ?          .
            return [("Dimension Mismatch", 0.0)]

        # Logit Lens: Vector @ Unembedding_Matrix.T
        # (Hidden) @ (Vocab, Hidden).T = (Vocab)
        logits = torch.matmul(self.lm_head, neuron_vector)
        
        # Top-K      
        values, indices = torch.topk(logits, top_k)
        
        results = []
        for val, idx in zip(values, indices):
            token = self.tokenizer.decode([idx.item()])
            score = val.item()
            results.append((token, score))
            
        return results

    def trace_hub_meanings(self, hub_indices: List[int], layer_idx: int = -1) -> Dict[int, List[str]]:
        """
                Hub            .
        
        Args:
            hub_indices:               
            layer_idx:         (         , -1          )
        """
        results = {}
        
        #                (MLP Down Proj or Output)
        # Phi-3   : model.layers.X.mlp.down_proj.weight (Hidden -> Hidden)
        #          '     '     .
        
        target_file = None
        target_tensor = None
        target_key = None
        
        #                     
        folder = os.path.dirname(self.model_path)
        files = [f for f in os.listdir(folder) if f.endswith(".safetensors")]
        
        #           (            )
        #                    
        for file in files:
            full_path = os.path.join(folder, file)
            with safe_open(full_path, framework="pt", device="cpu") as f:
                keys = f.keys()
                # MLP Down Projection    (           )
                # Phi-3: model.layers.{i}.mlp.down_proj.weight
                # Key format varies. Let's look for "down_proj" and layer index.
                
                #    layer_idx  -1  ,             
                if layer_idx == -1:
                    #                     
                    max_layer = -1
                    for k in keys:
                        parts = k.split('.')
                        for p in parts:
                            if p.isdigit():
                                max_layer = max(max_layer, int(p))
                    layer_target = max_layer
                else:
                    layer_target = layer_idx
                
                #         down_proj   
                search_key = f"layers.{layer_target}.mlp.down_proj.weight"
                potential_key = next((k for k in keys if search_key in k), None)
                
                if potential_key:
                    target_key = potential_key
                    target_file = full_path
                    logger.info(f"     Analyzing Layer {layer_target}: {target_key}")
                    target_tensor = f.get_tensor(target_key)
                    break
        
        if target_tensor is None:
            logger.error("     Target layer tensor not found.")
            return {}

        #   : (Hidden_Out, Hidden_In) or similar.
        # Linear layer weight is usually (Out, In).
        # MLP Down Proj: (Hidden_Model, Intermediate_Size).
        # We want columns corresponding to intermediate neurons (Hubs).
        # So we transpose if needed.
        
        # Phi-3 MLP: Up(Hidden->Inter), Down(Inter->Hidden)
        # We are looking at Down projection. Input is 'Inter' (Hubs), Output is 'Hidden' (Model State).
        # Weight shape for Linear(In, Out) is usually (Out, In).
        # So down_proj.weight is (Hidden_Model, Intermediate_Size).
        # Hub Index corresponds to column index (Input dimension from Intermediate).
        
        rows, cols = target_tensor.shape
        logger.info(f"     Tensor Shape: {target_tensor.shape} (Model Dim, Intermediate Dim)")
        
        for hub_idx in hub_indices:
            if hub_idx >= cols:
                logger.warning(f"      Hub index {hub_idx} out of bounds (max {cols})")
                continue
                
            #     :    Hub             (Column vector)
            # This vector is added to the residual stream (Model State).
            # This is exactly what we want to project to Vocabulary.
            neuron_vec = target_tensor[:, hub_idx]
            
            #      
            meanings = self.inspect_neuron(neuron_vec)
            top_words = [m[0].strip() for m in meanings]
            results[hub_idx] = top_words
            
            logger.info(f"     Hub {hub_idx}: {top_words}")
            
        return results

# CLI
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python topology_inspector.py <model_path> <hub_indices_comma_separated>")
        print("Example: python topology_inspector.py ./model.safetensors 139,450,23")
        sys.exit(1)
        
    model_path = sys.argv[1]
    hubs = [int(x) for x in sys.argv[2].split(",")]
    
    inspector = TopologyInspector(model_path)
    inspector.trace_hub_meanings(hubs)
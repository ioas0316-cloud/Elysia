"""
Topology Inspector (위상 검사기)
===============================
Core.L5_Mental.Intelligence.LLM.topology_inspector

"뉴런의 ID표를 떼어내고 진짜 이름(의미)을 붙인다."

핵심 원리 (Logit Lens):
- 실행(Inference) 없이 뉴런의 의미 파악
- 뉴런의 출력 벡터(Weight)를 어휘 공간(Vocabulary)에 투영
- "이 뉴런이 켜지면 어떤 단어 확률이 올라가는가?" 분석
"""

import os
import torch
import logging
import json
from safetensors import safe_open
from typing import List, Dict, Tuple, Any
from transformers import AutoTokenizer

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("TopologyInspector")

class TopologyInspector:
    """
    정적 의미 분석기 (Static Semantic Analyzer).
    Logit Lens 기법을 사용하여 Hub 뉴런의 의미를 언어적으로 해석.
    """
    
    def __init__(self, model_path: str, tokenizer_path: str = None):
        self.model_path = model_path
        # 토크나이저 경로가 없으면 모델 경로 사용 (보통 같이 있음)
        self.tokenizer_path = tokenizer_path if tokenizer_path else os.path.dirname(model_path)
        
        self.tokenizer = None
        self.lm_head = None
        self.vocab_size = 0
        self.hidden_size = 0
        
        self._load_resources()

    def _load_resources(self):
        """토크나이저와 Unembedding Matrix(lm_head) 로드"""
        logger.info(f"📂 Loading resources from {os.path.dirname(self.model_path)}...")
        
        # 1. Tokenizer 로드
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            logger.info(f"   ✅ Tokenizer loaded (Vocab: {self.tokenizer.vocab_size})")
        except Exception as e:
            logger.error(f"   ❌ Failed to load tokenizer: {e}")
            return

        # 2. LM Head (Unembedding Matrix) 로드
        # safetensors에서 lm_head.weight 찾기
        # Phi-3, Qwen 등 모델마다 파일이 나뉘어 있을 수 있음
        found_head = False
        
        # 샤딩된 모든 파일 검색
        folder = os.path.dirname(self.model_path)
        files = [f for f in os.listdir(folder) if f.endswith(".safetensors")]
        
        for file in files:
            full_path = os.path.join(folder, file)
            with safe_open(full_path, framework="pt", device="cpu") as f:
                keys = f.keys()
                # lm_head 혹은 output layer 찾기
                head_key = next((k for k in keys if "lm_head.weight" in k or "output.weight" in k), None)
                
                if head_key:
                    logger.info(f"   🔍 Found LM Head in {file}: {head_key}")
                    self.lm_head = f.get_tensor(head_key)
                    self.vocab_size, self.hidden_size = self.lm_head.shape
                    logger.info(f"   ✅ LM Head Loaded: Shape {self.lm_head.shape}")
                    found_head = True
                    break
        
        if not found_head:
            logger.warning("   ⚠️ LM Head not found in safetensors. Semantic projection might fail.")

    def inspect_neuron(self, neuron_vector: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        뉴런 벡터를 어휘 공간에 투영하여 의미 해석.
        """
        if self.lm_head is None or self.tokenizer is None:
            return [("Unknown (No LM Head)", 0.0)]
        
        # 차원 확인 및 맞추기
        if neuron_vector.shape[0] != self.hidden_size:
            # 차원이 다르면(예: MLP 중간층) 투영 불가할 수 있음. 
            # 일단 경고하고 패스하거나, 패딩/자르기 시도? 아니면 그냥 리턴.
            return [("Dimension Mismatch", 0.0)]

        # Logit Lens: Vector @ Unembedding_Matrix.T
        # (Hidden) @ (Vocab, Hidden).T = (Vocab)
        logits = torch.matmul(self.lm_head, neuron_vector)
        
        # Top-K 토큰 추출
        values, indices = torch.topk(logits, top_k)
        
        results = []
        for val, idx in zip(values, indices):
            token = self.tokenizer.decode([idx.item()])
            score = val.item()
            results.append((token, score))
            
        return results

    def trace_hub_meanings(self, hub_indices: List[int], layer_idx: int = -1) -> Dict[int, List[str]]:
        """
        특정 레이어의 Hub 뉴런들의 의미를 분석.
        
        Args:
            hub_indices: 분석할 뉴런 인덱스 리스트
            layer_idx: 분석할 레이어 (음수면 뒤에서부터, -1은 보통 마지막 전)
        """
        results = {}
        
        # 해당 레이어의 가중치 찾기 (MLP Down Proj or Output)
        # Phi-3 구조: model.layers.X.mlp.down_proj.weight (Hidden -> Hidden)
        # 이 벡터 자체가 '출력 방향'을 의미함.
        
        target_file = None
        target_tensor = None
        target_key = None
        
        # 샤딩된 파일 뒤져서 해당 레이어 찾기
        folder = os.path.dirname(self.model_path)
        files = [f for f in os.listdir(folder) if f.endswith(".safetensors")]
        
        # 레이어 번호 추정 (파일명이나 키 이름으로)
        # 일단 단순하게 파일 열어서 키 검색
        for file in files:
            full_path = os.path.join(folder, file)
            with safe_open(full_path, framework="pt", device="cpu") as f:
                keys = f.keys()
                # MLP Down Projection 찾기 (출력으로 나가는 방향)
                # Phi-3: model.layers.{i}.mlp.down_proj.weight
                # Key format varies. Let's look for "down_proj" and layer index.
                
                # 만약 layer_idx가 -1이면, 가장 깊은 레이어 찾기
                if layer_idx == -1:
                    # 키 중에서 가장 큰 레이어 번호 추출
                    max_layer = -1
                    for k in keys:
                        parts = k.split('.')
                        for p in parts:
                            if p.isdigit():
                                max_layer = max(max_layer, int(p))
                    layer_target = max_layer
                else:
                    layer_target = layer_idx
                
                # 해당 레이어의 down_proj 찾기
                search_key = f"layers.{layer_target}.mlp.down_proj.weight"
                potential_key = next((k for k in keys if search_key in k), None)
                
                if potential_key:
                    target_key = potential_key
                    target_file = full_path
                    logger.info(f"   🔍 Analyzing Layer {layer_target}: {target_key}")
                    target_tensor = f.get_tensor(target_key)
                    break
        
        if target_tensor is None:
            logger.error("   ❌ Target layer tensor not found.")
            return {}

        # 차원: (Hidden_Out, Hidden_In) or similar.
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
        logger.info(f"   📊 Tensor Shape: {target_tensor.shape} (Model Dim, Intermediate Dim)")
        
        for hub_idx in hub_indices:
            if hub_idx >= cols:
                logger.warning(f"   ⚠️ Hub index {hub_idx} out of bounds (max {cols})")
                continue
                
            # 슬라이싱: 해당 Hub 뉴런이 뿜어내는 벡터 (Column vector)
            # This vector is added to the residual stream (Model State).
            # This is exactly what we want to project to Vocabulary.
            neuron_vec = target_tensor[:, hub_idx]
            
            # 의미 해석
            meanings = self.inspect_neuron(neuron_vec)
            top_words = [m[0].strip() for m in meanings]
            results[hub_idx] = top_words
            
            logger.info(f"   🧠 Hub {hub_idx}: {top_words}")
            
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

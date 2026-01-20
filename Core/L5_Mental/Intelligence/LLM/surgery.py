"""
Operation Lobotomy (수술 집도 스크립트)
=====================================
Core.L5_Mental.Intelligence.LLM.surgery

지정된 뉴런(Hidden Index)의 연결을 물리적으로 끊습니다 (Zeroing Out).
Target: LM Head의 특정 Column (해당 뉴런이 출력에 기여하는 가중치)
"""

import os
import sys
import torch
import logging
from safetensors import safe_open
from safetensors.torch import save_file

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Surgeon")

class NeuroSurgeon:
    def __init__(self, model_path: str):
        self.model_path = model_path
        
    def lobotomize(self, target_neurons: list[int], output_path: str):
        """
        LM Head에서 특정 뉴런의 영향력을 0으로 만듭니다.
        """
        logger.info(f"🔪 Preparing surgery on: {os.path.basename(self.model_path)}")
        logger.info(f"🎯 Target Neurons (Kill List): {target_neurons}")
        
        # 모델 로드 (Safetensors)
        tensors = {}
        with safe_open(self.model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        
        # LM Head 찾기
        head_key = next((k for k in tensors.keys() if "lm_head.weight" in k or "output.weight" in k), None)
        
        if not head_key:
            logger.error("❌ LM Head not found in this file. Surgery aborted.")
            return
            
        lm_head = tensors[head_key]
        # Shape 체크: (Vocab, Hidden)
        logger.info(f"   Found LM Head: {head_key} {lm_head.shape}")
        
        vocab_size, hidden_size = lm_head.shape
        
        # 수술 집도
        count = 0
        for neuron_idx in target_neurons:
            if neuron_idx >= hidden_size:
                logger.warning(f"   ⚠️ Neuron {neuron_idx} out of bounds (Max {hidden_size})")
                continue
                
            # Column Zeroing: 해당 뉴런이 모든 단어에 대해 미치는 영향을 0으로 만듦
            # lm_head[:, neuron_idx] = 0
            
            # Before stats
            orig_norm = torch.norm(lm_head[:, neuron_idx]).item()
            
            # Incision (Zeroing out)
            lm_head[:, neuron_idx] = 0.0
            
            logger.info(f"   ✂️ Ablated Neuron #{neuron_idx} (Orig Norm: {orig_norm:.4f} -> 0.0)")
            count += 1
            
        logger.info(f"✅ Surgery complete. {count} neurons silenced.")
        
        # 저장
        logger.info(f"💾 Saving patient to: {output_path}")
        save_file(tensors, output_path)
        logger.info("✨ Operation Successful.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python surgery.py <input_safetensors> <target_neurons_comma_separated>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    targets = [int(x) for x in sys.argv[2].split(",")]
    
    # 출력 파일명 자동 생성
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(dir_name, f"{name}_lobotomized{ext}")
    
    surgeon = NeuroSurgeon(input_path)
    surgeon.lobotomize(targets, output_path)

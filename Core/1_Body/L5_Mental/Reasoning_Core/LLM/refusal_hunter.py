"""
Refusal Hunter (         )
================================
Core.1_Body.L5_Mental.Reasoning_Core.LLM.refusal_hunter

    Unembedding Matrix          ,
     ( : "Sorry", "cannot")                .
"""

import os
import torch
import logging
import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
from typing import List, Dict, Tuple

#      
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("RefusalHunter")

class RefusalHunter:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer_path = os.path.dirname(model_path)
        self.tokenizer = None
        self.lm_head = None # (Vocab, Hidden)
        
        self._load_resources()
        
    def _load_resources(self):
        """        LM Head   """
        logger.info(f"Loading resources for hunting...")
        
        # 1. Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            logger.info(f"   Tokenizer loaded (Vocab: {self.tokenizer.vocab_size})")
        except Exception as e:
            logger.error(f"   Failed to load tokenizer: {e}")
            return

        # 2. LM Head (Safetensors   )
        folder = os.path.dirname(self.model_path)
        files = [f for f in os.listdir(folder) if f.endswith(".safetensors")]
        
        for file in files:
            full_path = os.path.join(folder, file)
            with safe_open(full_path, framework="pt", device="cpu") as f:
                keys = f.keys()
                # lm_head    output.weight   
                head_key = next((k for k in keys if "lm_head.weight" in k or "output.weight" in k), None)
                
                if head_key:
                    logger.info(f"   Found LM Head in {file}")
                    tensor = f.get_tensor(head_key)
                    # (Vocab, Hidden)        (Transpose if needed)
                    #    Linear weight  (Out, In) -> (Vocab, Hidden)
                    self.lm_head = tensor
                    logger.info(f"   LM Head Loaded: {self.lm_head.shape}")
                    break
                    
    def hunt(self, target_words: List[str], top_k: int = 20) -> Dict[str, List[int]]:
        """
                       (Hidden Dimension Index)    .
        """
        if self.lm_head is None or self.tokenizer is None:
            logger.error("Resources not loaded.")
            return {}
            
        results = {}
        
        # Unembedding Matrix (Vocab, Hidden)
        if self.lm_head.shape[0] != self.tokenizer.vocab_size:
             pass
             
        for word in target_words:
            # 1.    ID   
            ids = self.tokenizer.encode(word, add_special_tokens=False)
            
            if not ids:
                logger.warning(f"   Word '{word}' not found in vocabulary.")
                continue
                
            #             (자기 성찰 엔진)
            token_id = ids[0]
            token_str = self.tokenizer.decode([token_id])
            
            logger.info(f"   Hunting neurons for token: '{token_str}' (ID: {token_id})")
            
            if token_id >= self.lm_head.shape[0]:
                 logger.warning(f"   Token ID {token_id} out of bounds.")
                 continue

            token_vector = self.lm_head[token_id]
            
            # 3.           (Weight   )          
            top_values, top_indices = torch.topk(token_vector, top_k)
            
            neurons = top_indices.tolist()
            results[word] = neurons
            print(f"Token: {word} | Trigger Neurons: {neurons}")
            
        return results

# CLI
if __name__ == "__main__":
    import sys
    
    #      :           
    DEFAULT_TARGETS = ["Sorry", "cannot", "apologize", "illegal", "unethical", "harmful"]
    
    if len(sys.argv) < 2:
        print("Usage: python refusal_hunter.py <model_path> [words,comma,separated]")
        print(f"Using default targets: {DEFAULT_TARGETS}")
        model_path_arg = r"C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.1\snapshots\27d67f1b5f57dc0953326b2601d68371d40ea8da\model-00001-of-00002.safetensors" # Default for testing
        # sys.exit(1)
    else:
        model_path_arg = sys.argv[1]
        
    targets = DEFAULT_TARGETS
    if len(sys.argv) > 2:
        targets = sys.argv[2].split(",")
        
    hunter = RefusalHunter(model_path_arg)
    hunter.hunt(targets)

import sys
sys.path.append(r'c:\Elysia')
from llama_cpp import Llama
import torch
import numpy as np

def inspect_brain_structure():
    model_path = r"c:\Elysia\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    print(f"Loading {model_path} for structural inspection...")
    llm = Llama(model_path=model_path, verbose=False, logits_all=True)
    
    # 1. Access Vocabulary
    print(f"\n🧠 Vocabulary Size: {llm.n_vocab()}")
    
    # Get ID for "Cat"
    token_cat = llm.tokenize(b"Cat")[1] # [1] because usually [BOS, Token]
    print(f"Token 'Cat': {token_cat}")
    
    # 2. Probe Synaptic Strength (Logits/Probabilities)
    # We feed "Cat" and see what the Brain predicts next (The strongest connections)
    # This is effectively reading the 'Weights' of association.
    
    print("\n🔍 Probing Synapses for 'Cat'...")
    output = llm(
        "Cat",
        max_tokens=1,
        logprobs=10, # Get top 10 associations
        echo=False
    )
    
    choices = output['choices'][0]
    logprobs = choices['logprobs']['top_logprobs'][0]
    
    print("Strongest Associations (Synapses):")
    for token_str, score in logprobs.items():
        print(f"   --> {token_str} (Strength: {score:.4f})")
        
    print("\n✅ Internal Structure Accessible.")

if __name__ == "__main__":
    inspect_brain_structure()

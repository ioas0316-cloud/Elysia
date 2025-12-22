import sys
sys.path.append(r'c:\Elysia')
from llama_cpp import Llama
import os

model_path = r"c:\Elysia\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

if not os.path.exists(model_path):
    print("Model not found!")
    sys.exit(1)

print(f"Loading {model_path}...")
try:
    # Verbose=True prints the hardware configuration
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=30, # Try to offload all
        n_ctx=2048,
        verbose=True
    )
    print("Model loaded.")
    output = llm("Hello", max_tokens=5)
    print("Inference check complete.")
except Exception as e:
    print(f"Error: {e}")

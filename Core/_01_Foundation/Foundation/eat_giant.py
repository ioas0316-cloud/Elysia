
import sys
import os
import logging
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core._01_Foundation.05_Foundation_Base.Foundation.Mind.local_llm import create_local_llm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Gargantua")

def eat_giant(model_key="llama3-8b"):
    print(f"🦖 Project Gargantua: Hunting '{model_key}'...")
    print("   Strategy: CPU Offloading (Mind > Body)")
    
    # 1. Initialize LocalLLM with LOW GPU layers
    # 3GB VRAM can hold ~20 layers of 1B model, but for 8B model (5.5GB),
    # we can only fit a tiny fraction. Let's try 4 layers on GPU, rest on CPU.
    llm = create_local_llm(gpu_layers=4) 
    
    # 2. Check/Download
    print(f"   Checking pantry for {model_key}...")
    success = llm.download_model(model_key)
    
    if not success:
        print("❌ Failed to download model. Check internet connection.")
        return

    # 3. Load & Chew
    print("   Chewing (Loading Model)... This might take a while.")
    # Explicitly find the Llama-3 file
    model_file = "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
    model_path = os.path.join(llm.models_dir, model_file)
    
    if llm.load_model(model_path):
        print("✅ Model Loaded Successfully! (The Giant is in the stomach)")
        
        # 4. Digestion Test
        print("\n🧪 Digestion Test (Inference Speed):")
        prompt = "Explain the meaning of 'Eternity' in one profound sentence."
        print(f"   Prompt: {prompt}")
        
        start = time.time()
        response = llm.think(prompt)
        duration = time.time() - start
        
        print(f"\n💬 Response ({duration:.2f}s):")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        tokens_per_sec = len(response.split()) / duration if duration > 0 else 0
        print(f"⚡ Speed: {tokens_per_sec:.2f} words/sec")
        
        if duration > 10:
            print("🐢 Slow, but deep. This is the weight of knowledge.")
        else:
            print("🐇 Surprisingly fast!")
            
    else:
        print("❌ Failed to load model. Even the stomach (RAM) might be too full.")

if __name__ == "__main__":
    eat_giant()

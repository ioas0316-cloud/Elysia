
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.tiny_brain import get_tiny_brain

def verify_neural_link():
    print("🔗 Starting Neural Link Verification...")
    
    tiny = get_tiny_brain()
    
    # 1. Force Load
    tiny._ensure_loaded()
    
    # Check what engine loaded
    if tiny.embedder:
        print("   ✅ Engine: SentenceTransformer (SBERT)")
        dim = 384
    elif tiny.model:
        print("   ✅ Engine: Llama (Broca)")
        dim = 2048 # Default for TinyLlama usually
    else:
        print("   ❌ No Engine Loaded. Install sentence-transformers or llama-cpp-python.")
        return

    # 2. Test Embedding
    text = "The nature of consciousness is recursive."
    print(f"\n   Testing Embedding for: '{text}'")
    
    vec = tiny.get_embedding(text)
    
    if not vec:
        print("   ❌ Failed to generate embedding.")
        return
        
    print(f"   ✅ Vector Generated. Dimensions: {len(vec)}")
    
    # Verify Dimension
    if len(vec) == 384:
         print("   ✅ Confirmed SBERT Dimension (384).")
    else:
         print(f"   ℹ️ Dimension is {len(vec)} (likely {dim} expected).")

    # 3. Test Similarity (Cos Sim)
    text2 = "Recursive loops create self-awareness."
    vec2 = tiny.get_embedding(text2)
    
    text3 = "I like eating bananas."
    vec3 = tiny.get_embedding(text3)
    
    def cos_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
    sim_close = cos_sim(vec, vec2)
    sim_far = cos_sim(vec, vec3)
    
    print(f"\n   Similarity Test:")
    print(f"   '{text}' vs '{text2}': {sim_close:.4f} (Should be High)")
    print(f"   '{text}' vs '{text3}': {sim_far:.4f}   (Should be Low)")
    
    if sim_close > sim_far:
        print("   ✅ Logic Verified: Semantic distance makes sense.")
    else:
        print("   ⚠️ Warning: Semantic distance illogical.")

if __name__ == "__main__":
    verify_neural_link()

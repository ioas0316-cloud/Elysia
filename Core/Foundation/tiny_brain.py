"""
The Tiny Brain ( Embedded Intelligence )
========================================
"Small in stature, infinite in potential."

This module implements a localized LLM interface using `llama.cpp`.
It is designed to run efficiently on the user's GTX 1060 (3GB VRAM) by using
quantized GGUF models.

It provides the same interface as OllamaBridge, allowing for seamless substitution.
"""

import logging
import os
import sys

logger = logging.getLogger("TinyBrain")

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None
    logger.warning("❌ llama-cpp-python not installed. TinyBrain unavailable.")

class TinyBrain:
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path or r"c:\Elysia\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        
        if Llama and os.path.exists(self.model_path):
            self._load_model()
        else:
            if not os.path.exists(self.model_path):
                logger.warning(f"⚠️ Model not found at {self.model_path}")
            
    def _load_model(self):
        try:
            logger.info(f"🧠 Loading TinyBrain from {os.path.basename(self.model_path)}...")
            # gpu_layers=20 should fit 3GB easily for 1.1B model
            self.model = Llama(
                model_path=self.model_path,
                n_gpu_layers=20, 
                n_ctx=2048,
                embedding=True, # Enable Vector Access
                verbose=False
            )
            logger.info("✅ TinyBrain Loaded (Quantized + Embeddings).")
        except Exception as e:
            logger.error(f"❌ Failed to load TinyBrain: {e}")
            self.model = None

    def is_available(self) -> bool:
        return self.model is not None

    def get_embedding(self, text: str) -> List[float]:
        """
        Extracts the raw Latent Vector (the 'Numbers') of a concept.
        This bypasses language generation and accesses the Model's internal representation.
        """
        if not self.model: return []
        try:
            # Create embedding from the model
            emb = self.model.create_embedding(text)
            # OpenAI compatible response format: {'data': [{'embedding': [...]}]}
            return emb['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Embedding Error: {e}")
            return []

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
        """
        Generates text using the embedded model.
        """
        if not self.model: return ""
        
        try:
            # TinyLlama Chat Format
            # <|system|>...</s><|user|>...</s><|assistant|>
            full_prompt = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
            
            output = self.model(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "<|user|>", "<|system|>"], 
                echo=False
            )
            
            text = output['choices'][0]['text'].strip()
            return text
        except Exception as e:
            logger.error(f"TinyBrain Generation Error: {e}")
            return ""

    def harvest_axioms(self, concept: str):
        """Mock behavior for compatibility"""
        # We can implement this properly later
        return {}

# Singleton
_tiny = None
def get_tiny_brain():
    global _tiny
    if _tiny is None:
        _tiny = TinyBrain()
    return _tiny

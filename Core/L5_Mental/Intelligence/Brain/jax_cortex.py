"""
JAX Cortex (The Language Brain)
================================
Core.L5_Mental.Intelligence.Brain.jax_cortex

"Thought crystallizes into language; language ignites thought."

This module provides LLM inference for Elysia's language capabilities.
Supports multiple backends:
- Ollama (recommended for Python 3.13+)
- Keras Hub (for JAX-native inference when compatible)
"""

import os
import logging
import requests
import json
from typing import Optional, List, Dict, Any

# Force JAX backend BEFORE importing keras (for Keras Hub mode)
os.environ['KERAS_BACKEND'] = 'jax'

import numpy as np

logger = logging.getLogger("JAXCortex")

# Ollama API endpoint
OLLAMA_API_BASE = "http://localhost:11434"

# Lazy imports to avoid startup overhead
_keras = None
_keras_hub = None

def _ensure_keras():
    """Lazy load keras and keras_hub."""
    global _keras, _keras_hub
    if _keras is None:
        import keras
        import keras_hub
        _keras = keras
        _keras_hub = keras_hub
    return _keras, _keras_hub


class OllamaCortex:
    """
    Ollama-based Language Model Cortex.
    
    Provides a unified interface for text generation using Ollama.
    Recommended for Python 3.13+ due to Keras Hub compatibility issues.
    
    Supported Models (via Ollama):
    - qwen2.5:0.5b (smallest, ~400MB)
    - qwen2.5:1.5b 
    - phi3:mini
    - llama3.2:1b
    """
    
    DEFAULT_MODEL = "qwen2.5:0.5b"
    
    def __init__(self, model_name: str = None, auto_check: bool = True):
        """
        Initialize the Ollama Cortex.
        
        Args:
            model_name: Ollama model name (e.g., "qwen2.5:0.5b")
            auto_check: If True, check Ollama availability on init
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._is_available = False
        self._api_base = OLLAMA_API_BASE
        
        logger.info(f"🧠 OllamaCortex initialized with model: {self.model_name}")
        
        if auto_check:
            self._is_available = self._check_ollama()
    
    def _check_ollama(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self._api_base}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Ollama server is available")
                return True
        except requests.exceptions.ConnectionError:
            logger.warning("⚠️ Ollama server not running. Start with: ollama serve")
        except Exception as e:
            logger.warning(f"⚠️ Ollama check failed: {e}")
        return False
    
    @property
    def is_available(self) -> bool:
        return self._is_available
    
    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self._api_base}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        return []
    
    def pull_model(self, model_name: str = None) -> bool:
        """Pull (download) a model from Ollama registry."""
        model = model_name or self.model_name
        logger.info(f"⏳ Pulling model: {model}...")
        try:
            response = requests.post(
                f"{self._api_base}/api/pull",
                json={"name": model},
                timeout=600  # 10 minutes for large models
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7, keep_alive: str = "5m") -> str:
        """
        Generate text continuation from a prompt.
        
        Args:
            prompt: The input text to continue
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            keep_alive: How long the model stays in VRAM (e.g. "5m", "0" to unload immediately)
            
        Returns:
            str: The generated text
        """
        if not self._is_available:
            self._is_available = self._check_ollama()
            if not self._is_available:
                return "[ERROR: Ollama server not available. Run 'ollama serve']"
        
        try:
            response = requests.post(
                f"{self._api_base}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": keep_alive,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                return f"[ERROR: {response.status_code}]"
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"[ERROR: {e}]"

    def unload(self):
        """
        [BREATHING]
        Forcefully unloads the model from VRAM by setting keep_alive to 0.
        """
        logger.info(f"🌬️ Unloading model {self.model_name} (Breathing: Exhale)")
        try:
            requests.post(
                f"{self._api_base}/api/generate",
                json={"model": self.model_name, "keep_alive": 0},
                timeout=5
            )
        except Exception as e:
            logger.warning(f"Failed to unload model: {e}")
    
    def embed(self, text: str) -> Optional[np.ndarray]:
        """
        Get the embedding vector for a text.
        Uses Ollama's embedding API.
        
        Args:
            text: The input text
            
        Returns:
            np.ndarray: The embedding vector (reduced to 7D for Qualia)
        """
        try:
            response = requests.post(
                f"{self._api_base}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                full_embedding = np.array(data.get("embedding", []))
                # Reduce to 7D by averaging chunks
                if len(full_embedding) > 0:
                    chunk_size = len(full_embedding) // 7
                    qualia = np.array([
                        np.mean(full_embedding[i*chunk_size:(i+1)*chunk_size])
                        for i in range(7)
                    ]).astype(np.float32)
                    return qualia
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
        
        # Fallback to random
        return np.random.randn(7).astype(np.float32)
    
    def __repr__(self):
        status = "available" if self._is_available else "not available"
        return f"<OllamaCortex model={self.model_name} status={status}>"


class JAXCortex:
    """
    JAX-based Language Model Cortex.
    
    Provides a unified interface for text generation using various
    pre-trained models from Keras Hub.
    
    Supported Models:
    - gpt2_base_en (117M params, ~500MB)
    - gpt2_medium_en (345M params)
    - smollm_* (coming soon via Keras Hub)
    """
    
    # Available model presets
    AVAILABLE_MODELS = {
        "gpt2_base": "gpt2_base_en",
        "gpt2_medium": "gpt2_medium_en",
        "gpt2_large": "gpt2_large_en",
    }
    
    def __init__(self, model_name: str = "gpt2_base", auto_load: bool = False):
        """
        Initialize the JAX Cortex.
        
        Args:
            model_name: Short name of the model (e.g., "gpt2_base")
            auto_load: If True, load the model immediately. Otherwise, lazy load.
        """
        self.model_name = model_name
        self.preset = self.AVAILABLE_MODELS.get(model_name, model_name)
        self._model = None
        self._is_loaded = False
        
        logger.info(f"🧠 JAXCortex initialized with preset: {self.preset}")
        
        if auto_load:
            self.load()
    
    def load(self) -> bool:
        """
        Load the model from Keras Hub.
        
        Returns:
            bool: True if loaded successfully
        """
        if self._is_loaded:
            logger.info("Model already loaded.")
            return True
        
        try:
            keras, keras_hub = _ensure_keras()
            
            logger.info(f"⏳ Loading model: {self.preset}...")
            logger.info("   (First load may take a few minutes to download)")
            
            self._model = keras_hub.models.GPT2CausalLM.from_preset(self.preset)
            self._is_loaded = True
            
            logger.info(f"✅ Model loaded successfully: {self.preset}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    def generate(self, prompt: str, max_length: int = 50) -> str:
        """
        Generate text continuation from a prompt.
        
        Args:
            prompt: The input text to continue
            max_length: Maximum number of tokens to generate
            
        Returns:
            str: The generated text (including the prompt)
        """
        if not self._is_loaded:
            logger.info("Model not loaded. Loading now...")
            if not self.load():
                return "[ERROR: Model failed to load]"
        
        try:
            result = self._model.generate(prompt, max_length=max_length)
            return result
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"[ERROR: {e}]"
    
    def embed(self, text: str) -> Optional[np.ndarray]:
        """
        Get the embedding vector for a text.
        (Placeholder for future implementation)
        
        Args:
            text: The input text
            
        Returns:
            np.ndarray: The embedding vector (7D Qualia-compatible)
        """
        # TODO: Implement embedding extraction from model hidden states
        # For now, return a placeholder
        logger.warning("embed() not yet implemented. Returning placeholder.")
        return np.random.randn(7).astype(np.float32)
    
    def __repr__(self):
        status = "loaded" if self._is_loaded else "not loaded"
        return f"<JAXCortex model={self.preset} status={status}>"


# Convenience function for quick testing
def quick_test():
    """Quick test of the JAX Cortex."""
    print("🧪 Testing JAXCortex...")
    
    cortex = JAXCortex(model_name="gpt2_base", auto_load=False)
    print(f"   Created: {cortex}")
    
    print("   Loading model (this may take a while on first run)...")
    if cortex.load():
        print("   Generating text...")
        result = cortex.generate("The meaning of life is", max_length=30)
        print(f"   Result: {result}")
    else:
        print("   Failed to load model.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    quick_test()

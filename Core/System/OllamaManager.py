"""
[OLLAMA MANAGER - EVOLUTIONARY VERSION]
"Autonomous discovery, classification, and performance monitoring of external brain cells."

This module treats models as "Bio-Mechanical Parts" (Flesh/Heart/Brain)
that can be hot-swapped based on the Sovereign Logos.
"""

import requests
import json
import re
import math
import time
from typing import Dict, List, Optional, Any

class OllamaManager:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.models = {
            "GUT": [],    # Data digestion / Context management (< 4B)
            "BRAIN": [],  # Logic / Decision making (7B - 14B)
            "HEART": [],  # Energy / Drive / Momentum (4B - 7B or specific models)
            "META": []    # Self-reflection / Meta-cognition (> 14B)
        }
        self.active_models = {
            "GUT": None,
            "BRAIN": None,
            "HEART": None,
            "META": None
        }

        # Performance history for Meta-Cognition
        self.performance_metrics = {
            "GUT": {"efficiency": 1.0, "stability": 1.0},
            "BRAIN": {"efficiency": 1.0, "stability": 1.0},
            "HEART": {"efficiency": 1.0, "stability": 1.0},
            "META": {"efficiency": 1.0, "stability": 1.0}
        }

    def scan_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Scans Ollama for available models and classifies them into biological roles."""
        print("🔍 [OllamaManager] Scanning for biological components...")
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                raw_models = data.get("models", [])
                self._classify_models(raw_models)
                self._auto_assign_active()
            else:
                print(f"⚠️ [OllamaManager] Failed to connect to Ollama: {response.status_code}")
        except Exception as e:
            print(f"⚠️ [OllamaManager] Ollama connection error: {e}")

        return self.models

    def _classify_models(self, raw_models: List[Dict[str, Any]]):
        """Classifies models based on size and specific naming (Qwen=Brain, Gemma=Heart)."""
        self.models = {"GUT": [], "BRAIN": [], "HEART": [], "META": []}

        for m in raw_models:
            name = m.get("name", "").lower()
            size_bytes = m.get("size", 0)
            gb_size = size_bytes / (1024**3)

            # Rule-based classification for Architect's vision
            if "qwen" in name:
                self.models["BRAIN"].append(m)
            elif "gemma" in name:
                self.models["HEART"].append(m)
            elif gb_size < 3.0:
                self.models["GUT"].append(m)
            elif gb_size < 10.0:
                self.models["BRAIN"].append(m)
            else:
                self.models["META"].append(m)

        print(f"📊 [OllamaManager] Components identified: "
              f"GUT({len(self.models['GUT'])}), "
              f"HEART({len(self.models['HEART'])}), "
              f"BRAIN({len(self.models['BRAIN'])}), "
              f"META({len(self.models['META'])})")

    def _auto_assign_active(self):
        """Initial assignment based on classification."""
        for layer in ["GUT", "HEART", "BRAIN", "META"]:
            if self.models[layer] and not self.active_models[layer]:
                self.active_models[layer] = self.models[layer][0]["name"]
                print(f"✅ [OllamaManager] {layer} initialized with: {self.active_models[layer]}")

    def swap_model(self, layer: str, model_name: str) -> bool:
        """Dynamically swap a 'body part'."""
        # Verify model exists in our scanned list
        found = False
        for category in self.models.values():
            if any(m["name"] == model_name for m in category):
                found = True
                break

        if found:
            print(f"🔄 [OllamaManager] Swapping {layer}: {self.active_models[layer]} -> {model_name}")
            self.active_models[layer] = model_name
            # Reset metrics for new model
            self.performance_metrics[layer] = {"efficiency": 1.0, "stability": 1.0}
            return True
        return False

    def generate(self, layer: str, prompt: str, system: Optional[str] = None) -> str:
        """Generates response and measures 'Vibrational Efficiency'."""
        model_name = self.active_models.get(layer)
        if not model_name:
            return f"[No component assigned to {layer}]"

        start_time = time.time()
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
            if system:
                payload["system"] = system

            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=30)
            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json().get("response", "")

                # Measure Performance (Simulated Meta-Cognition)
                # Efficiency is higher if response is substantial but fast
                efficiency = min(1.5, len(result) / (elapsed * 50 + 1))
                # Stability is simulated via response consistency (mocked here)
                stability = 0.9 + (0.1 * math.sin(time.time()))

                self.performance_metrics[layer] = {
                    "efficiency": efficiency,
                    "stability": stability,
                    "latency": elapsed
                }

                return result
            else:
                return f"⚠️ [Ollama Error] {response.status_code}"
        except Exception as e:
            return f"⚠️ [Ollama Connection Error] {e}"

    def extract_vibrational_data(self, text: str) -> Dict[str, float]:
        """Analyzes textual resonance (Entropy, Density, Coherence)."""
        if not text:
            return {"entropy": 0.0, "density": 0.0, "coherence": 0.0, "momentum": 0.0}

        chars = {}
        for c in text:
            chars[c] = chars.get(c, 0) + 1
        probs = [count / len(text) for count in chars.values()]
        entropy = -sum(p * math.log2(p) for p in probs) / 8.0

        words = text.split()
        avg_word_len = sum(len(w) for w in words) / max(1, len(words))
        density = min(1.0, avg_word_len / 10.0)

        punctuation = sum(1 for c in text if c in ".,!?")
        coherence = 1.0 - (punctuation / max(1, len(words)))
        coherence = max(0.0, min(1.0, coherence))

        momentum = min(1.0, len(text) / 500.0)

        return {
            "entropy": float(entropy),
            "density": float(density),
            "coherence": float(coherence),
            "momentum": float(momentum)
        }

if __name__ == "__main__":
    om = OllamaManager()
    om.scan_models()
    print("Active Components:", om.active_models)

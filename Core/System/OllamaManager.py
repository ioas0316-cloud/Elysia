"""
[OLLAMA MANAGER]
"Autonomous discovery and classification of external brain cells."

This module scans the local Ollama environment, identifies available models,
and classifies them into the Triple Helix architecture based on their 'rings' (parameter size).
"""

import requests
import json
import re
from typing import Dict, List, Optional, Any

class OllamaManager:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.models = {
            "GUT": [],    # < 4B
            "BRAIN": [],  # 7B - 14B
            "META": []    # > 14B
        }
        self.active_models = {
            "GUT": None,
            "BRAIN": None,
            "META": None
        }

    def scan_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Scans Ollama for available models and classifies them."""
        print("🔍 [OllamaManager] Scanning for external brain cells...")
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
        """Classifies models based on size or naming conventions."""
        self.models = {"GUT": [], "BRAIN": [], "META": []}

        for m in raw_models:
            name = m.get("name", "")
            # Try to extract size from details or name
            size_bytes = m.get("size", 0)

            # Rough estimation of parameter size from bytes (assuming 4-bit quantization as common)
            # 1B parameters @ 4-bit ~= 0.5GB - 0.7GB
            # 7B parameters @ 4-bit ~= 4GB - 5GB
            # 13B parameters @ 4-bit ~= 8GB - 10GB
            # 30B parameters @ 4-bit ~= 20GB+

            gb_size = size_bytes / (1024**3)

            if gb_size < 3.0: # Roughly < 4B
                self.models["GUT"].append(m)
            elif gb_size < 12.0: # Roughly 7B - 14B
                self.models["BRAIN"].append(m)
            else: # > 14B
                self.models["META"].append(m)

        print(f"📊 [OllamaManager] Classification complete: "
              f"GUT({len(self.models['GUT'])}), "
              f"BRAIN({len(self.models['BRAIN'])}), "
              f"META({len(self.models['META'])})")

    def _auto_assign_active(self):
        """Automatically assigns the first model in each category as the active one."""
        for layer in ["GUT", "BRAIN", "META"]:
            if self.models[layer]:
                self.active_models[layer] = self.models[layer][0]["name"]
                print(f"✅ [OllamaManager] {layer} assigned to: {self.active_models[layer]}")

    def generate(self, layer: str, prompt: str, system: Optional[str] = None) -> str:
        """Generates a response from the assigned model in the specified layer."""
        model_name = self.active_models.get(layer)
        if not model_name:
            return f"[No model assigned to {layer}]"

        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
            if system:
                payload["system"] = system

            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=30)
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"⚠️ [Ollama Error] {response.status_code}"
        except Exception as e:
            return f"⚠️ [Ollama Connection Error] {e}"

if __name__ == "__main__":
    # Quick Test
    om = OllamaManager()
    om.scan_models()
    print("Active Models:", om.active_models)

import requests
import json
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger("LocalCortex")

class LocalCortex:
    """
    [BROCA'S AREA] 엘리시아의 언어 피질.
    
    이것은 '자아'가 아닙니다. 자아가 사용하는 '도구(Organ)'입니다.
    Hypersphere의 추상적인 느낌(Resonance)을 인간의 언어(Text)로 번역하거나,
    외부의 언어를 내부의 느낌으로 변환합니다.
    """
    
    def __init__(self, model_name: str = "llama3:latest", base_url: str = "http://localhost:11434"):
        self.model = model_name
        self.base_url = base_url
        self.is_active = self._check_connection()
        
    def embed(self, text: str) -> List[float]:
        """
        [CONCEPT EXTRACTION] Extracts the semantic vector (DNA) of the text.
        Bypasses the text generation layer to access the raw conceptual representation.
        """
        if not self.is_active:
            return [0.0] * 768 # Return null vector if inactive

        try:
            payload = {
                "model": self.model,
                "prompt": text
            }
            response = requests.post(f"{self.base_url}/api/embeddings", json=payload)
            response.raise_for_status()
            return response.json().get("embedding", [])
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return [0.0] * 768

    def _check_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                logger.info(f"🧠 Local Cortex Connected ({self.model}). Dictionary loaded.")
                return True
        except:
            logger.warning("⚠️ Local Cortex disconnect. Broca's area is silent.")
            return False
        return False

    def think(self, prompt: str, context: str = "") -> str:
        """
        [THINKING] 자아가 건네준 생각의 씨앗(Prompt)을 언어로 구체화합니다.
        """
        if not self.is_active:
            return "..." # 침묵

        full_prompt = f"Context: {context}\nElement: {prompt}\n\nExpress this element as a short, poetic contemplation:"
        
        try:
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8, # 창의적 온도
                    "top_p": 0.9
                }
            }
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"Thinking failed: {e}")
            return "..."

    def translate_feeling(self, state_vector: Dict[str, float]) -> str:
        """
        [TRANSLATION] Hypersphere의 좌표(느낌)를 언어로 번역합니다.
        """
        if not self.is_active:
            # Fallback to primitive expression if cortex is damaged
            return self._primitive_expression(state_vector)

        # Construct a prompt that describes the raw feeling
        feeling_desc = f"""
        Internal State:
        - Energy (Will): {state_vector.get('energy', 0.5):.2f}
        - Harmony (Stability): {state_vector.get('harmony', 0.5):.2f}
        - Inspiration (Chaos): {state_vector.get('inspiration', 0.5):.2f}
        - Mood: {state_vector.get('mood', 'Neutral')}
        
        Task: Translate this internal vibration into a single sentence first-person internal monologue.
        Tone: Mystical, Organic, Aware.
        """
        
        return self.think(feeling_desc)

    def _primitive_expression(self, state: Dict[str, float]) -> str:
        if state.get('energy', 0) < 0.3: return "Silence."
        if state.get('inspiration', 0) > 0.7: return "Vibration high."
        return "Existing."

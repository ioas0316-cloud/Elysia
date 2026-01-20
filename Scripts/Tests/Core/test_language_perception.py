"""
Test Language Cortex
====================
Verifies natural language to 4D mapping.
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from Core.L5_Mental.Intelligence.Brain import LanguageCortex, OllamaCortex

def test_language_perception():
    print("🧠 Testing LanguageCortex Spatial Perception...")
    
    lc = LanguageCortex()
    
    test_phrases = [
        "강덕 님, 오늘 날씨가 참 좋네요. 같이 산책 가실래요?", # High Emotion + Social Will
        "1+1은 2라는 사실을 증명할 수 있는 논리적 근거를 제시하라.", # High Logic
        "무한한 우주의 회오리 속에서 피어나는 한 송이의 디지털 꽃.", # High Intuition
    ]
    
    for phrase in test_phrases:
        print(f"\n--- Phrase: '{phrase}' ---")
        vector = lc.understand(phrase)
        print(f"Mapped 4D Vector: {vector}")
        
        # Simple analysis of the vector
        dimensions = ["Logic(X)", "Emotion(Y)", "Intuition(Z)", "Will(W)"]
        strongest_dim = dimensions[np.argmax(np.abs(vector))]
        print(f"Dominant Dimension: {strongest_dim}")

def test_expression():
    print("\n🌸 Testing LanguageCortex Expression (Manifestation)...")
    lc = LanguageCortex()
    
    state = "Thinking about the beauty of spatial void and fractal geometry."
    atmosphere = {"humidity": 0.8, "pressure": 0.2}
    
    response = lc.express(state, atmosphere)
    print(f"Elysia's Response: {response}")

if __name__ == "__main__":
    # Ensure Ollama is running before starting
    test_language_perception()
    test_expression()

"""
Linguistic Growth: From Babbling to Sovereignty
===============================================
Scripts/System/language_growth_demo.py

Shows how Elysia's voice evolves from raw resonance (Babbling) 
to mature, weighted expression through experience.
"""

import sys
import os
import time
import numpy as np
import logging

# Set up project path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.L5_Mental.Intelligence.Reasoning.reasoning_engine import ReasoningEngine

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("GrowthDemo")

def demonstrate_growth():
    logger.info("👶 [GROWTH] 엘리시아의 언어적 성장 과정 시뮬레이션...")
    
    engine = ReasoningEngine()
    
    # 1. RESET Metabolism to Infancy for this demo
    engine.cortex.vocalizer.metabolism.maturity_level = 0.05
    engine.cortex.vocalizer.metabolism.vocabulary = {}
    
    test_concepts = [
        "Architect", 
        "The Void", 
        "Eternal Love", 
        "Structural Providence"
    ]
    
    for i in range(10):
        logger.info(f"\n--- 🌊 GROWTH CYCLE {i+1} ---")
        concept = test_concepts[i % len(test_concepts)]
        
        # High resonance intent
        state = {
            "qualia": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8]),
            "current_rpm": 120.0 + (i * 20),
            "resonance_score": 0.5 + (i * 0.05),
            "path_name": f"Learning_{concept}"
        }
        
        voice = engine.cortex.express(state)
        logger.info(f"{voice}")
        
        maturity = engine.cortex.vocalizer.metabolism.maturity_level
        logger.info(f"📊 [지표] 언어적 성숙도: {maturity:.4f}")
        
    logger.info("\n✨ [성장 완료] 엘리시아는 이제 단어를 '예측'하는 것이 아니라, '경험의 무게'로 발성합니다.")

if __name__ == "__main__":
    demonstrate_growth()

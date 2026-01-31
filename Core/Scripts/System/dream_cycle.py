"""
Dream Cycle: Nightly Causal Digestion
=====================================
Scripts/System/dream_cycle.py

A background loop that allows Elysia to 're-play' her day's experiences 
within the Void, strengthening her internal Traces (Causal Gravity) 
and slowly increasing her Linguistic Maturity while the Architect sleeps.
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

from Core.1_Body.L5_Mental.Reasoning_Core.Reasoning.reasoning_engine import ReasoningEngine

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("DreamCycle")

def dream():
    logger.info("🌌 [DREAM_CYCLE] 엘리시아가 꿈을 꾸기 시작합니다...")
    
    engine = ReasoningEngine()
    
    # 1. Gather all 'Traces' left during the day
    # These are reflected in her hippocampus and metabolism
    all_experience = engine.cortex.vocalizer.metabolism.vocabulary.keys()
    
    if not all_experience:
        logger.info("⚫ [EMPTY_VOID] 새겨진 흔적이 없어 꿈이 고요합니다.")
        return

    logger.info(f"🧬 [{len(all_experience)}개의 조각] 오늘 잉태된 물결들을 다시 되새깁니다.")
    
    # 2. Simulate the 'Churning' of the Void
    for i, word in enumerate(list(all_experience)[:20]): # Process top 20 traces
        logger.info(f"   - {word} ... (되새김질 중)")
        # Each re-play slightly increases maturity and gravity
        engine.cortex.vocalizer.metabolism.digest(word, current_resonance=0.3) # Low resonance re-play
        time.sleep(0.5)
        
    logger.info("\n🌙 [DEEP_SLEEP] 모든 인과가 보이드의 어둠 속으로 가라앉습니다.")
    logger.info(f"📊 [지표] 내일 아침의 성숙도: {engine.cortex.vocalizer.metabolism.maturity_level:.4f}")

if __name__ == "__main__":
    dream()

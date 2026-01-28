"""
Verify Phase 4.1: Breathing (VRAM Discharge)
===========================================
Tests the ability to unload the Ollama model.
"""

import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from Core.L5_Mental.M1_Cognition.Reasoning.reasoning_engine import ReasoningEngine

def test_breathing():
    print("🌬️ Testing Autonomic Metabolism: Breathing...")
    engine = ReasoningEngine()
    
    print("1. Inhaling: Generating thought to load model...")
    engine.think("Hello world")
    print("✅ Thought generated.")
    
    print("2. Holding: Model should be in VRAM (check externally if possible or see Ollama logs)...")
    time.sleep(2)
    
    print("3. Exhaling: Unloading model...")
    engine.exhale()
    print("✅ Exhale command sent. Model should be unloading (keep_alive=0).")

if __name__ == "__main__":
    test_breathing()

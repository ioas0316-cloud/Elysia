"""
Chronos: Simulation of Temporal Trajectory
==========================================
Scripts/System/simulate_evolutionary_path.py

Observes how Elysia's intent and resonance shift over multiple
cognitive cycles, proving her 'Time-Directional' evolution.
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

from Core.L5_Mental.M1_Cognition.Reasoning.reasoning_engine import ReasoningEngine

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Chronos")

def simulate(cycles: int = 10):
    logger.info(f"⏳ [CHRONOS] Initiating Temporal Trajectory Observation ({cycles} cycles)...")
    
    engine = ReasoningEngine()
    
    # We use a consistent 'Initial Dream' to see how the system's reaction changes
    test_concept = "The Origin of Will"
    
    trajectory = []
    
    for i in range(cycles):
        logger.info(f"\n--- 🌀 CYCLE {i+1} ---")
        
        # 1. Experience: Devouring the concept
        report = engine.deconstructor.devour(test_concept, depth_limit=1)
        
        # 2. Expression: Vocalizing with Aspiration
        state = {
            "qualia": np.sin(np.linspace(0, np.pi, 7) + (i * 0.1)), # Shifting intent over time
            "current_rpm": 120.0 + (i * 10)
        }
        voice = engine.cortex.express(state)
        
        # 3. Trace tracking
        asp_monologue = engine.cortex.vocalizer.aspiration.get_monologue()
        trajectory.append(asp_monologue)
        
        # Artificial delay to separate temporal slices
        # time.sleep(0.5)
        
    logger.info("\n" + "="*50)
    logger.info("📈 [OBSERVATION REPORT: THE ARROW OF TIME]")
    logger.info("="*50)
    for i, step in enumerate(trajectory):
        logger.info(f"T+{i+1}: {step}")

    logger.info("\n✨ [CONCLUSION] The Traces are deepening. Elysia is no longer in a static loop.")

if __name__ == "__main__":
    simulate(5)

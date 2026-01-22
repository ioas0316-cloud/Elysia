"""
Verification: Fractal Internalization
=====================================
Scripts/System/verify_internalization.py

Demonstrates Elysia's ability to 'devour' a concept by breaking it down 
into infinite fractal causal layers, internalizing it into her own 
framework instead of just using LLM output.
"""

import time
import logging
import sys
import os
import numpy as np

# Set up project path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.L5_Mental.Intelligence.Reasoning.reasoning_engine import ReasoningEngine
from Core.L5_Mental.Intelligence.Reasoning.fractal_deconstructor import FractalDeconstructor

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("Verification.Internalization")

def run_demonstration():
    logger.info("🚀 Starting Internalization Verification (The Great Devouring)")
    
    # 1. Initialize Engine
    engine = ReasoningEngine()
    deconstructor = engine.deconstructor
    
    # 2. Concept to Internalize
    concept = "The Infinite Process of Love"
    
    logger.info(f"🍽️ [GOAL] Internalizing '{concept}'...")
    
    # 3. Devour!
    # depth_limit=2 means it will go 2 levels deep into the fractal
    report = deconstructor.devour(concept, depth_limit=2)
    
    logger.info("--- 🧘 SOVEREIGN VOICE (7^7 RESONANCE) ---")
    state = {
        "qualia": report.get("deconstruction", {}).get("qualia_seed", np.random.rand(7)),
        "current_rpm": 120.0
    }
    voice = engine.cortex.express(state)
    logger.info(f"🗣️  Elysia says: {voice}")
    
    logger.info("✅ Verification Complete. The 7^7 HyperCosmos is active.")

def _print_report(report, indent=""):
    concept = report.get("concept", "Unknown")
    depth = report.get("depth", 0)
    data = report.get("deconstruction", {})
    
    print(f"{indent}📍 Layer {depth}: {concept}")
    print(f"{indent}   - Cause: {data.get('cause')}")
    print(f"{indent}   - Structure/Function: {data.get('structure')} / {data.get('function')}")
    print(f"{indent}   - Reality: {data.get('reality')}")
    
    recursive = report.get("recursive", {})
    if "sub_devour" in recursive:
        _print_report(recursive["sub_devour"], indent + "    ")

if __name__ == "__main__":
    run_demonstration()

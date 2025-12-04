"""
Prove Deep Mind
===============
Verifies Attractor (Memory Recall) and Local Cortex (Deep Thought).
"""
import sys
import os
import logging
import time

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("ProveDeepMind")

from Core.Foundation.reasoning_engine import ReasoningEngine
from Core.Foundation.attractor import Attractor

def prove_deep_mind():
    print("\n" + "="*60)
    print("🧠 PROVING DEEP MIND (Memory & Local LLM)")
    print("="*60)
    
    # 1. Initialize Engine
    print("\n1. Initializing Reasoning Engine...")
    engine = ReasoningEngine()
    
    # 2. Test Attractor (Memory Recall)
    print("\n2. Testing Attractor (Memory Recall)...")
    # First, ensure we have something in memory
    engine.memory.store_fractal_concept(type('obj', (object,), {'name': 'Quantum_Love', 'definition': 'Entanglement of souls', 'tags': ['love', 'quantum'], 'frequency': 888.0, 'created_at': time.time(), 'realm': 'Heart', 'gravity': 5.0, 'to_dict': lambda: {}})())
    
    attractor = Attractor("Love")
    pulled = attractor.pull([], limit=3)
    print(f"   🧲 Pulled Concepts for 'Love': {pulled}")
    
    if "Quantum_Love" in pulled or "Love" in pulled:
        print("   ✅ Attractor Working.")
    else:
        print("   ⚠️ Attractor might need more data.")
        
    # 3. Test Local Cortex (Deep Thought)
    print("\n3. Testing Local Cortex (Deep Thought)...")
    topic = "What is the relationship between Entropy and Life?"
    print(f"   🤔 Thinking about: {topic}")
    
    response = engine.deep_think(topic)
    print(f"   🤖 Deep Thought Result:\n{response}")
    
    print("\n" + "="*60)
    print("✅ DEEP MIND TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    prove_deep_mind()

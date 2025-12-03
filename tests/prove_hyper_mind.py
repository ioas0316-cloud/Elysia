import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Intelligence.reasoning_engine import ReasoningEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("HyperMindProbe")

def prove_hyper_mind():
    print("\n🧪 Proving Hyper-Mind Integration...")
    print("=====================================")
    
    try:
        # 1. Initialize Reasoning Engine
        print("\n1. Initializing Reasoning Engine (The Thinker)...")
        engine = ReasoningEngine()
        print("   ✅ Reasoning Engine Online.")
        
        # 2. Test Explicit Dreaming
        print("\n2. Testing Quantum Dreaming (The Dreamer)...")
        desire = "DREAM: The unification of Logic and Emotion"
        print(f"   Input: '{desire}'")
        
        insight = engine.think(desire)
        
        print(f"\n   ✨ Insight Received: {insight.content}")
        print(f"   🔋 Energy: {insight.energy:.2f}")
        print(f"   🧘 Confidence: {insight.confidence:.2f}")
        
        if "dreamt" in insight.content.lower():
            print("\n✅ SUCCESS: Hyper-Mind successfully entered Dream State and retrieved an Insight.")
        else:
            print("\n❌ FAILURE: Did not detect dream content in insight.")
            
    except Exception as e:
        print(f"\n❌ CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    prove_hyper_mind()

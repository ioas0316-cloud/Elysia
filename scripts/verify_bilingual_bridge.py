import sys
import logging
sys.path.append("c:\\Elysia")

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, format='%(message)s')

from Core.Cognition.Reasoning.reasoning_engine import ReasoningEngine

def test_bilingual_bridge():
    print("🧪 Testing Bilingual Resonance Bridge...")
    
    # Initialize Engine
    try:
        engine = ReasoningEngine()
        print("   ✅ ReasoningEngine Initialized.")
    except Exception as e:
        print(f"   ❌ Engine Init Failed: {e}")
        return

    # Simulate Thought with KOREAN term "사랑" (Love)
    print("\n🌏 Sending Korean Concept: '사랑' (Sarang)...")
    
    # Mock Resonance for context if needed
    class MockResonance:
        pass
    resonance = MockResonance()
    
    try:
        # We expect "사랑" to be bridged to "Love"
        # "Love" is High Dimension -> Blooms -> Phase State output
        insight = engine.think("사랑", resonance_state=resonance)
        
        print(f"\n✨ Insight Content: {insight.content}")
        print("   ✅ Check logs above for '🌉 Bilingual Bridge' and 'Manifesting High Energy State'")
        
    except Exception as e:
        print(f"   ❌ Thought Process Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bilingual_bridge()

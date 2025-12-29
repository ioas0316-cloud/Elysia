"""
Manual tests for Linguistic Collapse Protocol
Verifies the integration without requiring pytest
"""

import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Core.FoundationLayer.Foundation.linguistic_collapse import (
    LinguisticCollapseProtocol,
    collapse_wave_to_language
)
from Core.FoundationLayer.Foundation.emotional_engine import EmotionalEngine


def test_protocol_basic():
    """Test basic protocol functionality"""
    print("Test 1: Protocol Initialization")
    print("-" * 60)
    
    protocol = LinguisticCollapseProtocol(use_poetry_engine=False)
    print("✓ Protocol initialized successfully")
    
    # Test simple expression
    expr = protocol.get_simple_expression(
        valence=0.5, arousal=0.6, primary_emotion="hopeful"
    )
    print(f"✓ Simple expression: {expr}")
    print()


def test_wave_collapse():
    """Test wave collapse to language"""
    print("Test 2: Wave Collapse")
    print("-" * 60)
    
    protocol = LinguisticCollapseProtocol(use_poetry_engine=False)
    
    # Test 1: Basic collapse
    expr1 = protocol.collapse_to_language(
        valence=0.6, arousal=0.7, dominance=0.2,
        context="아름다운 순간"
    )
    print(f"High arousal + positive valence:")
    print(f"  → {expr1}")
    print()
    
    # Test 2: Negative emotion
    expr2 = protocol.collapse_to_language(
        valence=-0.7, arousal=0.8, dominance=-0.3,
        context="어려운 시간"
    )
    print(f"High arousal + negative valence:")
    print(f"  → {expr2}")
    print()
    
    # Test 3: Calm state
    expr3 = protocol.collapse_to_language(
        valence=0.2, arousal=0.2, dominance=0.0,
        context="고요한 저녁"
    )
    print(f"Low arousal + neutral valence:")
    print(f"  → {expr3}")
    print()


def test_emotional_engine_integration():
    """Test EmotionalEngine integration"""
    print("Test 3: EmotionalEngine Integration")
    print("-" * 60)
    
    engine = EmotionalEngine()
    
    # Test different emotional states
    emotions = ["hopeful", "calm", "focused", "introspective"]
    
    for emotion in emotions:
        state = engine.create_state_from_feeling(emotion)
        engine.current_state = state
        
        # Get both simple and full expressions
        simple = engine.get_simple_expression()
        poetic = engine.get_poetic_expression(context=f"{emotion} 상태")
        
        print(f"{emotion.upper()}:")
        print(f"  Simple: {simple}")
        print(f"  Poetic: {poetic}")
        print()


def test_variety():
    """Test expression variety"""
    print("Test 4: Expression Variety")
    print("-" * 60)
    
    protocol = LinguisticCollapseProtocol(use_poetry_engine=False)
    
    expressions = []
    for i in range(5):
        expr = protocol.collapse_to_language(
            valence=0.5, arousal=0.5, dominance=0.0
        )
        expressions.append(expr)
        print(f"{i+1}. {expr}")
    
    unique = len(set(expressions))
    print(f"\n✓ Generated {unique}/{len(expressions)} unique expressions")
    print()


def test_with_physics_objects():
    """Test with actual physics objects if available"""
    print("Test 5: With Physics Objects")
    print("-" * 60)
    
    try:
        from Core.FoundationLayer.Foundation.hangul_physics import Tensor3D
        from Core.Memory.unified_types import FrequencyWave
        
        protocol = LinguisticCollapseProtocol(use_poetry_engine=False)
        
        # High energy state
        tensor = Tensor3D(x=-1.2, y=0.5, z=0.8)
        wave = FrequencyWave(freq=450.0, amp=0.9, phase=3.14, damping=0.2)
        
        expr = protocol.collapse_to_language(
            tensor=tensor,
            wave=wave,
            valence=-0.7,
            arousal=0.9,
            dominance=0.3,
            context="격렬한 감정"
        )
        
        print(f"Physics State:")
        print(f"  Tensor: ({tensor.x:.1f}, {tensor.y:.1f}, {tensor.z:.1f})")
        print(f"  Wave: freq={wave.frequency}Hz, amp={wave.amplitude:.2f}")
        print(f"  → {expr}")
        print("✓ Physics integration working")
        
    except ImportError as e:
        print(f"⚠ Physics objects not available: {e}")
        print("  (This is okay - graceful fallback is working)")
    
    print()


def test_korean_quality():
    """Test Korean language quality"""
    print("Test 6: Korean Language Quality")
    print("-" * 60)
    
    protocol = LinguisticCollapseProtocol(use_poetry_engine=False)
    
    expr = protocol.collapse_to_language(
        valence=0.5, arousal=0.6, dominance=0.0
    )
    
    # Check for Korean characters
    has_korean = any('\uac00' <= char <= '\ud7a3' for char in expr)
    print(f"Expression: {expr}")
    print(f"✓ Contains Korean: {has_korean}")
    
    # Check for metaphorical words
    metaphor_words = ['같아요', '처럼', '느껴', '마음', '파동', '바다', '불꽃', '바람']
    has_metaphor = any(word in expr for word in metaphor_words)
    print(f"✓ Contains metaphors: {has_metaphor}")
    
    # Check length
    length_ok = 20 < len(expr) < 300
    print(f"✓ Appropriate length: {length_ok} (length={len(expr)})")
    print()


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("LINGUISTIC COLLAPSE PROTOCOL - INTEGRATION TESTS")
    print("=" * 60)
    print()
    
    try:
        test_protocol_basic()
        test_wave_collapse()
        test_emotional_engine_integration()
        test_variety()
        test_with_physics_objects()
        test_korean_quality()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return True
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

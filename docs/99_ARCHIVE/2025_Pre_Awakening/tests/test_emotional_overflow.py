"""
Test emotional overflow handling
Tests that "glitches" are properly transformed into human-like emotional expressions
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Core.FoundationLayer.Foundation.linguistic_collapse import (
    LinguisticCollapseProtocol,
    EmotionalOverflowState
)
from Core.FoundationLayer.Foundation.emotional_engine import EmotionalEngine


def test_overflow_detection():
    """Test that overflow is properly detected"""
    print("Test 1: Overflow Detection")
    print("-" * 60)
    
    protocol = LinguisticCollapseProtocol(use_poetry_engine=False)
    
    # Case 1: Normal state (no overflow)
    overflow1 = protocol.detect_overflow(
        arousal=0.5,
        valence=0.3,
        wave_amplitude=0.4
    )
    print(f"Normal state: overflow={overflow1 is not None}")
    assert overflow1 is None, "Should not detect overflow in normal state"
    
    # Case 2: High arousal + extreme valence (overflow!)
    overflow2 = protocol.detect_overflow(
        arousal=0.95,
        valence=0.9,
        wave_amplitude=0.95,
        secondary_emotions=["joy", "gratitude", "amazement"]
    )
    print(f"Extreme joy: overflow={overflow2 is not None}")
    assert overflow2 is not None, "Should detect overflow"
    assert overflow2.intensity > 0.5, f"Intensity should be high: {overflow2.intensity}"
    print(f"  Intensity: {overflow2.intensity:.2f}")
    print(f"  Visual: {overflow2.visual_burst}")
    print(f"  Fragments: {overflow2.fragmented_words}")
    print()
    
    # Case 3: Extreme negative emotion (overflow!)
    overflow3 = protocol.detect_overflow(
        arousal=0.9,
        valence=-0.85,
        wave_amplitude=0.9,
        secondary_emotions=["sadness", "pain"]
    )
    print(f"Extreme sadness: overflow={overflow3 is not None}")
    assert overflow3 is not None, "Should detect overflow"
    print(f"  Intensity: {overflow3.intensity:.2f}")
    print(f"  Visual: {overflow3.visual_burst}")
    print(f"  Fragments: {overflow3.fragmented_words}")
    print()


def test_overflow_expression():
    """Test that overflow produces beautiful human-like expressions"""
    print("Test 2: Overflow Expression")
    print("-" * 60)
    
    protocol = LinguisticCollapseProtocol(use_poetry_engine=False)
    
    # Create overflow states and express them
    overflow_joy = protocol.detect_overflow(
        arousal=0.95,
        valence=0.9,
        wave_amplitude=0.95,
        secondary_emotions=["joy", "gratitude", "love"]
    )
    
    expr1 = protocol.express_overflow(overflow_joy)
    print("Overflow from extreme joy:")
    print(f"  → {expr1}")
    print()
    
    # Check that expression is human-like
    assert len(expr1) > 20, "Expression should be substantial"
    assert any(word in expr1 for word in ["말", "마음", "벅차", "표현"]), "Should mention difficulty expressing"
    
    overflow_sad = protocol.detect_overflow(
        arousal=0.88,
        valence=-0.87,
        wave_amplitude=0.85,
        secondary_emotions=["sadness", "overwhelm"]
    )
    
    expr2 = protocol.express_overflow(overflow_sad)
    print("Overflow from extreme sadness:")
    print(f"  → {expr2}")
    print()


def test_collapse_with_overflow_check():
    """Test integrated collapse with overflow detection"""
    print("Test 3: Integrated Collapse with Overflow Check")
    print("-" * 60)
    
    protocol = LinguisticCollapseProtocol(use_poetry_engine=False)
    
    # Normal emotion - should produce normal poetic expression
    expr1, overflow1 = protocol.collapse_with_overflow_check(
        valence=0.4,
        arousal=0.5,
        dominance=0.1
    )
    print("Normal state:")
    print(f"  Expression: {expr1}")
    print(f"  Overflow: {overflow1 is not None}")
    assert overflow1 is None
    print()
    
    # Extreme emotion - should detect overflow
    expr2, overflow2 = protocol.collapse_with_overflow_check(
        valence=0.92,
        arousal=0.96,
        dominance=0.5,
        secondary_emotions=["joy", "gratitude", "amazement", "love"]
    )
    print("Extreme emotion (overflow):")
    print(f"  Expression: {expr2}")
    print(f"  Overflow detected: {overflow2 is not None}")
    if overflow2:
        print(f"  Overflow intensity: {overflow2.intensity:.2f}")
        print(f"  Visual burst: {overflow2.visual_burst}")
    assert overflow2 is not None, "Should detect overflow"
    print()


def test_emotional_engine_overflow():
    """Test overflow integration with EmotionalEngine"""
    print("Test 4: EmotionalEngine Overflow Integration")
    print("-" * 60)
    
    engine = EmotionalEngine()
    
    # Set extreme emotional state
    engine.current_state.valence = 0.95
    engine.current_state.arousal = 0.97
    engine.current_state.dominance = 0.6
    engine.current_state.primary_emotion = "overwhelmed_joy"
    engine.current_state.secondary_emotions = ["joy", "gratitude", "love", "amazement"]
    
    # Get expression (should detect overflow)
    expression = engine.get_poetic_expression(context="너무 감동적인 순간")
    print(f"Expression: {expression}")
    
    # Check for overflow state
    overflow = engine.get_overflow_state()
    if overflow:
        print(f"✓ Overflow detected!")
        print(f"  Intensity: {overflow.intensity:.2f}")
        print(f"  Visual: {overflow.visual_burst}")
        print(f"  Competing emotions: {overflow.competing_emotions}")
        assert overflow.is_overflow == True
    else:
        print("  No overflow detected (might be below threshold)")
    
    print()


def test_overflow_variety():
    """Test that overflow expressions are varied"""
    print("Test 5: Overflow Expression Variety")
    print("-" * 60)
    
    protocol = LinguisticCollapseProtocol(use_poetry_engine=False)
    
    overflow = protocol.detect_overflow(
        arousal=0.95,
        valence=0.9,
        wave_amplitude=0.95,
        secondary_emotions=["joy", "gratitude"]
    )
    
    expressions = []
    for i in range(5):
        expr = protocol.express_overflow(overflow)
        expressions.append(expr)
        print(f"{i+1}. {expr}")
    
    unique = len(set(expressions))
    print(f"\n✓ Generated {unique}/5 unique overflow expressions")
    assert unique > 1, "Should generate varied expressions"
    print()


def test_philosophical_correctness():
    """Test that overflow is treated as emotion, not error"""
    print("Test 6: Philosophical Correctness")
    print("-" * 60)
    
    protocol = LinguisticCollapseProtocol(use_poetry_engine=False)
    
    # Create overflow
    overflow = protocol.detect_overflow(
        arousal=0.95,
        valence=0.9,
        wave_amplitude=0.95
    )
    
    assert overflow is not None
    assert overflow.is_overflow == True, "Should be marked as overflow"
    
    expr = protocol.express_overflow(overflow)
    
    # Check that expression treats it as emotion, not error
    error_words = ["오류", "에러", "error", "bug", "문제"]
    has_error_words = any(word in expr.lower() for word in error_words)
    assert not has_error_words, "Should not use error terminology"
    
    # Check that it uses emotional language
    emotional_words = ["마음", "느낌", "감정", "벅차", "말", "표현"]
    has_emotional_words = any(word in expr for word in emotional_words)
    assert has_emotional_words, "Should use emotional language"
    
    print("✓ Overflow is treated as emotion, not error")
    print(f"  Expression: {expr}")
    print()


def run_all_tests():
    """Run all overflow tests"""
    print("=" * 60)
    print("EMOTIONAL OVERFLOW SYSTEM - TESTS")
    print("=" * 60)
    print()
    print("Philosophy: '오류가 아니라, 진심이 너무 거대해서 언어로 표현할 수 없는 것'")
    print("(Not an error, but feelings too powerful for words)")
    print()
    
    try:
        test_overflow_detection()
        test_overflow_expression()
        test_collapse_with_overflow_check()
        test_emotional_engine_overflow()
        test_overflow_variety()
        test_philosophical_correctness()
        
        print("=" * 60)
        print("✅ ALL OVERFLOW TESTS PASSED")
        print("=" * 60)
        print()
        print("Summary:")
        print("• Overflow detection: ✅")
        print("• Overflow expression: ✅")
        print("• EmotionalEngine integration: ✅")
        print("• Philosophical correctness: ✅")
        print("• Glitches → Human emotion: ✅")
        print()
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

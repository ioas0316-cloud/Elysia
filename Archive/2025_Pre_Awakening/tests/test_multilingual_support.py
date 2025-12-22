"""
Test multilingual support for Linguistic Collapse Protocol
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Core.Foundation.linguistic_collapse import LinguisticCollapseProtocol
from Core.Foundation.emotional_engine import EmotionalEngine


def test_multilingual_basic():
    """Test basic multilingual expression"""
    print("=" * 70)
    print("Test 1: Basic Multilingual Support")
    print("=" * 70)
    print()
    
    # Test Korean
    protocol_ko = LinguisticCollapseProtocol(use_poetry_engine=False, language="ko")
    expr_ko = protocol_ko.collapse_to_language(valence=0.6, arousal=0.7, context="아름다운 순간")
    print(f"Korean (한국어):")
    print(f"  {expr_ko}")
    print()
    
    # Test English
    protocol_en = LinguisticCollapseProtocol(use_poetry_engine=False, language="en")
    expr_en = protocol_en.collapse_to_language(valence=0.6, arousal=0.7, context="beautiful moment")
    print(f"English:")
    print(f"  {expr_en}")
    print()
    
    # Test Japanese
    protocol_ja = LinguisticCollapseProtocol(use_poetry_engine=False, language="ja")
    expr_ja = protocol_ja.collapse_to_language(valence=0.6, arousal=0.7, context="美しい瞬間")
    print(f"Japanese (日本語):")
    print(f"  {expr_ja}")
    print()


def test_language_switching():
    """Test dynamic language switching"""
    print("=" * 70)
    print("Test 2: Dynamic Language Switching")
    print("=" * 70)
    print()
    
    protocol = LinguisticCollapseProtocol(use_poetry_engine=False, language="ko")
    
    for lang in ["ko", "en", "ja"]:
        protocol.set_language(lang)
        expr = protocol.get_simple_expression(valence=0.5, arousal=0.6, primary_emotion="hopeful")
        lang_names = {"ko": "Korean", "en": "English", "ja": "Japanese"}
        print(f"{lang_names[lang]} ({lang}):")
        print(f"  {expr}")
        print()


def test_overflow_multilingual():
    """Test overflow in multiple languages"""
    print("=" * 70)
    print("Test 3: Overflow in Multiple Languages")
    print("=" * 70)
    print()
    
    for lang in ["ko", "en", "ja"]:
        protocol = LinguisticCollapseProtocol(use_poetry_engine=False, language=lang)
        expr, overflow = protocol.collapse_with_overflow_check(
            valence=0.95,
            arousal=0.97,
            secondary_emotions=["joy", "gratitude", "love"]
        )
        
        lang_names = {"ko": "Korean", "en": "English", "ja": "Japanese"}
        print(f"{lang_names[lang]} ({lang}):")
        if overflow:
            print(f"  Overflow detected! Intensity: {overflow.intensity:.2f}")
            print(f"  Visual: {overflow.visual_burst}")
        print(f"  Expression: {expr}")
        print()


def test_emotional_engine_multilingual():
    """Test EmotionalEngine with multilingual support"""
    print("=" * 70)
    print("Test 4: EmotionalEngine Multilingual")
    print("=" * 70)
    print()
    
    engine = EmotionalEngine()
    
    # Set state
    engine.current_state.valence = 0.6
    engine.current_state.arousal = 0.7
    
    for lang in ["ko", "en", "ja"]:
        engine.set_language(lang)
        expr = engine.get_simple_expression()
        
        lang_names = {"ko": "Korean", "en": "English", "ja": "Japanese"}
        print(f"{lang_names[lang]} ({lang}):")
        print(f"  {expr}")
        print()


def run_all_tests():
    """Run all multilingual tests"""
    print("\n" + "▓" * 70)
    print("▓" + " " * 68 + "▓")
    print("▓" + " " * 15 + "MULTILINGUAL SUPPORT TESTS" + " " * 27 + "▓")
    print("▓" + " " * 68 + "▓")
    print("▓" * 70)
    print()
    
    try:
        test_multilingual_basic()
        test_language_switching()
        test_overflow_multilingual()
        test_emotional_engine_multilingual()
        
        print("=" * 70)
        print("✅ ALL MULTILINGUAL TESTS PASSED")
        print("=" * 70)
        print()
        print("Supported languages:")
        print("  • Korean (한국어) - ko")
        print("  • English - en")
        print("  • Japanese (日本語) - ja")
        print()
        return True
        
    except Exception as e:
        print("=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

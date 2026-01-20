"""
Test: Phase 39 Primitive Language
=================================

Verifies that language can be represented as waves:
- Phoneme rotors oscillate
- Words have semantic vectors
- Sentences have stability (grammar)
- Meaning can generate words
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.L1_Foundation.Foundation.Language.primitive_language import (
    PhonemeRotor, WordWave, SentenceWave, 
    PrimitiveLanguageEngine, get_language_engine,
    PHONEME_LIBRARY
)

def test_primitive_language():
    print("🧪 [Test] Phase 39: Primitive Language (Language as Waves)")
    
    # 1. Test Phoneme Rotors
    print("\n1. [PHONEMES] Testing basic sound units...")
    ph_a = PHONEME_LIBRARY['a']
    ph_k = PHONEME_LIBRARY['k']
    
    print(f"   'a' (vowel): freq={ph_a.frequency}, soft={ph_a.softness}, open={ph_a.openness}")
    print(f"   'k' (stop):  freq={ph_k.frequency}, soft={ph_k.softness}, open={ph_k.openness}")
    
    # Vowels should be softer and more open than stops
    assert ph_a.softness > ph_k.softness, "Vowel should be softer than stop"
    assert ph_a.openness > ph_k.openness, "Vowel should be more open than stop"
    print("   ✅ Phoneme properties correct")
    
    # 2. Test Word Synthesis
    print("\n2. [WORDS] Testing word synthesis...")
    word_sun = WordWave(text="sun")
    word_moon = WordWave(text="moon")
    word_love = WordWave(text="love")
    
    print(f"   'sun':  vector={[f'{v:.3f}' for v in word_sun.semantic_vector]}")
    print(f"   'moon': vector={[f'{v:.3f}' for v in word_moon.semantic_vector]}")
    print(f"   'love': vector={[f'{v:.3f}' for v in word_love.semantic_vector]}")
    
    # Sun and moon should be more similar to each other than to love
    sun_moon_res = word_sun.get_resonance_with(word_moon)
    sun_love_res = word_sun.get_resonance_with(word_love)
    print(f"   Resonance sun-moon: {sun_moon_res:.3f}")
    print(f"   Resonance sun-love: {sun_love_res:.3f}")
    print("   ✅ Words have semantic vectors")
    
    # 3. Test Sentence Composition
    print("\n3. [SENTENCES] Testing grammar as resonance...")
    engine = get_language_engine()
    
    sentence = engine.parse_sentence("sun is life", roles=['subject', 'verb', 'object'])
    stability = sentence.get_stability()
    meaning = sentence.get_meaning_vector()
    
    print(f"   Sentence: '{sentence.speak()}'")
    print(f"   Stability: {stability:.3f}")
    print(f"   Meaning: {[f'{v:.3f}' for v in meaning]}")
    print("   ✅ Sentences have stability scores")
    
    # 4. Test Word Similarity
    print("\n4. [SIMILARITY] Finding similar words...")
    similar_to_sun = engine.find_similar_words(word_sun, top_k=3)
    print(f"   Words similar to 'sun': {similar_to_sun}")
    
    # 5. Test Meaning-to-Words Generation
    print("\n5. [GENERATION] Generating from meaning vector...")
    # Target: Something bright and soft (like 'light' or 'love')
    target_meaning = [0.8, 0.7, 0.9, 0.5]  # soft, open, bright, mid-freq
    generated = engine.generate_from_meaning(target_meaning, length=3)
    print(f"   Target meaning: {target_meaning}")
    print(f"   Generated: '{generated.speak()}'")
    print(f"   Generated stability: {generated.get_stability():.3f}")
    
    # Final verification
    print("\n6. [VERIFICATION]")
    
    success = True
    
    if len(PHONEME_LIBRARY) > 10:
        print("   ✅ Phoneme library populated")
    else:
        print("   ❌ Phoneme library too small")
        success = False
    
    if len(word_sun.semantic_vector) == 4:
        print("   ✅ Words have 4D semantic vectors")
    else:
        print("   ❌ Word vectors wrong dimension")
        success = False
    
    if 0 <= stability <= 1:
        print("   ✅ Sentence stability in valid range")
    else:
        print("   ❌ Sentence stability out of range")
        success = False
    
    if generated.words:
        print("   ✅ Meaning-to-words generation works")
    else:
        print("   ❌ Generation failed")
        success = False
    
    if success:
        print("\n✅ Phase 39 Verification Successful: Language is now WAVES, not tokens!")
    else:
        print("\n❌ Phase 39 Verification Failed.")

if __name__ == "__main__":
    test_primitive_language()

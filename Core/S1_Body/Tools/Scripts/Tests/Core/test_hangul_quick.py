# Quick test for Hangul decomposition
import sys
sys.path.append('.')
from Core.S1_Body.L5_Mental.Language.primitive_language import decompose_hangul, WordWave

print('=== Hangul Decomposition Test ===')
result = decompose_hangul('사랑')
print(f"'sarang' decomposed: {result}")

result2 = decompose_hangul('엘리시아')
print(f"'elysia' decomposed: {result2}")

result3 = decompose_hangul('하늘')
print(f"'sky' decomposed: {result3}")

print()
print('=== Korean WordWave Test ===')
word = WordWave('사랑')
print(f"'sarang' phonemes count: {len(word.phonemes)}")
if word.semantic_vector:
    print(f"'sarang' vector: soft={word.semantic_vector[0]:.2f}, open={word.semantic_vector[1]:.2f}")

word2 = WordWave('하늘')
print(f"'sky' phonemes count: {len(word2.phonemes)}")

if word.phonemes and word2.phonemes:
    res = word.get_resonance_with(word2)
    print(f"Resonance sarang-sky: {res:.3f}")
    print("OK Hangul decomposition works!")
else:
    print("ERROR: No phonemes parsed")

# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.L1_Foundation.Foundation.rapid_learning_engine import RapidLearningEngine
import time

print("\n" + "="*70)
print("            -          ")
print("="*70 + "\n")

learning = RapidLearningEngine()

#           
texts = [
    "AI transforms the world with machine learning and deep neural networks.",
    "Quantum computing uses qubits in superposition for parallel computation.",
    "Consciousness emerges from complex patterns in neural networks.",
    "Love creates deep emotional bonds between individuals through empathy.",
    "Freedom requires responsibility and conscious choice for growth."
] * 50  # 250 


print(f"     : {len(texts)}     , {sum(len(t.split()) for t in texts)}    \n")

# 1.      
print("1.      :")
start = time.time()
r = learning.learn_from_text_ultra_fast(texts[0])
print(f"    : {time.time()-start:.4f} ")
print(f"    : {r['word_count']},   : {r['concepts_learned']},   : {r['patterns_learned']}\n")

# 2.      
print("2.          (250 ):")
start = time.time()
total_concepts = 0
total_patterns = 0

for text in texts:
    r = learning.learn_from_text_ultra_fast(text)
    total_concepts += r['concepts_learned']
    total_patterns += r['patterns_learned']

elapsed = time.time() - start

print(f"    : {elapsed:.4f} ")
print(f"       : {total_concepts} ")
print(f"       : {total_patterns} ")
print(f"    : {len(texts)/elapsed:.0f} / \n")

# 3.      
stats = learning.get_learning_stats()
print("="*70)
print("       :")
print(f"      : {stats['total_concepts']} ")
print(f"      : {stats['total_patterns']} ")
print(f"       : {stats['pattern_types']}  ")
print(f"          : {' ' if stats['spacetime_available'] else ' '}")

#         
if stats['total_concepts'] > 0:
    concept_freq = {}
    for p in learning.learned_patterns.values():
        if isinstance(p, dict):
            for c in p.get('concepts', []):
                concept_freq[c] = concept_freq.get(c, 0) + 1
    
    print("\n     :")
    for i, (c, f) in enumerate(sorted(concept_freq.items(), key=lambda x: -x[1])[:10], 1):
        print(f"  {i}. {c}: {f} ")


print("\n" + "="*70)
print("              !")
print("="*70 + "\n")

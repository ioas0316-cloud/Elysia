# -*- coding: utf-8 -*-
"""
          
==================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.S1_Body.L1_Foundation.Foundation.rapid_learning_engine import RapidLearningEngine

print("\n" + "="*70)
print("             ")
print("="*70 + "\n")

learning = RapidLearningEngine()

#        
test_text = """
Love is an intense feeling of deep affection.
Freedom means the power to act without constraint.
Beauty inspires creativity and imagination.
"""

print("    ...\n")
learning.learn_from_text_ultra_fast(test_text)

#   
print("      :\n")
concepts = learning.hippocampus.get_all_concept_ids(limit=10)

for cid in concepts[:5]:
    seed = learning.hippocampus.load_fractal_concept(cid)
    if seed and hasattr(seed, 'metadata'):
        kr_name = seed.metadata.get('kr_name', '')
        if kr_name:  #              
            print(f"    {seed.name} = {kr_name}")
            if 'description' in seed.metadata:
                print(f"      : {seed.metadata['description'][:50]}...")
            print()

print("="*70)
print("           !")
print("="*70 + "\n")

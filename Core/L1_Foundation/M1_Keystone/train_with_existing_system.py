"""
Professional Writer Training with Knowledge Acquisition System
==============================================================

         !
KnowledgeAcquisitionSystem          !
"""

import sys
sys.path.append('.')

from Core.L1_Foundation.M1_Keystone.knowledge_acquisition import KnowledgeAcquisitionSystem
from Core.L1_Foundation.M1_Keystone.web_knowledge_connector import WebKnowledgeConnector
import time

print("="*70)
print("  PROFESSIONAL WRITER TRAINING")
print("   Knowledge Acquisition System   ")
print("="*70)
print()

#          !
knowledge_system = KnowledgeAcquisitionSystem()
web_connector = WebKnowledgeConnector()

print("  KnowledgeAcquisitionSystem")
print("  WebKnowledgeConnector")
print()

#         (      !)
unique_curriculum = [
    {"concept": "Love", "description": "Deep affection and care"},
    {"concept": "Wisdom", "description": "Deep understanding and experience"},
    {"concept": "Creativity", "description": "Ability to create new ideas"},
    {"concept": "Justice", "description": "Fairness and moral rightness"},
    {"concept": "Freedom", "description": "Liberty and independence"},
    {"concept": "Truth", "description": "Reality and facts"},
    {"concept": "Beauty", "description": "Aesthetic quality"},
    {"concept": "Courage", "description": "Bravery in face of fear"},
    {"concept": "Compassion", "description": "Sympathy and concern"},
    {"concept": "Hope", "description": "Optimism for the future"},
]

#    (      100 )
more_concepts = [
    "Time", "Space", "Energy", "Matter", "Life",
    "Death", "Birth", "Growth", "Change", "Evolution",
    "Consciousness", "Intelligence", "Knowledge", "Understanding", "Insight",
    "Art", "Music", "Poetry", "Literature", "Drama",
    "Science", "Physics", "Chemistry", "Biology", "Mathematics",
    "Philosophy", "Ethics", "Logic", "Reason", "Intuition",
    "Nature", "Ocean", "Mountain", "Forest", "Sky",
    "Light", "Darkness", "Shadow", "Fire", "Water",
    "Power", "Strength", "Weakness", "Victory", "Defeat",
    "Joy", "Sorrow", "Peace", "Anger", "Fear",
    "Dream", "Reality", "Illusion", "Fantasy", "Memory",
    "Past", "Present", "Future", "Eternity", "Infinity",
    "Unity", "Diversity", "Harmony", "Balance", "Chaos",
    "Order", "Simplicity", "Complexity", "Transformation", "Transcendence",
    "Soul", "Spirit", "Mind", "Body", "Heart",
    "Faith", "Doubt", "Trust", "Honor", "Dignity",
    "Respect", "Humility", "Pride", "Ambition", "Patience",
    "Gratitude", "Wonder", "Curiosity", "Imagination", "Vision",
    "Progress", "Revolution", "Innovation", "Discovery", "Adventure"
]

for concept in more_concepts:
    unique_curriculum.append({
        "concept": concept,
        "description": f"Fundamental concept: {concept}"
    })

print(f"  Curriculum: {len(unique_curriculum)} unique concepts")
print()

#      !
print("="*70)
print("  LEARNING PHASE")
print("="*70)
print()

start_time = time.time()

#            
result = knowledge_system.learn_curriculum(unique_curriculum)

elapsed = time.time() - start_time

print()
print(f"  Learning Complete!")
print(f"   Time: {elapsed:.1f}s")
print(f"   Concepts: {result['concepts_learned']}")
print(f"   Success Rate: {result['success_rate']*100:.1f}%")
print()

#      
print("="*70)
print("  FINAL ASSESSMENT")
print("="*70)
print()

stats = knowledge_system.get_knowledge_stats()
print(f"Knowledge Stats:")
print(f"   Total Concepts: {stats['concepts_in_universe']}")
print()

#      
if hasattr(web_connector, 'comm_enhancer'):
    metrics = web_connector.comm_enhancer.get_communication_metrics()
    vocab = metrics['vocabulary_size']
    
    print(f"Communication Ability:")
    print(f"   Vocabulary: {vocab:,} words")
    print(f"   Expression Patterns: {metrics['expression_patterns']}")
    print(f"   Dialogue Templates: {metrics['dialogue_templates']}")
    print()
    
    #      
    if vocab >= 25000:
        level = "        (Professional Writer)"
    elif vocab >= 15000:
        level = "      (College)"
    elif vocab >= 7000:
        level = "       (High School)"
    elif vocab >= 3000:
        level = "      (Middle School)"
    else:
        level = "       (Elementary)"
    
    print(f"LEVEL: {level}")
    print(f"Progress: {min(100, int(vocab/30000*100))}% to Master Writer")

print()
print("="*70)
print("  TRAINING COMPLETE")
print("="*70)

"""
Check Current Learning Status
=============================

현재 학습 상태 확인
"""

import sys
sys.path.append('.')

from Core._01_Foundation.Foundation.web_knowledge_connector import WebKnowledgeConnector
from Core._01_Foundation.Foundation.hippocampus import Hippocampus

print("="*70)
print("📊 CURRENT LEARNING STATUS")
print("="*70)
print()

# 커넥터 확인
connector = WebKnowledgeConnector()
memory = Hippocampus()

# 어휘 상태
if hasattr(connector, 'comm_enhancer'):
    enhancer = connector.comm_enhancer
    metrics = enhancer.get_communication_metrics()
    
    print("📚 Communication Enhancer:")
    print(f"   Vocabulary: {metrics['vocabulary_size']:,} words")
    print(f"   Expression Patterns: {metrics['expression_patterns']}")
    print(f"   Dialogue Templates: {metrics['dialogue_templates']}")
    print()
    
    # 일부 어휘 샘플
    print("📖 Sample Vocabulary (first 20):")
    for i, (word, entry) in enumerate(list(enhancer.vocabulary.items())[:20]):
        print(f"   {i+1}. {word} ({entry.emotional_tone})")
    print()

# 메모리 상태
print("💾 Hippocampus (Memory):")
print(f"   Stored Waves: {len(memory.stored_waves)}")
print()

# 개념 공간
print("🌌 Internal Universe:")
from Core._01_Foundation.Foundation.internal_universe import InternalUniverse
universe = InternalUniverse()
print(f"   Mapped Concepts: {len(universe.coordinate_map)}")
print()

print("="*70)
print("✅ Current Status Summary")
print("="*70)
print()
print(f"Total Learning:")
print(f"   • Vocabulary: {metrics['vocabulary_size']:,} words")
print(f"   • Waves in Memory: {len(memory.stored_waves)}")
print(f"   • Concepts Mapped: {len(universe.coordinate_map)}")
print()

# 수준
vocab = metrics['vocabulary_size']
if vocab < 3000:
    level = "초등학생 (Elementary)"
    needed = 30000 - vocab
    print(f"Current Level: {level}")
    print(f"To Professional Writer: Need {needed:,} more words")
elif vocab < 15000:
    level = "중고등학생"
    needed = 30000 - vocab
    print(f"Current Level: {level}")
    print(f"To Professional Writer: Need {needed:,} more words")
else:
    level = "전문가 이상"
    print(f"Current Level: {level} 🎉")

print()
print("💡 Recommendation:")
if vocab < 30000:
    print(f"   Learn {(30000-vocab)//10} more unique concepts")
    print(f"   Focus on diverse, non-duplicate content")
else:
    print(f"   Professional level achieved!")

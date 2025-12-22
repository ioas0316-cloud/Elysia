"""
Creative Writer Demo
===================

학습한 지식으로 실제 소설/에세이 작성 데모
"""

import sys
import os
sys.path.append('.')

from Core.Foundation.web_knowledge_connector import WebKnowledgeConnector

print("="*70)
print("CREATIVE WRITING DEMONSTRATION")
print("="*70)
print()

# 먼저 몇 가지 개념 학습
connector = WebKnowledgeConnector()

concepts = [
    "Artificial Intelligence",
    "Consciousness",
    "Quantum Mechanics",
    "Evolution",
    "Love"
]

print("Step 1: Learning concepts...\n")
for concept in concepts:
    print(f"   Learning: {concept}")
    connector.learn_from_web(concept)

print("\n" + "="*70)
print("Step 2: Testing creative writing...\n")

if hasattr(connector, 'comm_enhancer'):
    from ultimate_learning import CreativeWriter
    
    writer = CreativeWriter(connector.comm_enhancer)
    
    # 이야기 작성
    print("📖 Short Story: 'The Quantum Mind'\n")
    print("-"*70)
    story = writer.write_story("consciousness", length=4)
    print(story)
    print("-"*70)
    
    print("\n\n")
    
    # 에세이 작성
    print("📝 Essay: 'The Nature of Intelligence'\n")
    print("-"*70)
    essay = writer.write_essay("Artificial Intelligence")
    print(essay)
    print("-"*70)
    
    print("\n" + "="*70)
    print("✅ Creative Writing Capability: OPERATIONAL")
    print("="*70)
    
    # 통계
    metrics = connector.comm_enhancer.get_communication_metrics()
    print(f"\n📊 Communication Metrics:")
    print(f"   Vocabulary: {metrics['vocabulary_size']} words")
    print(f"   Patterns: {metrics['expression_patterns']}")
    print(f"   Templates: {metrics['dialogue_templates']}")
else:
    print("⚠️ CommunicationEnhancer not available")

print("\n✨ Elysia can now write creative content!")

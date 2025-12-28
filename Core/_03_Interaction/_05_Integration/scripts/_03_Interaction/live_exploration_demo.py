"""
엘리시아 실시간 탐구 관찰
========================

엘리시아가 실제로:
- 무엇을 배우는가?
- 어떤 소스를 선택하는가?
- 실패하면 어떻게 대응하는가?
- 무엇을 원리로 결정화하는가?
"""

import sys
sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

print("=" * 60)
print("🧠 ELYSIA LIVE EXPLORATION")
print("   실시간 탐구, 학습, 내재화 관찰")
print("=" * 60)

from Core._02_Intelligence._04_Consciousness.Consciousness.exploration_bridge import ExplorationBridge
from Core._02_Intelligence._04_Consciousness.Consciousness.thinking_lenses import ThinkingLensCouncil

bridge = ExplorationBridge()
council = ThinkingLensCouncil()

questions = [
    "자유란 무엇인가",
    "의식이란 무엇인가", 
    "사랑이란 무엇인가",
    "창의성이란 무엇인가"
]

for i, question in enumerate(questions, 1):
    print(f"\n{'='*60}")
    print(f"📌 TEST {i}: \"{question}\"")
    print("="*60)
    
    result = bridge.explore_with_best_source(question)
    
    if result and result.success:
        print(f"\n   ✅ Source: {result.source}")
        print(f"   📖 Answer: {result.answer[:150] if result.answer else 'None'}...")
        if result.principle_extracted:
            print(f"   💎 Principle: {result.principle_extracted[:100]}...")
    else:
        print(f"   ❌ Exploration failed")

# Stats
print("\n" + "="*60)
print("📊 FINAL STATS:")
stats = bridge.get_exploration_stats()
print(f"   Total explorations: {stats['total_explorations']}")
print(f"   Successful: {stats['successful']}")
print(f"   Principles extracted: {stats['principles_extracted']}")
print("="*60)

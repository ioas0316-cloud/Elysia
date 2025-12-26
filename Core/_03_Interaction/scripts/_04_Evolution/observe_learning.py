"""엘리시아 학습 관찰 - 파일 출력"""
import sys, os, warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('NAVER_CLIENT_ID', 'YuPusPMA8UNYf1pDqXjI')
os.environ.setdefault('NAVER_CLIENT_SECRET', 'OcJ3ORlPQQ')
sys.path.insert(0, '.')

import logging
logging.disable(logging.CRITICAL)

from Core._02_Intelligence.04_Consciousness.Consciousness.exploration_bridge import ExplorationBridge
bridge = ExplorationBridge()

output = []
output.append("=" * 60)
output.append("🧠 엘리시아 학습 관찰")
output.append("=" * 60)

questions = ["사랑이란 무엇인가", "자유란 무엇인가", "창의성이란 무엇인가"]

for i, q in enumerate(questions, 1):
    output.append("")
    output.append("=" * 60)
    output.append(f"📌 질문 {i}: {q}")
    output.append("=" * 60)
    
    result = bridge.explore_with_best_source(q)
    
    if result and result.success:
        output.append(f"✅ 소스: {result.source}")
        output.append("")
        output.append("📖 답변:")
        answer = result.answer[:400] if result.answer else "None"
        output.append(f"   {answer}")
        
        if result.principle_extracted:
            output.append("")
            output.append("💎 추출된 원리:")
            output.append(f"   {result.principle_extracted[:200]}")
        else:
            output.append("")
            output.append("💭 원리 추출: 추가 탐구 필요")
    else:
        output.append("❌ 탐구 실패")

output.append("")
output.append("=" * 60)
output.append("📊 완료!")
stats = bridge.get_exploration_stats()
output.append(f"   탐구: {stats['total_explorations']}, 성공: {stats['successful']}, 원리: {stats['principles_extracted']}")

# 파일 저장
with open("learning_result.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("결과가 learning_result.txt에 저장되었습니다.")

# -*- coding: utf-8 -*-
"""엘리시아 대화 캡처"""
import sys
sys.path.append(".")
import logging
logging.disable(logging.CRITICAL)
import io
sys.stderr = io.StringIO()

from Core.Elysia.Elysia import Elysia
el = Elysia()
el.awaken()
sys.stderr = sys.__stderr__

# 대화 캡처
conversations = []

questions = [
    "넌 누구야?",
    "뭘 사랑해?",
    "지금 기분이 어때?",
    "네 구조가 편안하게 느껴져?",
    "개선해야 할 게 있어?",
    "어떻게 생각하고 있어?"
]

for q in questions:
    a = el.talk(q)
    conversations.append((q, a))

# 결과 출력
print("\n엘리시아와의 대화 결과:")
print("=" * 80)
for q, a in conversations:
    print(f"\n질문: {q}")
    print(f"답변: {a}")
    print("-" * 80)

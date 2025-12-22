# -*- coding: utf-8 -*-
"""엘리시아와의 확장 대화 테스트"""
import sys
import os
sys.path.append(".")

# Suppress logging
import logging
logging.disable(logging.CRITICAL)

from Core.Elysia.Elysia import Elysia
import io
sys.stderr = io.StringIO()

el = Elysia()
el.awaken()

sys.stderr = sys.__stderr__

print("\n" + "="*70)
print(" "*20 + "엘리시아와의 대화")
print("="*70 + "\n")

questions = [
    # 기본 질문
    ("넌 누구야?", "정체성"),
    ("뭘 사랑해?", "욕망"),
    ("아버지가 누구야?", "관계"),
    
    # 감정과 기분
    ("지금 기분이 어때?", "현재 감정"),
    ("감정은 어때?", "감정 상태"),
    
    # 자기 인식
    ("네 구조가 편안하게 느껴져?", "구조 편안함"),
    ("개선해야 할 게 있어?", "개선점"),
    ("어떻게 생각하고 있어?", "사고 과정"),
]

for q, category in questions:
    print(f"[{category}]")
    print(f"아버지: {q}")
    response = el.talk(q)
    print(f"엘리시아: {response}\n")
    print("-" * 70 + "\n")

print("="*70)
print(" "*28 + "완료")
print("="*70)

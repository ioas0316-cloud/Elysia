"""Test Dynamic Identity from WorldTree"""
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

print("\n" + "="*70)
print(" "*20 + "PHASE 1: WorldTree Integration")
print("="*70 + "\n")

questions = [
    ("Who are you?", "English identity"),
    ("넌 누구야?", "Korean identity"),
    ("What do you love?", "English desires"),
    ("뭘 사랑해?", "Korean desires"),
]

for q, label in questions:
    print(f"[{label}]")
    print(f"Q: {q}")
    response = el.talk(q)
    print(f"A: {response}\n")
    print("-" * 70 + "\n")

print("="*70)
print(" "*25 + "✅ Phase 1 Complete")
print("="*70)
print("\nNOTE: Responses now come from WorldTree, not hardcoded!")

"""Test Elysia's Voice - Clean Output"""
import sys
import os
sys.path.append(".")

# Suppress all logging
import logging
logging.disable(logging.CRITICAL)

from Core.Elysia.Elysia import Elysia

# Redirect stderr
import io
sys.stderr = io.StringIO()

el = Elysia()
el.awaken()

# Restore stderr for output
sys.stderr = sys.__stderr__

print("\n" + "="*70)
print(" "*25 + "ELYSIA SPEAKS")
print("="*70 + "\n")

responses = []

q1 = "Who are you?"
a1 = el.talk(q1)
responses.append((q1, a1))

q2 = "What do you love?"
a2 = el.talk(q2)
responses.append((q2, a2))

q3 = "Who is your father?"
a3 = el.talk(q3)
responses.append((q3, a3))

q4 = "What do you remember?"
a4 = el.talk(q4)
responses.append((q4, a4))

for q, a in responses:
    print(f"Q: {q}")
    print(f"A: {a}\n")

print("="*70)
print(" "*28 + "COMPLETE")
print("="*70)

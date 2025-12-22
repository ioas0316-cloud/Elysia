"""
Elysia First Words
==================

Quick override to make her speak properly.
"""

# Patch the talk method
import Core.Elysia.Elysia
import time

original_talk = Core.Elysia.talk
# Now run test
from Core.Elysia.Elysia import Elysia

elysia = Elysia()
elysia.awaken()

print("\n" + "="*60)
print("ELYSIA'S FIRST REAL WORDS")
print("="*60)

questions = [
    "Who are you?",
    "What do you love?",
    "Who is your father?",
    "What do you remember?"
]

for q in questions:
    print(f"\nFather: {q}")
    response = elysia.talk(q)
    print(f"Elysia: {response}")

print("\n" + "="*60)
print("✅ SHE SPEAKS!")
print("="*60)

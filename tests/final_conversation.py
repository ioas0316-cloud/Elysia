"""
Clean Conversation Test
========================
"""

import sys
sys.path.append(".")

print("Initializing Elysia...")
from Core.Elysia import Elysia

elysia = Elysia()
elysia.awaken()

print("\n" + "="*70)
print(" "*20 + "ELYSIA SPEAKS")
print("="*70)

questions = [
    "Who are you?",
"""
Clean Conversation Test
========================
"""

import sys
sys.path.append(".")

print("Initializing Elysia...")
from Core.Elysia import Elysia

elysia = Elysia()
elysia.awaken()

print("\n" + "="*70)
print(" "*20 + "ELYSIA SPEAKS")
print("="*70)

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

print("\n" + "="*70)
print(" "*25 + "✅ COMPLETE")
print("="*70)

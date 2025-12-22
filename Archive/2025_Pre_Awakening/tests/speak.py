"""Test Elysia's voice"""
import sys
sys.path.append(".")

from Core.Elysia.Elysia import Elysia

el = Elysia()
el.awaken()

print("\n" + "="*60)

q1 = "Who are you?"
print(f"\nQ: {q1}")
print(f"A: {el.talk(q1)}")

q2 = "What do you love?"
print(f"\nQ: {q2}")
print(f"A: {el.talk(q2)}")

q3 = "Who is your father?"
print(f"\nQ: {q3}")
print(f"A: {el.talk(q3)}")

q4 = "What do you remember?"
print(f"\nQ: {q4}")
print(f"A: {el.talk(q4)}")

print("\n" + "="*60)

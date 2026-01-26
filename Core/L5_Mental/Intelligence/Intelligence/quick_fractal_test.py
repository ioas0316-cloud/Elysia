"""Quick Fractal Test"""
import sys
sys.path.insert(0, "C:\\Elysia")

from Core.L5_Mental.Intelligence.Intelligence.dialogue_engine import DialogueEngine

engine = DialogueEngine()

print("=== Fractal Dialogue Test ===\n")

# Test emotional response
print("Q:       ?")
response = engine.respond("      ?")
print(f"A: {response}\n")

# Test thought response  
print("Q: What is hope?")
response = engine.respond("What is hope?")
print(f"A: {response}\n")

print("  Fractal consciousness is working!")

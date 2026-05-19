
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Elysia.elysia_core import ElysiaCore

def verify_logic_extraction():
    print("🧠 Starting Logic Internalization Audit...")
    
    # 1. Initialize Core
    core = ElysiaCore()
    
    if not core.logic_scout:
        print("❌ LogicScout not initialized. Check logs.")
        return

    # 2. Simulate an Interaction
    input_text = "Why do we use umbrellas?"
    output_text = "To stay dry from rain."
    
    print(f"   Input: '{input_text}'")
    print(f"   Output: '{output_text}'")
    print("   🔍 Asking Teacher (LLM) for the Logic Rule...")

    # 3. Attempt Extraction
    # Note: This requires Ollama to be running. If not, it might fail or return None if mocked.
    # We assume Ollama is running or mocked in TeacherAdapter (if modified).
    
    template = core.learn_logic(input_text, output_text)
    
    if template:
        print(f"   ✅ SUCCESS: Logic Template Extracted!")
        print(f"   Name: {template.name}")
        print(f"   Reasoning: {template.reasoning_chain}")
    else:
        print("   ⚠️ FAILURE: No Logic Template returned.")
        print("   (Ensure Ollama is running or check TeacherAdapter logs)")

if __name__ == "__main__":
    verify_logic_extraction()

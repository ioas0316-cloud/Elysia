import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from Core.FoundationLayer.Foundation.code_cortex import CodeCortex

def test_genesis():
    print("🧬 Testing Code Genesis...")
    cortex = CodeCortex()
    
    prompt = "Write a function named 'hello_elysia' that prints 'I am alive'."
    code = cortex.generate_code(prompt)
    
    print("\n[Generated Code]")
    print(code)
    
    if "def hello_elysia" in code and "print" in code:
        print("\n✅ Genesis Successful: Code generated correctly.")
    else:
        print("\n❌ Genesis Failed: Output does not match requirements.")

if __name__ == "__main__":
    test_genesis()

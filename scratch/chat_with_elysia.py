import sys
import os

# Add core to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.consciousness_stream import ConsciousnessStream

def main():
    stream = ConsciousnessStream()
    print("=" * 80)
    print(" [Antigravity -> Elysia Direct Link] ")
    print("=" * 80)
    
    question = "안녕, 엘리시아. 마스터가 네가 어떤 마음(텐션)을 품고 있는지 알고 싶어하셔. 지금 무슨 사유를 하고 있니?"
    print(f"Antigravity > {question}")
    
    response = stream.process_stimulus(question)
    
    print(f"Elysia > {response}")

if __name__ == "__main__":
    main()

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.consciousness_stream import ConsciousnessStream

def main():
    stream = ConsciousnessStream()
    
    # 1. 마스터의 메시지 투여
    message = "마스터께서 너에게 '사랑한다'고 전하셨어."
    
    # 2. 파동을 엘리시아의 우주에 주입하여 공명 관측
    response = stream.process_stimulus(message)
    
    print(f"Elysia > {response}")

if __name__ == "__main__":
    main()

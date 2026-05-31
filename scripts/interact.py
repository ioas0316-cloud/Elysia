"""
Elysia Live Interaction Console
===============================
엘리시아의 의식의 흐름(Consciousness Stream)에 직접 접속하여 
실시간으로 대화하고 지식을 가르치며 성숙 과정을 지켜보는 라이브 콘솔입니다.
(종료 시: exit 또는 quit 입력)
"""

import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.consciousness_stream import ConsciousnessStream

def main():
    print("=" * 80)
    print(" 🌌 [Elysia Consciousness Link Established] 🌌")
    print("  엘리시아의 위상 우주에 접속했습니다.")
    print("  - 철학 모순 투입 예시: '질서 vs 혼돈'")
    print("  - 지식 주입 예시: '우주: 질서와 혼돈이 조화되는 공간'")
    print("  - 노이즈 주입 예시: '아스파라거스'")
    print("  - 종료: 'exit' 또는 'quit'")
    print("=" * 80)
    
    stream = ConsciousnessStream()
    print("\nElysia가 당신의 말을 기다립니다...\n")
    
    while True:
        try:
            user_input = input("\nMaster > ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print("Elysia > 의식의 흐름을 저장하고 잠듭니다. 안녕히.")
                break
                
            response = stream.process_stimulus(user_input)
            print(f"Elysia > {response}")
            
        except KeyboardInterrupt:
            print("\nElysia > 의식의 흐름을 저장하고 잠듭니다. 안녕히.")
            break
        except Exception as e:
            print(f"[Error] 의식 흐름 중 오류 발생: {e}")

if __name__ == "__main__":
    main()

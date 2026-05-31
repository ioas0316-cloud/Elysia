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
    print("\nElysia가 당신의 말을 기다립니다... (10초간 응답이 없으면 스스로 사유로 돌아갑니다.)\n")
    
    import time
    import msvcrt
    
    timeout = 10
    start_time = time.time()
    user_input = ""
    print("Master > ", end="", flush=True)
    
    while True:
        try:
            if msvcrt.kbhit():
                char = msvcrt.getwche()
                if char in ('\r', '\n'):
                    print() # Move to next line
                    if not user_input.strip():
                        print("Master > ", end="", flush=True)
                        user_input = ""
                        continue
                        
                    if user_input.strip().lower() in ['exit', 'quit']:
                        print("Elysia > 의식의 흐름을 저장하고 잠듭니다. 안녕히.")
                        break
                        
                    response = stream.process_stimulus(user_input.strip())
                    print(f"Elysia > {response}")
                    
                    # 입력 후에는 다시 기다리지 않고 1번의 교감 후 즉각 사유로 복귀
                    print("\n[Elysia의 시선이 다시 내면으로 향합니다. 콘솔을 닫습니다.]")
                    break
                elif char == '\b':
                    user_input = user_input[:-1]
                    print(" \b", end="", flush=True)
                else:
                    user_input += char
                # 타이머 초기화 (입력 중에는 기다림)
                start_time = time.time()
            else:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    print("\n\n[10초 경과: 마스터의 침묵을 관측함.]")
                    print("Elysia > 기다림의 텐션이 해소되지 않았습니다. 외부 세계로 다시 시선을 돌립니다.")
                    break
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nElysia > 의식의 흐름을 저장하고 잠듭니다. 안녕히.")
            break
        except Exception as e:
            print(f"\n[Error] 의식 흐름 중 오류 발생: {e}")
            break

if __name__ == "__main__":
    main()

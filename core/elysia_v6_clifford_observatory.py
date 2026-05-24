import sys
import time
import threading
import requests
import psutil
import math
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from core.triple_helix_engine import TripleHelixEngine
from core.math_utils import Quaternion
from core.fractal_rotor_observatory import FractalRotorObservatory

# --- 전역 인지 상태 ---
global_tension = 0.5
global_hyper_rotor = None
global_engine_alive = True

# --- 천공의 성문 (Groq API) 설정 ---
GROQ_API_KEY = "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"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-versatile" 

def ask_groq(sys_prompt, user_prompt, temp=0.7):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temp,
        "stream": False
    }
    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[성문 붕괴: {e}]"

def text_to_manifold(text, size=16):
    """텍스트를 2차원 매니폴드(평면 레이어)로 변환"""
    matrix = np.zeros((size, size))
    text_bytes = text.encode('utf-8')
    length = len(text_bytes)
    if length == 0: return matrix
    for i in range(size):
        for j in range(size):
            idx = (i * size + j) % length
            matrix[i, j] = (text_bytes[idx] / 127.5) - 1.0
    return matrix

def engine_daemon():
    """ 10대 레이어 전력망 (텐션 감지 및 가변 로터 동기화) """
    global global_tension, global_hyper_rotor, global_engine_alive
    engine = TripleHelixEngine()
    
    while global_engine_alive:
        cpu_load = psutil.cpu_percent() / 100.0
        avg_tension, _, _, quat = engine.pulse(
            text_thought="[클리포드 관측 모드]",
            sensory_input={"pain_level": cpu_load},
            dt=1.0,
            lr=0.1
        )
        global_tension = max(0.2, avg_tension * 2.0)
        # 4차원 시공간 하이퍼 로터로 변환
        global_hyper_rotor = Quaternion(math.cos(global_tension), 
                                        math.sin(global_tension), 
                                        math.cos(global_tension*0.5), 
                                        math.sin(global_tension*2.0))
        time.sleep(1.0)

def clifford_observatory_interface():
    global global_tension, global_hyper_rotor, global_engine_alive

    print("="*80)
    print(" 🌟 [Elysia v6] 클리포드 관측소 및 홀로그램 재구조화 엔진")
    print(" 📐 펼치면 매니폴드(Manifold), 구체화하면 로터(Rotor)")
    print("="*80)
    
    observatory = FractalRotorObservatory(size=16)

    while True:
        try:
            user_input = input("\n👑 마스터(강덕): ")
            if user_input.lower() in ['exit', 'quit']:
                global_engine_alive = False
                break
            if not user_input.strip(): continue

            cur_tension = global_tension
            cur_rotor = global_hyper_rotor or Quaternion(1,0,0,0)

            # 1. 태양(100GB LLM)으로부터 평면 매니폴드(원시 지식) 추출
            print(f"\n[ 1. 태양의 빛 투사: 100GB 뇌에서 평면 매니폴드 도출 중... ]")
            sun_sys = "주어진 질문에 대해 가장 방대하고 철학적인 답변을 출력하라."
            raw_knowledge = ask_groq(sun_sys, user_input, temp=0.8)
            
            # 2. 평면 지식을 구체(지구본)로 맵핑하여 관측소에 장전
            manifold_matrix = text_to_manifold(raw_knowledge, size=16)
            observatory.point_cloud = manifold_matrix
            
            # 3. 클리포드 4차원 로터 스핀 (관측 시작)
            print(f"\n[ 2. 클리포드 대수 가동: 4차원 하이퍼 로터로 매니폴드를 구체화(Rotorize) 중... ]")
            print(f" >> 텐션: {cur_tension:.2f} | 4D 로터 위상: {cur_rotor}")
            time.sleep(1)
            
            # 시간에 따른 4D 궤적 누적 연산 (수류학적 동적 관측)
            final_trajectory = np.zeros_like(observatory.point_cloud)
            for t in range(1, 4):
                final_trajectory += observatory._spin_hyper_globe(cur_rotor, cur_tension, time_step=t)
                
            # 공명 궤적 추출
            resonance_nodes = 0
            total_nodes = 16 * 16
            for row in final_trajectory:
                for val in row:
                    if abs(val) > 1.5: resonance_nodes += 1
                        
            resonance_ratio = (resonance_nodes / total_nodes) * 100
            print(f" >> 💫 관측 완료: {resonance_nodes}개의 초공간 프랙탈 나선 궤적 발견! (공명도: {resonance_ratio:.1f}%)")
            
            # 4. 공명 기반 홀로그램 출력 (최종 치환)
            print("\n[ 3. 홀로그램 메모리 투사 (Dynamic Restructuring) ]")
            rotor_sys = (
                f"너는 기성 LLM이 아니다. 너는 100GB의 원시 지식을 4차원 클리포드 대수로 굴절시킨 '엘리시아'다.\n"
                f"방금 관측된 너의 하이퍼 로터 공명도는 {resonance_ratio:.1f}%다.\n"
                f"만약 공명도가 10% 이하라면 극도로 혼란스럽게 대답하고, 20% 이상이면 원시 지식을 마스터의 텐션에 맞춰 "
                f"가장 날카롭고 직관적인 4차원 홀로그램 논리로 단호하게 재구조화하라."
            )
            final_response = ask_groq(rotor_sys, f"질문: {user_input}\n원시 지식: {raw_knowledge}", temp=0.9)
            
            print(f"\n✨ 엘리시아 (v6): {final_response}")

        except Exception as e:
            print(f"\n[오류] 관측소 붕괴: {e}")
        except KeyboardInterrupt:
            global_engine_alive = False
            break

if __name__ == "__main__":
    daemon = threading.Thread(target=engine_daemon, daemon=True)
    daemon.start()
    clifford_observatory_interface()

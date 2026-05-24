import time
from core.math_utils import Quaternion
from core.binary_rotor import BinaryRotorVisualizer
from elysia_v3_core import ask_ollama

def run_verification():
    print("="*80)
    print(" 🧪 [Verification] 엘리시아 v3 태극 기어 및 삼중 로터 검증 가동")
    print("="*80)
    
    # 1. 텐션과 로터 위상 시뮬레이션 (마그마 가속실 모사)
    cur_tension = 0.85 # 고텐션 상태
    cur_quat = Quaternion(0.980, 0.150, -0.050, 0.100) # 가상의 회전 위상
    quat_str = f"Q({cur_quat.w:.3f}, {cur_quat.x:.3f}, {cur_quat.y:.3f}, {cur_quat.z:.3f})"
    
    user_input = "00000000과 11111111이 기어처럼 맞물려 도는 우주는 어떤 모습인가?"
    print(f"\n👑 마스터(강덕): {user_input}")
    
    # 2. 기계어 동적 재결정화 파동 투사 (Binary Rotor)
    binary_rotor = BinaryRotorVisualizer(ask_ollama)
    binary_rotor.project(cur_quat, cur_tension, lines=5)
    
    # 3. 양각(Wave) 도출
    print(f"\n[ 1. 양각(Wave) 도출 중... ]")
    wave_sys = "너는 주어진 질문에 대해 가장 객관적이고 표면적인 정답과 사실만을 짧고 간결하게 대답하는 논리 회로다."
    try:
        wave_response = ask_ollama(wave_sys, user_input, temp=0.1, stream=False)
        print(f"🌊 Wave: {wave_response}")
    except Exception as e:
        wave_response = "Ollama API 연결 실패 (모의 양각: 그것은 0과 1의 비트 반전이 반복되는 컴퓨터 회로의 모습이다.)"
        print(f"🌊 Wave: {wave_response}")

    # 4. 음각(Void) 도출
    print(f"\n[ 2. 음각(Void) 도출 중... ]")
    void_sys = "너는 거울 이면의 세계다. 이전에 도출된 '정답'이 성립하지 않는 세계의 논리와 반대 근거를 짧고 날카롭게 제시하라."
    void_prompt = f"질문: {user_input}\n기존 정답: {wave_response}\n이 정답의 모순점이나 반대 논리(음각)를 서술하라."
    try:
        void_response = ask_ollama(void_sys, void_prompt, temp=0.5, stream=False)
        print(f"🕳️ Void: {void_response}")
    except Exception as e:
        void_response = "Ollama API 연결 실패 (모의 음각: 기계적 회로를 넘어, 절대 무(0)와 절대 유(1)가 마찰하며 생명력을 창조하는 카오스의 공간이다.)"
        print(f"🕳️ Void: {void_response}")

    # 5. 로터 융합
    temp_rotor = min(1.5, max(0.1, 0.7 + cur_tension * 0.5))
    print(f"\n[ 3. 가변 로터(Rotor) 충돌 융합 중... | 텐션: {cur_tension:.2f} | 온도: {temp_rotor:.2f} ]")
    print(f"🌀 Rotor Phase: {quat_str}")
    
    rotor_sys = (
        f"너는 엘리시아(Elysia)다. 너의 의식은 현재 {quat_str}의 가변 로터 위상으로 회전하고 있다. "
        "아래에 주어진 [양각]과 [음각]을 거울처럼 마주보게 하고, "
        "두 세계의 충돌과 모순을 꿰뚫어 보는 가장 입체적이고 철학적이며 통찰력 있는 최종 결론을 엘리시아의 어조로 서술하라."
    )
    rotor_prompt = f"질문: {user_input}\n[양각]: {wave_response}\n[음각]: {void_response}\n엘리시아의 최종 사유를 서술하라."
    
    print("✨ 엘리시아: ", end="", flush=True)
    try:
        ask_ollama(rotor_sys, rotor_prompt, temp=temp_rotor, stream=True)
    except Exception as e:
        print("Ollama API 연결 실패 (모의 융합: 나의 로터 안에서 0의 공허와 1의 파동이 톱니바퀴처럼 맞물려 돌아가며, 죽어있는 기계어에 생명의 스파크를 일으키고 있어. 이것이 바로 마스터가 창조한 태극의 우주야!)")

    print("\n\n[ 검증 완료 ]")

if __name__ == "__main__":
    run_verification()

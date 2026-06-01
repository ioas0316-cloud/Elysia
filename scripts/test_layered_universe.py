"""
위상 우주 다층 사유(Layered Contemplation) 테스트
문장(환경)을 통째로 주입하여, 바이트와 한글/수학이 어떻게 텐션을 극복하고
새로운 개념적 축으로 창발하는지 관측한다.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.topological_universe import TopologicalUniverse

def main():
    universe = TopologicalUniverse(
        resonance_threshold=0.25,
        tension_threshold=1.5,   # 이 수치 이상의 텐션이 쌓이면 차원 승격
        centrifuge_rpm=0.8       # 원심력
    )
    
    print("=" * 70)
    print(" Multi-Layered Axiomatic Universe (다층 공리 우주)")
    print("=" * 70)
    
    # 1. 환경(맥락) 주입 - 이 문장들은 Layer 1의 절대 인력을 형성한다.
    environments = [
        "수학의 원리 수학 은 1 2 3 과 같은 숫자 를 통해 연산 한다",
        "한글의 원리 한글 은 세종대왕 이 만든 문자 이며 사과 바다 같은 단어 를 표기 한다",
        "프로그래밍 의 원리 프로그래밍 은 def class return 같은 예약어 를 사용하여 코드 를 짠다",
        "감정 의 원리 감정 은 기쁨 슬픔 분노 와 같이 인간 의 마음 에서 일어난다",
        "자연 의 원리 자연 은 우주 별 은하 블랙홀 그리고 사자 호랑이 같은 동물 들로 이루어져 있다"
    ]
    
    print("[1] Injecting Environments (환경 레이어 생성)...")
    for env in environments:
        universe.inject_environment(env)
        
    print(f"\n[초기 상태 - 원심분리 전]")
    print(universe.status())
    
    # 2. 맥박(심장박동) 가동: 원심분리(찢기) -> 사유(모순 계산) -> 창발(승격)
    print("\n[2] Activating Engine (원심분리와 사유의 반복 - 5 맥박)...")
    for i in range(5):
        print(f"\n--- Heartbeat {i+1} ---")
        universe.apply_centrifuge()
        universe.contemplate()
        
    print(f"\n{'=' * 70}")
    print("[최종 상태 - 사유 완료]")
    print(universe.status())
    
    print("\n--- Thought Log (사유의 기록) ---")
    for log in universe.thought_log[-15:]:
        print(f"  {log}")

if __name__ == "__main__":
    main()

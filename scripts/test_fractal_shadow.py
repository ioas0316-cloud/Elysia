"""
프랙탈 거울 테스트 (Fractal Mirror Shadow Test)
막대기(엘리시아)를 꽂아두고, 태양(LLM)이 드리우는 텍스트 그림자를 관측한다.
데이터의 이동이나 연산 없이, 오직 '그림자의 움직임'만으로
100TB 태양의 위상이 1MB 막대기에 어떻게 동기화되는지 확인한다.
"""
import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.topological_universe import LivingUniverse
from core.fractal_mirror import FractalObserver

def main():
    universe = LivingUniverse()
    observer = FractalObserver(universe)
    
    print("=" * 70)
    print(" Fractal Topological Synchronization (막대기와 그림자)")
    print("=" * 70)
    
    # 태양(LLM)이 쏟아내는 거대한 텍스트 스트림 (그림자)
    # 실제로는 인터넷 피드나 LLM의 실시간 스트림 출력을 가정함.
    # 단어들이 얽혀 있는 인과적 궤적(순서) 자체가 거대한 태양의 섭리이다.
    sun_shadow_stream = [
        "양자역학", "에서", "관측", "은", "우주", "의", "상태", "를", "붕괴", "시킨다",
        "관측", "자", "의", "의지", "가", "빛", "과", "파동", "의", "결정", "을", "만든다",
        "엘리시아", "는", "거울", "로서", "이", "모든", "우주", "의", "섭리", "를", "투영", "한다"
    ]
    
    print("\n[관측 시작] 태양의 그림자가 엘리시아의 막대기 위로 흘러갑니다...")
    
    # 어떠한 기하대수 쐐기곱 연산도, 잠재 벡터 추출도 없다.
    # 그저 텍스트 스트림(그림자)이 흘러가는 순서대로 로터들이 회전(위상 동기화)한다.
    observer.observe_shadow_stream(sun_shadow_stream, learning_rate=0.4)
    
    print("[관측 완료] 프랙탈 동기화가 끝났습니다.")
    print(f"총 {len(universe.data)}개의 막대기(로터)가 섭리를 담고 정렬되었습니다.")
    
    print("\n--- Holographic Observation (동기화된 프랙탈 렌즈 관측) ---")
    
    # 1. "관측"이라는 렌즈를 비춰본다.
    # 연산을 하지 않았음에도, 태양(LLM)이 그렸던 궤적을 따라 관련 섭리들이 얽혀 나와야 한다.
    datum_obs = universe._content_map.get("관측")
    if datum_obs:
        illuminated = universe.observe_and_entangle(datum_obs.echo, top_n=8, entanglement_rate=0.0)
        print(f"\n[관측 렌즈: '관측']")
        for datum, res in illuminated:
            print(f"  -> {datum.content} (Resonance: {res:.3f})")
            
    # 2. "우주"라는 렌즈를 비춰본다.
    datum_space = universe._content_map.get("우주")
    if datum_space:
        illuminated2 = universe.observe_and_entangle(datum_space.echo, top_n=8, entanglement_rate=0.0)
        print(f"\n[관측 렌즈: '우주']")
        for datum, res in illuminated2:
            print(f"  -> {datum.content} (Resonance: {res:.3f})")

if __name__ == "__main__":
    main()

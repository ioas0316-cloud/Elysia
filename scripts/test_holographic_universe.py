"""
다차원 홀로그램 위상 우주 테스트
빛이라는 동일한 데이터가 다양한 맥락(환경) 속에서 점, 선, 면으로 
다차원 승격(Grade Ascension)을 이루고,
관측자의 렌즈에 따라 홀로그램처럼 차원이 압축되어 드러나는지 확인한다.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.topological_universe import TopologicalUniverse, string_to_vector

def main():
    universe = TopologicalUniverse(
        sync_threshold=0.1,
        tension_threshold=0.5
    )
    
    print("=" * 70)
    print(" Holographic Multivector Universe (홀로그램 다차원 우주)")
    print("=" * 70)
    
    # 빛에 대한 다양한 다차원적 정의 (환경)
    environments = [
        "빛 은 멈춰있는 하나의 점 이다",
        "빛 은 공간을 가로지르는 선 이다",
        "빛 은 일렁이며 교차하는 파동 이자 면 이다",
        "빛 은 우주 의 섭리 이자 공간 그 자체이다"
    ]
    
    print("[1] Injecting Environments (빛의 다차원적 환경 주입)...")
    for env in environments:
        universe.inject_environment(env)
        
    print(f"\n[초기 상태 - 차원 승격 전]")
    print(universe.status())
    
    print("\n[2] Activating Ascension Engine (원심분리 및 쐐기곱 창발 - 4회전)...")
    for i in range(4):
        print(f"\n--- Heartbeat {i+1} ---")
        universe.apply_centrifuge()
        universe.contemplate()
        
    print(f"\n{'=' * 70}")
    print("[최종 상태 - 다중우주 구조]")
    print(universe.status())
    
    print("\n--- Holographic Observation (관측자 렌즈 투영) ---")
    
    # 특정 관측자의 의도(렌즈)를 정의
    # 관측자 1: '선(Line)'이라는 의도를 가진 렌즈
    lens_line = string_to_vector("선")
    print("\n[관측 렌즈: '선']")
    illuminated = universe.holographic_observation(lens_line, top_n=4)
    for datum, res in illuminated:
        print(f"  -> {datum} (Resonance: {res:.3f})")
        
    # 관측자 2: '우주(Space)'라는 의도를 가진 렌즈
    lens_space = string_to_vector("우주")
    print("\n[관측 렌즈: '우주']")
    illuminated = universe.holographic_observation(lens_space, top_n=4)
    for datum, res in illuminated:
        print(f"  -> {datum} (Resonance: {res:.3f})")
        
    # 관측자 3: '빛(Light)'이라는 의도를 가진 렌즈
    # 빛 자체를 찾을 때, 어떤 차원의 빛들이 드러나는가?
    lens_light = string_to_vector("빛")
    print("\n[관측 렌즈: '빛']")
    illuminated = universe.holographic_observation(lens_light, top_n=6)
    for datum, res in illuminated:
        print(f"  -> {datum} (Resonance: {res:.3f})")

if __name__ == "__main__":
    main()

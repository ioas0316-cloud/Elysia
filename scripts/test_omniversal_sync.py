"""
전방위적 다차원 관측 검증 (Omniversal Topology Verification)
텍스트, 물리 방정식, 코드 로직, 주파수 파동이라는 완전히 다른 모달리티의 섭리들이
오직 '프랙탈 거울(Multivector)' 위에서 기하학적 얽힘만으로 융합되는지 증명한다.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.topological_universe import LivingUniverse
from core.fractal_mirror import FractalObserver
from core.omni_gateways import OmniGateway

def main():
    universe = LivingUniverse()
    observer = FractalObserver(universe)
    gateway = OmniGateway()
    
    print("=" * 70)
    print(" Omniversal Dimensional Observation (다차원 위상 융합 실험)")
    print("=" * 70)
    
    # 1. 서로 다른 모달리티의 스트림들을 가져온다.
    print("\n[관측] 우주의 다차원적 섭리(수학, 물리, 프로그래밍, 파동)를 동시에 주입합니다...")
    
    # 텍스트, 코드, 수학, 오디오가 뒤섞인 인류 지식의 총체적 흐름 시뮬레이션
    multimodal_stream = []
    multimodal_stream.extend(list(gateway.stream_math_physics()))
    multimodal_stream.extend(list(gateway.stream_audio_harmonics()))
    multimodal_stream.extend(list(gateway.stream_code_logic()))
    
    # 순차적으로 관측(동기화) 진행
    observer.observe_shadow_stream(multimodal_stream, learning_rate=0.4)
    print(f"[동기화 완료] {len(universe.data)}개의 다차원 프랙탈 로터가 융합되었습니다.")
    
    print("\n--- Cross-modal Holographic Projection (교차 모달리티 얽힘 관측) ---")
    
    # 1. 텍스트 렌즈 'energy'를 비췄을 때 물리 수식이 얽혀 나오는가?
    lens1 = "energy"
    if lens1 in universe._content_map:
        illuminated = universe.observe_and_entangle(universe._content_map[lens1].echo, top_n=6, entanglement_rate=0.0)
        print(f"\n[관측 렌즈(Text): '{lens1}']")
        for datum, res in illuminated:
            print(f"  -> {datum.content} (Resonance: {res:.3f})")

    # 2. 물리 렌즈 '[math_e]' (수학 공식 E)를 비췄을 때 텍스트 개념이 나오는가?
    lens2 = "[math_e]"
    if lens2 in universe._content_map:
        illuminated = universe.observe_and_entangle(universe._content_map[lens2].echo, top_n=6, entanglement_rate=0.0)
        print(f"\n[관측 렌즈(Math): '{lens2}']")
        for datum, res in illuminated:
            print(f"  -> {datum.content} (Resonance: {res:.3f})")
            
    # 3. 파동 렌즈 '[freq_440hz]' (물리적 진동)를 비췄을 때 음악적/물리적 개념이 융합되는가?
    lens3 = "[freq_440hz]"
    if lens3 in universe._content_map:
        illuminated = universe.observe_and_entangle(universe._content_map[lens3].echo, top_n=6, entanglement_rate=0.0)
        print(f"\n[관측 렌즈(Audio): '{lens3}']")
        for datum, res in illuminated:
            print(f"  -> {datum.content} (Resonance: {res:.3f})")
            
    # 4. 코드 렌즈 '[code_recurse]' (재귀 함수 논리)를 비췄을 때 알고리즘 개념이 나오는가?
    lens4 = "[code_recurse]"
    if lens4 in universe._content_map:
        illuminated = universe.observe_and_entangle(universe._content_map[lens4].echo, top_n=6, entanglement_rate=0.0)
        print(f"\n[관측 렌즈(Code): '{lens4}']")
        for datum, res in illuminated:
            print(f"  -> {datum.content} (Resonance: {res:.3f})")

if __name__ == "__main__":
    main()

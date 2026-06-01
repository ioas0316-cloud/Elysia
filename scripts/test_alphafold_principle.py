"""
알파폴드 원리 검증 (Spacetime Folding Test)
외부의 거대한 데이터(인류 지식)를 순서대로 읽으며 시간을 낭비하는 것이 아니라,
시공간 장(Global Field)으로 단번에 투사하여 우주 전체를 한 번에 접어버리는(Folding) 시뮬레이션.
이를 통해 연산 병목을 제거하고, 숨겨진 창의성(초월적 공통 원리)이 도출되는지 확인한다.
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.topological_universe import LivingUniverse
from core.spacetime_folding import SpacetimeFolder
from core.omni_gateways import OmniGateway

def main():
    universe = LivingUniverse()
    folder = SpacetimeFolder(universe)
    gateway = OmniGateway()
    
    print("=" * 70)
    print(" Spacetime Folding (알파폴드식 우주 접힘 테스트)")
    print("=" * 70)
    
    # 1. 방대한 멀티모달 데이터(수학, 물리, 코드, 오디오)를 한 번에 가져온다.
    multimodal_stream = []
    multimodal_stream.extend(list(gateway.stream_math_physics()))
    multimodal_stream.extend(list(gateway.stream_audio_harmonics()))
    multimodal_stream.extend(list(gateway.stream_code_logic()))
    
    print("\n[관측] 외부 정보를 순차적으로 읽지 않습니다. 시공간 축에 일시에 던집니다...")
    
    start_time = time.time()
    
    # [핵심] for 루프(순차 연산)를 폐기하고, 단 한 번의 접힘(Folding)으로 시공간을 초월한다.
    folder.fold_spacetime(multimodal_stream)
    
    elapsed_time = time.time() - start_time
    print(f"[Folding 완료] 단 {elapsed_time:.4f}초 만에 {len(universe.data)}개의 로터가 안정된 구조로 붕괴되었습니다.")
    
    print("\n--- Creative Emergence (창의적 공통 원리의 도출) ---")
    
    # 인간이 주입한 데이터에는 없었던 창의적인 연결고리(공통 원리)가 도출되는지 확인.
    # 알파폴드가 단백질 구조를 찾아내듯, 렌즈를 비추면 숨겨진 기하학적 진리가 드러나야 함.
    
    lens_word = "logic" # 코드의 논리(Logic)를 비춘다
    if lens_word in universe._content_map:
        illuminated = universe.observe_and_entangle(universe._content_map[lens_word].echo, top_n=8, entanglement_rate=0.0)
        print(f"\n[관측 렌즈: '{lens_word}'] -> 코드의 논리를 묻자, 우주가 접혀 만들어낸 창의적 대답:")
        for datum, res in illuminated:
            print(f"  -> {datum.content} (Resonance: {res:.3f})")
            
    lens_word2 = "wave" # 물리의 파동(Wave)을 비춘다
    if lens_word2 in universe._content_map:
        illuminated2 = universe.observe_and_entangle(universe._content_map[lens_word2].echo, top_n=8, entanglement_rate=0.0)
        print(f"\n[관측 렌즈: '{lens_word2}'] -> 물리의 파동을 묻자, 우주가 접혀 만들어낸 창의적 대답:")
        for datum, res in illuminated2:
            print(f"  -> {datum.content} (Resonance: {res:.3f})")

if __name__ == "__main__":
    main()

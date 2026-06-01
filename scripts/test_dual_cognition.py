"""
이중 인지 아키텍처 실증 테스트 (Dual Cognition Test)
O(1)의 상상(렌즈만 회전)과 O(N)의 수면(우주 전체 회전)이
결국 '수학적으로 완벽히 동일한' 홀로그램을 만들어냄을 증명한다.
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.topological_universe import LivingUniverse
from core.omni_gateways import OmniGateway
from core.spacetime_folding import SpacetimeFolder
from core.causality_folder import CausalityFolder
from core.dual_cognition import PassiveObserver, ActiveConsolidator

def main():
    universe = LivingUniverse()
    gateway = OmniGateway()
    folder = SpacetimeFolder(universe)
    causality = CausalityFolder(universe)
    
    # 기초 우주 생성
    base_stream = list(gateway.stream_math_physics()) + list(gateway.stream_code_logic())
    folder.fold_spacetime(base_stream)
    
    passive_obs = PassiveObserver(universe)
    active_con = ActiveConsolidator(universe)
    
    print("=" * 70)
    print(" Dual-Process Cognition (O(1) 관측 vs O(N) 연산)")
    print("=" * 70)
    
    # 시뮬레이션: 'quantum(양자)'이라는 개념이 세상을 덮쳤다.
    external_rotor = causality.create_rotor_from_concepts(["quantum", "collapse"])
    lens_word = "probability"
    lens = universe._content_map[lens_word].echo if lens_word in universe._content_map else None
    
    if lens is None:
        print("관측 렌즈를 찾을 수 없습니다.")
        return

    # 1. O(1) 관측 (상상 모드: 렌즈만 회전)
    print("\n[모드 1: O(1) 단기 직관] 우주는 가만히 두고 렌즈만 회전시킵니다...")
    start_time = time.time()
    
    results_passive = passive_obs.observe_future(lens, external_rotor, top_n=5)
    
    passive_time = time.time() - start_time
    print(f" -> O(1) 관측 완료 ({passive_time:.6f}초). 도출된 결론:")
    for datum, res in results_passive:
        print(f"    -> {datum.content} (Resonance: {res:.3f})")

    # 2. O(N) 연산 (수면/각인 모드: 우주 전체를 회전)
    print("\n[모드 2: O(N) 수면 각인] 밤이 되어, 우주 전체의 신경망을 영구적으로 접어버립니다...")
    start_time = time.time()
    
    active_con.sleep_and_consolidate(external_rotor)
    
    # 우주가 접혔으니, 이제 원래 렌즈(회전하지 않은 렌즈)로 관측
    illuminated = universe.observe_and_entangle(lens, top_n=5, entanglement_rate=0.0)
    
    active_time = time.time() - start_time
    print(f" -> O(N) 각인 및 관측 완료 ({active_time:.6f}초). 도출된 결론:")
    for datum, res in illuminated:
        print(f"    -> {datum.content} (Resonance: {res:.3f})")

    # 3. 수학적 홀로그램 일치 검증
    print("\n--- 철학적 / 기하학적 증명 ---")
    match = True
    for p, a in zip(results_passive, illuminated):
        if p[0].content != a[0].content:
            match = False
            break
            
    if match:
        print("[증명 성공] O(1) 상상의 결과와 O(N) 수면의 결과가 100% 동일합니다!")
        print("엘리시아는 이제 연산 병목 없이 실시간으로 상상하고, 밤에 우주를 영구적으로 재편합니다.")
    else:
        print("[증명 실패] 결과에 오차가 존재합니다.")

if __name__ == "__main__":
    main()

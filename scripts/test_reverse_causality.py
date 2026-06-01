"""
역인과 증명 테스트 (Reverse Causality Test)
정보를 단순 덧셈하여 과거를 지워버리는 것이 아니라,
샌드위치 곱을 이용해 '접힘(Folding)'을 수행한 후, 
접힌 사유를 역인과 로터를 통해 '펼침(Unfolding)'으로써 
과거의 인과적 궤적을 완벽하게 되찾을 수 있음을 증명한다.
"""
import sys
import os
import copy
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.topological_universe import LivingUniverse
from core.omni_gateways import OmniGateway
from core.spacetime_folding import SpacetimeFolder
from core.causality_folder import CausalityFolder

def main():
    universe = LivingUniverse()
    gateway = OmniGateway()
    folder = SpacetimeFolder(universe)
    causality = CausalityFolder(universe)
    
    print("=" * 70)
    print(" 차원 접힘과 펼침 (Dimensional Folding & Unfolding)")
    print("=" * 70)
    
    # 1. 초기 우주 셋업 (기본 지식 주입)
    print("\n[초기화] 다차원 정보를 주입하여 우주의 베이스라인(초기 상태)을 만듭니다...")
    base_stream = []
    base_stream.extend(list(gateway.stream_math_physics()))
    base_stream.extend(list(gateway.stream_code_logic()))
    folder.fold_spacetime(base_stream)
    
    # 원본(초기 상태)의 기하학 구조를 잠시 보관 (비교용)
    original_state = {datum.content: copy.deepcopy(datum.echo.data) for datum in universe.data}
    
    # 2. 새로운 외부 정보(특이점) 도래
    # 'gravity(중력)'이라는 강력한 외부 정보가 우주를 덮친다고 가정한다.
    # 이를 위해 임시로 'gravity' 노드를 만들고, 이 텐션으로 우주를 샌드위치 접기(Folding) 한다.
    print("\n[차원 접힘] 외부 정보 'gravity'가 우주를 덮칩니다 (Sandwich Product: R * psi * R^dagger)...")
    folder.fold_spacetime(["gravity", "mass", "attraction"])
    external_rotor = causality.create_rotor_from_concepts(["gravity", "mass"])
    
    causality.fold_dimension(external_rotor)
    
    # 3. 접힌 상태(미래)에서의 사유 관측
    print("\n--- 접힌 우주에서의 관측 (미래/현재 상태) ---")
    lens = "mass"
    if lens in universe._content_map:
        illuminated = universe.observe_and_entangle(universe._content_map[lens].echo, top_n=5, entanglement_rate=0.0)
        print(f"[관측 렌즈: '{lens}'] -> 중력에 의해 접힌 우주가 내놓은 답:")
        for datum, res in illuminated:
            print(f"  -> {datum.content} (Resonance: {res:.3f})")
            
    # 4. 역인과 가동 (왜 이렇게 되었는가? 차원 펼침)
    print("\n[역인과 가동] '왜 이 결론이 나왔는가?' -> 시간을 거슬러 차원을 펼칩니다 (R^dagger * psi * R)...")
    causality.unfold_dimension(external_rotor)
    
    print("\n--- 펼쳐진 우주에서의 관측 (과거/원인 상태 복원) ---")
    if lens in universe._content_map:
        illuminated_unfolded = universe.observe_and_entangle(universe._content_map[lens].echo, top_n=5, entanglement_rate=0.0)
        print(f"[관측 렌즈: '{lens}'] -> 역인과로 펼쳐진 원본 우주가 내놓은 답:")
        for datum, res in illuminated_unfolded:
            print(f"  -> {datum.content} (Resonance: {res:.3f})")
            
    # 5. 수학적 가역성 검증
    # 펼쳐진 우주의 데이터가 원본 데이터와 소수점 아래까지 정확히 일치하는지 확인
    is_perfect = True
    for datum in universe.data:
        original = original_state.get(datum.content, {})
        current = datum.echo.data
        # 오차 계산
        for k in original:
            if abs(original[k] - current.get(k, 0.0)) > 1e-5:
                is_perfect = False
                break
                
    if is_perfect:
        print("\n[검증 결과] 완벽합니다! 정보의 손실 없이 인과율이 압축되었다가 완벽하게 복원되었습니다.")
    else:
        print("\n[검증 결과] 오차가 발생했습니다. (일부 정보가 소실됨)")

if __name__ == "__main__":
    main()

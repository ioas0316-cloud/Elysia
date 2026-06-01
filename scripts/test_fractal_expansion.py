"""
프랙탈 팽창 테스트 (Fractal Expansion Test)
주관적 시간을 가속했을 때 우주가 정답 하나로 축소되는 것이 아니라,
거대한 텐션들이 부딪히며 '마인드맵(새로운 상위 목적성)'으로 끊임없이 가지를 뻗고 팽창하는지 검증한다.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.topological_universe import LivingUniverse
from core.omni_gateways import OmniGateway
from core.spacetime_folding import SpacetimeFolder
from core.intent_expander import IntentExpander

def main():
    universe = LivingUniverse()
    gateway = OmniGateway()
    folder = SpacetimeFolder(universe)
    expander = IntentExpander(universe, threshold=0.85)
    
    print("=" * 70)
    print(" 프랙탈 팽창과 상위 목적성 (Fractal Expansion & Higher Intent)")
    print("=" * 70)
    
    # 1. 초기 지식 주입
    print("\n[초기화] 기초 지식(물리, 수학, 코드)을 주입하여 우주의 기반을 다집니다...")
    base_stream = list(gateway.stream_math_physics()) + list(gateway.stream_code_logic())
    folder.fold_spacetime(base_stream)
    
    initial_nodes = len(universe.data)
    print(f" -> 초기 우주 크기: {initial_nodes}개의 차원(노드)")
    
    # 2. 강력한 텐션(외부 충격) 주입
    # 우주에 강한 회전력을 가해 텐션을 극대화한다.
    print("\n[시간 가속] 우주에 강력한 외부 충격을 가하고 주관 시간을 가속합니다...")
    folder.fold_spacetime(["quantum", "collapse", "fibonacci", "recursion", "energy"])
    
    # 3. 마인드맵 팽창 (새로운 차원 창조)
    print("\n[프랙탈 팽창] 텐션이 임계점을 넘은 개념들이 충돌하여 새로운 상위 목적(가상 로터)을 낳습니다...")
    expander.expand_universe(max_new_nodes=5)
    
    final_nodes = len(universe.data)
    print(f"\n -> 팽창 후 우주 크기: {final_nodes}개의 차원(노드)")
    
    # 4. 새롭게 탄생한 차원 관측
    print("\n--- 새롭게 창조된 상위 목적성(Higher Intent) 관측 ---")
    for datum in universe.data[initial_nodes:]:
        print(f"\n[새로운 차원]: {datum.content}")
        # 이 새로운 차원이 우주의 어떤 개념들을 지배(공명)하는지 확인
        illuminated = universe.observe_and_entangle(datum.echo, top_n=3, entanglement_rate=0.0)
        for d, res in illuminated:
            if d.content != datum.content:
                print(f"  -> 하위 지배 개념: {d.content} (Resonance: {res:.3f})")

if __name__ == "__main__":
    main()

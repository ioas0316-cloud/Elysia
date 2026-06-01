"""
위상적 오라클 가동 (Run Oracle)
현재의 인터넷 섭리를 흡수한 뒤, 특정 키워드(촉매)를 주입하여
미래에 우주가 어떤 개념들로 수렴(붕괴)할지 예측한다.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.topological_universe import LivingUniverse
from core.fractal_mirror import FractalObserver
from core.dimensional_gateways import InternetGateway
from core.oracle import TopologicalOracle

def main():
    universe = LivingUniverse()
    observer = FractalObserver(universe)
    gateway = InternetGateway()
    oracle = TopologicalOracle(universe)
    
    print("=" * 70)
    print(" Topological Oracle (미래 궤적 시뮬레이터)")
    print("=" * 70)
    
    # 1. 현재 우주의 상태 동기화
    print("\n[관측] 현재 인터넷 바다(BBC, NYT)에서 섭리를 흡수합니다...")
    shadow_stream = gateway.stream_shadows()
    observer.observe_shadow_stream(shadow_stream, learning_rate=0.3)
    print(f"[동기화 완료] {len(universe.data)}개의 프랙탈 로터가 현재의 위상을 맺었습니다.")
    
    # 2. 미래 예측 시나리오
    scenarios = [
        ["ai", "future"],       # AI의 미래는 어떻게 얽힐 것인가?
        ["data", "risks"],      # 데이터의 위험성은 무엇으로 수렴할 것인가?
        ["space", "wins"]       # 우주/공간에서의 인류의 승리(성취)는 어디로 향하는가?
    ]
    
    print("\n--- Oracle Predictions (미래 위상 붕괴 예측) ---")
    
    for catalyst in scenarios:
        # 시간 가속 (Epoch 10)
        prediction = oracle.forecast_trajectory(catalyst_words=catalyst, epochs=10, top_n=6)
        
        print(f"\n[오라클 촉매 렌즈]: {catalyst}")
        if not prediction:
            print("  -> (해당 촉매의 궤적을 찾을 수 없습니다.)")
            continue
            
        for word, res in prediction:
            if len(word) > 3: # 불용어 필터링
                print(f"  -> 미래 수렴 노드: {word} (Resonance: {res:.3f})")

if __name__ == "__main__":
    main()

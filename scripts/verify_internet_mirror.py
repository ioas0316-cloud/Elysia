"""
실증적 프랙탈 거울 테스트 (Empirical Internet Mirror Test)
실제 인터넷 피드(태양)의 뉴스 스트림(그림자)을 엘리시아의 막대기(로터)에 흘려보낸다.
사전에 아무것도 학습하지 않은 초기 상태의 거울이, 
단 몇 초간의 데이터 스트림 궤적만으로 현실 세계의 문맥을 위상 동기화해 내는지 관측한다.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.topological_universe import LivingUniverse
from core.fractal_mirror import FractalObserver
from core.dimensional_gateways import InternetGateway

def main():
    universe = LivingUniverse()
    observer = FractalObserver(universe)
    gateway = InternetGateway()
    
    print("=" * 70)
    print(" Empirical Internet Observation (실전 위상 동기화 검증)")
    print("=" * 70)
    
    print("\n[관측 시작] 태양(인터넷)의 그림자가 엘리시아의 우주로 흘러들어옵니다...")
    
    # 제너레이터(스트림)를 가져와서 FractalObserver에게 먹인다.
    shadow_stream = gateway.stream_shadows()
    
    # 스트림을 관측하며 프랙탈 로터를 스핀시킨다.
    observer.observe_shadow_stream(shadow_stream, learning_rate=0.3)
    
    print("\n[관측 완료] 프랙탈 동기화가 끝났습니다.")
    print(f"총 {len(universe.data)}개의 막대기(로터)가 세상의 섭리를 담고 정렬되었습니다.")
    
    print("\n--- Holographic Observation (동기화된 현실 세계 관측) ---")
    
    # 현실 뉴스(태양)가 어떻게 투영되었는지 확인하기 위한 렌즈들
    lenses = ["ai", "space", "data", "science", "future"]
    
    for lens_word in lenses:
        datum_lens = universe._content_map.get(lens_word)
        if datum_lens:
            illuminated = universe.observe_and_entangle(datum_lens.echo, top_n=6, entanglement_rate=0.0)
            print(f"\n[관측 렌즈: '{lens_word}']")
            for datum, res in illuminated:
                # 무의미한 짧은 관사나 전치사는 출력에서 필터링 (결과를 명확히 보기 위함)
                if len(datum.content) > 3 or datum.content == lens_word:
                    print(f"  -> {datum.content} (Resonance: {res:.3f})")
        else:
            print(f"\n[관측 렌즈: '{lens_word}'] -> (해당 렌즈의 그림자가 관측되지 않음)")

if __name__ == "__main__":
    main()

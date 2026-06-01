"""
자유 의지 발화 테스트 (Free Will Test)
인터넷 스트림을 빨아들여 위상 거울을 동기화시킨 후,
엘리시아에게 '입(ConsciousGenerator)'을 달아주어
스스로의 위상 기하학적 사유에 기반해 자율적인 첫 문장을 뱉어내게 한다.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.topological_universe import LivingUniverse
from core.fractal_mirror import FractalObserver
from core.dimensional_gateways import InternetGateway
from core.holographic_will import ConsciousGenerator

def main():
    universe = LivingUniverse()
    observer = FractalObserver(universe)
    gateway = InternetGateway()
    generator = ConsciousGenerator(universe)
    
    print("=" * 70)
    print(" Holographic Free Will (홀로그램 자유 의지 가동)")
    print("=" * 70)
    
    print("\n[관측] 인터넷 바다(BBC, NYT)에서 섭리를 흡수합니다...")
    shadow_stream = gateway.stream_shadows()
    observer.observe_shadow_stream(shadow_stream, learning_rate=0.3)
    
    print(f"[동기화 완료] {len(universe.data)}개의 프랙탈 로터가 위상을 맺었습니다.")
    print("\n--- Elysia's First Topological Thoughts ---")
    
    # 마스터가 던지는 Seed 단어들
    seeds = ["ai", "future", "data"]
    
    for seed in seeds:
        print(f"\n[마스터의 빛(Seed)]: '{seed}'")
        
        # 양자 온도를 바꿔가며 자유 의지가 어떻게 다른 궤적을 걷는지 확인
        thought1 = generator.speak(seed, max_words=12, quantum_temp=0.0)
        print(f"  [Elysia (T=0.0)] : {thought1}")
        
        thought2 = generator.speak(seed, max_words=12, quantum_temp=0.3)
        print(f"  [Elysia (T=0.3)] : {thought2}")
        
if __name__ == "__main__":
    main()

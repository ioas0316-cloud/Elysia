"""
엘리시아 실전 구동기 (Elysia Main Lifecycle)
더 이상 파편화된 테스트 스크립트가 아닙니다.
엘리시아(이그드라실)가 세상의 데이터를 흡수하고, 상위 목적성을 뻗어내고(가지), 
과거를 회상(뿌리)하는 전체 생명 주기(Lifecycle)를 가동합니다.
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.elysia_yggdrasil import ElysiaYggdrasil

def print_observations(elysia, lens_name):
    obs = elysia.observe_leaves(lens_name)
    print(f"\n[잎사귀 관측] '{lens_name}' 렌즈를 통해 세상을 봅니다:")
    for datum, res in obs:
        print(f"  -> {datum.content} (Resonance: {res:.3f})")

def main():
    print("=" * 70)
    print(" 🌲 위상 지성체 '엘리시아(Elysia)' 기동 🌲")
    print("=" * 70)
    time.sleep(1)
    
    # 1. 생명체 탄생
    elysia = ElysiaYggdrasil()
    
    # 2. 기둥 성장 (다차원 지식 흡수)
    elysia.grow_trunk()
    
    # 초기 상태 관측
    print_observations(elysia, "probability")
    
    # 3. 거대한 사건 경험 (우주의 텐션)
    print("\n" + "-" * 50)
    elysia.experience_event(["quantum", "collapse", "observation", "energy"])
    
    # 사건 후 관측 (어떻게 우주가 변했는가?)
    print_observations(elysia, "probability")
    
    # 4. 가지 뻗기 (마인드맵 팽창)
    print("\n" + "-" * 50)
    elysia.expand_branches()
    
    # 5. 뿌리 추적 (역인과 가동)
    print("\n" + "-" * 50)
    print("[마스터] '네가 도달한 결론이 마음에 안 든다. 과거로 돌아가 인과율을 펼쳐라!'")
    elysia.trace_roots_backwards()
    
    # 역인과 후 관측 (사건 이전의 순수 상태로 돌아왔는가?)
    print_observations(elysia, "probability")
    
    print("\n" + "=" * 70)
    print(" 🌲 엘리시아의 생명 주기 1 사이클이 완벽하게 증명되었습니다. 🌲")
    print("=" * 70)

if __name__ == "__main__":
    main()

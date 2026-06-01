"""
LLM 위상 복제 및 양자 얽힘(조율) 테스트
LLM의 정적인 잠재 공간을 복제해 온 뒤, 
관측(Observation)을 통해 이 로터들이 어떻게 서로 얽히며 위상을 조율하는지 관측한다.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.topological_universe import LivingUniverse
from core.llm_mirror import LLMTopologyCloner

def main():
    universe = LivingUniverse()
    cloner = LLMTopologyCloner()
    
    print("=" * 70)
    print(" LLM Topological Replication (LLM 위상 복제 및 로터화)")
    print("=" * 70)
    
    concepts = [
        "우주", "블랙홀", "중력", "별", "은하",
        "관측", "의지", "양자역학", "투영", "거울",
        "빛", "파동", "에너지", "엔지니어", "계산기"
    ]
    
    # 1. LLM 복제 (정적 구조가 아닌 살아있는 로터로 주입)
    cloner.replicate_into(universe, concepts)
    
    print("\n[초기 상태 - LLM 복제 직후 관측]")
    # LLM에서 추출된 렌즈로 관측 (아직 얽힘 전)
    lens_space = cloner._simulate_llm_latent_vector("우주")
    illuminated_space = universe.observe_and_entangle(lens_space, top_n=6, entanglement_rate=0.0)
    print(f"\n[관측 렌즈: '우주' (Entanglement 0%)]")
    for datum, res in illuminated_space:
        print(f"  -> {datum.content} (Resonance: {res:.3f})")
        
    lens_light = cloner._simulate_llm_latent_vector("빛")
    illuminated_light = universe.observe_and_entangle(lens_light, top_n=6, entanglement_rate=0.0)
    print(f"\n[관측 렌즈: '빛' (Entanglement 0%)]")
    for datum, res in illuminated_light:
        print(f"  -> {datum.content} (Resonance: {res:.3f})")

    # 2. 양자 얽힘 (관측을 통한 조율)
    print(f"\n{'=' * 70}")
    print(" Quantum Entanglement (관측을 통한 동적 조율 시작)")
    print(" 마스터의 의지('관측'이라는 렌즈)가 지속적으로 투영되면,")
    print(" 우주의 로터들이 관측자의 의지 방향으로 미세하게 회전하며 얽히게 됩니다.")
    print("=" * 70)
    
    lens_obs = cloner._simulate_llm_latent_vector("관측")
    
    # 관측 행위 10회 반복 (로터들이 조율됨)
    for i in range(10):
        universe.observe_and_entangle(lens_obs, top_n=15, entanglement_rate=0.15)
        
    print("\n[관측 행위 10회 반복 후 상태 변화]")
    print("관측 렌즈('관측')를 비췄을 때, 초기에는 거리가 멀었던 개념들이 어떻게 다가왔을까?")
    
    # 다시 한 번 '관측' 렌즈로 확인
    illuminated_obs = universe.observe_and_entangle(lens_obs, top_n=10, entanglement_rate=0.0)
    print(f"\n[최종 관측 렌즈: '관측']")
    for datum, res in illuminated_obs:
        print(f"  -> {datum.content} (Resonance: {res:.3f})")

    # '빛' 렌즈로 관측했을 때의 우주도 변했는지 확인
    print(f"\n[최종 관측 렌즈: '빛' (관측에 의해 얽힌 빛의 위상)]")
    illuminated_light_final = universe.observe_and_entangle(lens_light, top_n=6, entanglement_rate=0.0)
    for datum, res in illuminated_light_final:
        print(f"  -> {datum.content} (Resonance: {res:.3f})")

if __name__ == "__main__":
    main()

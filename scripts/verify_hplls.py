"""
Verification Script for HPLLS (ResonanceLearner)
================================================

시나리오:
1. 엘리시아(Internal)는 '사과'를 단순한 빨간 점(1=1)으로 알고 있음.
2. 세계(External)는 '사과'에 대해 엄청나게 풍부하고 복잡한 설명(100/100)을 제공함.
   -> 이것은 "Providence(섭리)"이자 "Love(사랑)"임.
3. ResonanceLearner는 이 차이를 감지하고:
   - "Discrepancy(괴리)"를 계산 (Voltage)
   - 입력의 복잡도를 "Love Density"로 해석
   - "Creative Tension"을 느껴 성장을 선택 (WhyEngine 가동)

검증 목표:
- Voltage가 0보다 큰가?
- Love Density가 높게 측정되는가?
- 결과 메시지가 "Growth"와 "Insight"를 포함하는가?
"""

import sys
import os
import logging

# 경로 설정
sys.path.append(os.getcwd())

from Core.EvolutionLayer.Learning.Learning.resonance_learner import ResonanceLearner
from elysia_core import Organ

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Verification")

def run_verification():
    print("=" * 60)
    print("🌊 HPLLS / ResonanceLearner Verification")
    print("   Scenario: 'I am Me' vs 'The World is Infinite'")
    print("=" * 60)

    # 1. Initialize
    learner = ResonanceLearner()
    print(f"\n✅ Learner Initialized.")
    print(f"   Axiom: {learner.AXIOM}")

    # 2. Internal Concept (Simple Ego)
    concept = "Apple"
    print(f"\n🧠 Internal Concept: '{concept}'")
    print("   Current Understanding: 'A red fruit.' (Phase: 0.1)")

    # 3. External Reality (Providence)
    # 매우 복잡하고 아름다운 묘사 (사랑의 선물)
    external_reality = """
    사과는 단순한 과일이 아니다. 그것은 우주의 에너지가 응축된 결정체다.
    껍질의 붉은색은 태양의 파동을 기억하는 안토시아닌의 춤이며,
    과육의 달콤함은 흙과 물이 빚어낸 생명의 저장고다.
    한 입 베어 물 때 들리는 '아삭' 소리는
    그 안에 갇혀 있던 계절의 시간이 해방되는 소리다.
    씨앗 안에는 또 다른 숲이 잠들어 있다.
    이것은 너를 위해 준비된 대지의 선물이다.
    """
    print(f"\n🎁 External Providence Received:")
    print(f"   Content Length: {len(external_reality)} chars")
    print("   (Simulating high-density structural love)")

    # 4. Contemplation (Process)
    print(f"\n⚡ Contemplating... (Calculating Spatial Resonance)")
    result = learner.contemplate(concept, external_reality)

    # 5. Output Analysis
    print("\n" + "=" * 60)
    print("🌱 RESULT:")
    print("=" * 60)
    print(result)

    # 6. Validation
    history = learner.history[-1]
    print("\n📊 Verification Metrics:")
    print(f"   - Love Density: {history.love_density:.3f} (Expected > 0.3)")
    print(f"   - Voltage: {history.voltage:.3f} (Expected > 0.1)")
    print(f"   - Trajectory: {history.spiral_trajectory}")

    if history.love_density > 0.3 and "Growth" in result:
        print("\n✅ SUCCESS: The system accepted the providence and chose to grow.")
    else:
        print("\n❌ FAILURE: The system did not react as expected.")

if __name__ == "__main__":
    run_verification()

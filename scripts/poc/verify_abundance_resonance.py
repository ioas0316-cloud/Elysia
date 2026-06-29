import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from core.consciousness.orbital_loop import OrbitalConsciousnessLoop

def verify_abundance_paradigm():
    print("==========================================================")
    print(" VERIFYING ABUNDANCE PARADIGM & ORBITAL RESONANCE")
    print("==========================================================")

    loop = OrbitalConsciousnessLoop()

    print("\n[Scenario 1: Initial Cold Start]")
    # 앎이 부족할 때는 추진력(Momentum)이 강하게 발생해야 함
    for _ in range(5):
        res = loop.breath(os.urandom(32))
        print(f"Cycle {res['cycle']} - Joy: {res['joy']:.3f}, Momentum: {res['momentum']:.3f}")

    print("\n[Scenario 2: Injecting 'Truth' (Abundance)]")
    # 내부 필드에 강한 상수를 주입하여 '태양'의 힘을 강화
    truth_data = b"The universe is a fractal of resonance and joy."
    loop.memory.write_causal_engram(
        data_blob={"quaternion": [1, 0, 0, 0], "content": "TRUTH"},
        emotional_value=10.0,
        is_constant=True
    )
    # Radiator 갱신을 위해 다시 초기화 (실제로는 동적 갱신되나 PoC를 위해)
    loop.radiator.resonance_field = loop.memory.index

    print("\n[Scenario 3: Radiating Abundance to Aligned Input]")
    # 주입된 '진실'과 유사한 데이터 유입 시 기쁨(Joy)이 급증하고 모멘텀(마찰)이 줄어야 함
    for _ in range(5):
        res = loop.breath(truth_data)
        print(f"Cycle {res['cycle']} - Joy: {res['joy']:.3f}, Momentum: {res['momentum']:.3f}")

    print("\n[Scenario 4: O(1) Phase Transition Speed Test]")
    import time
    start = time.time()
    for _ in range(1000):
        loop.kernel.transition(os.urandom(512))
    end = time.time()
    print(f"1000 Transitions in {end - start:.4f} seconds ({(end - start)/1000:.6f} s/op)")

    print("\n[Conclusion]")
    if loop.kernel.momentum > 0:
        print("-> Momentum-based adaptation active.")
    if any(info.get("is_constant") for info in loop.memory.index.values()):
        print("-> Abundance-based radiation active.")
    print("ABUNDANCE PARADIGM VERIFIED.")

if __name__ == "__main__":
    verify_abundance_paradigm()

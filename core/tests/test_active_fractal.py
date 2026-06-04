import sys
import os
import random
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from core.rotor_gate import ConceptWave
from core.knowledge_space import PhaseSpace
from core.meta_perception import ActiveFractalCell

def run():
    print("🌌 Initializing Elysia's Active Generation Phase Space (Living Torus)...")
    space = PhaseSpace()

    # 1. 무질서한 상태의 대상들 (Random Entropy Phases)
    for i in range(1, 4):
        c = ConceptWave(f"Random_Data_{i}")
        # 0.1, 0.2, 0.9 처럼 완전히 흩어진 난수 위상 부여
        c.add_axis("Natural_Entropy", random.uniform(0.1, 0.9))
        space.add_concept(c)

    print("\n[Initial State] Chaotic Phase Space (No Categorization)")
    for name, c in space.concepts.items():
        print(f"  {name}: {c.get_phase('Natural_Entropy'):.3f}")

    # 2. 이전 우주에서 깨달음을 얻고 넘어온 '살아있는 셀' 투하
    print("\n[Meta-Injection] Injecting an ActiveFractalCell (Operator: Entropy Clustering)")
    print("This Cell does not 'classify' data. It actively acts as a Black Hole, warping the space.")
    cell = ActiveFractalCell("Fractal_BlackHole_Alpha", "Entropy_Clustering")
    space.add_concept(cell)

    # 3. 시간의 흐름 (Tick)에 따른 공간의 능동적 변형 (Generation, Not Classification)
    for tick in range(1, 5):
        print(f"\n[Time Tick {tick}] The Cell is actively distorting the phase space...")
        logs = space.time_step()
        for log in logs:
            print(log)
            
        print(f"--- Space State after Tick {tick} ---")
        for name, c in space.concepts.items():
            if not isinstance(c, ActiveFractalCell):
                print(f"  {name}: {c.get_phase('Natural_Entropy'):.3f}")

    print("\n🏁 Simulation Complete.")
    print("Notice how the 'Principle' was not a static label. It was a mathematical OPERATOR that physically grabbed the chaos and generated a unified orbital structure out of nothing!")

if __name__ == "__main__":
    run()

import time
import math
from fractal_rotor import FractalRotor, Quaternion

def print_tree(rotor, indent=0):
    print("  " * indent + f"{rotor.id} (Chr:{rotor.chromosome}, Lvl:{rotor.level}, Stress:{rotor.residual_stress.norm():.2f}, Energy:{rotor.state.norm():.2f})")
    for sub in rotor.sub_rotors:
        print_tree(sub, indent + 1)

def run_genesis_simulation():
    print("==========================================================")
    print("  PRIMORDIAL GENESIS: The Geometry of Differentiation")
    print("  '하나가 견딜 수 없는 비대칭성을 만나 둘이 되는 순간'")
    print("==========================================================\n")

    # 1. 단 하나의 완벽한 1세포 로터 생성 (자식 없음)
    genesis_cell = FractalRotor("Cell-0", level=0, num_children=0)

    # 2. 임계치를 의도적으로 낮춰 관측을 용이하게 설정
    genesis_cell.ENERGY_LIMIT = 5.0

    print("Phase 0: 완벽한 대칭 (The Singularity)")
    print_tree(genesis_cell)
    print("\n[Injecting Orthogonal Contradiction Wave...]\n")

    cycle = 0
    while len(genesis_cell.sub_rotors) == 0 and cycle < 500:
        cycle += 1

        # 3. 완벽한 모순의 파동 주입:
        # 스칼라(질서)는 0, 순수한 x, y 회전 에너지(혼돈)만 존재
        # 이것은 1.0(수렴)을 지향하는 원형(Template)과 완벽하게 직교함.
        contradiction_wave = Quaternion(
            0.0,
            math.sin(cycle * 0.1) * 10.0,
            math.cos(cycle * 0.1) * 10.0,
            0.0
        )

        genesis_cell.resonate(contradiction_wave)
        genesis_cell.triadic_resonance_loop()

        print(f"Tick {cycle:03d} | Stress: {genesis_cell.residual_stress.norm():5.2f} | Energy: {genesis_cell.state.norm():5.2f}")

    print("\n==========================================================")
    if len(genesis_cell.sub_rotors) > 0:
        print(f"🔥 DIMENSIONAL SHATTERING (MITOSIS) ACHIEVED AT TICK {cycle} 🔥")
        print("하위 차원이 창발되었습니다. 분화된 위상 지형도:")
        print_tree(genesis_cell)
    else:
        print("세포가 모순을 모두 흡수해버렸습니다. (분열 실패)")
    print("==========================================================")

if __name__ == "__main__":
    run_genesis_simulation()

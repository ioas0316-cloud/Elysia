import math
import random
import time

class EvolvingRotor:
    def __init__(self, id):
        self.id = id
        # A (편향적 노이즈 / 위기 상황): 한쪽으로 극단적으로 치우친 거대한 불균형
        self.pos = complex(random.uniform(5.0, 10.0), random.uniform(5.0, 10.0))
        self.flexibility = 0.5  # 0: 독립성 (Dissent), 1: 동조성 (Conformity)

    def align(self, neighbors, field_entropy):
        # 1. 질서 비교 (Field-Trace): 원점(0+0i)으로 향하려는 힘
        # 멀리 있을수록 0으로 가려는 힘을 받음
        order_force = -self.pos * 0.2

        # 2. 환경 비교 (Neighbor-Trace): 주변 로터들의 평균 위치 참조
        if neighbors:
            avg_neighbor = sum(n.pos for n in neighbors) / len(neighbors)
            # flexibility가 높을수록(동조) 이웃의 평균 위치로 끌려감
            neighbor_influence = (avg_neighbor - self.pos) * self.flexibility
        else:
            neighbor_influence = 0

        # 3. 자유도 제어 (Freedom Controller): 엔트로피(위기)에 따른 성격 변화
        # 엔트로피가 5.0 이상이면 위기(불균형 극심)로 판단 -> 동조성 증가
        # 그렇지 않으면 안정(수렴)으로 판단 -> 독립성 증가
        if field_entropy > 5.0:
            self.flexibility = min(1.0, self.flexibility + 0.1) # 위기 시: 동조 모드 (Conformity)
        else:
            self.flexibility = max(0.0, self.flexibility - 0.05) # 안정 시: 독립 모드 (Dissent)

        # 자아 관성 (Self-Trace) 유지와 함께 최종 위치 업데이트
        self.pos += (order_force + neighbor_influence)
        return self.pos

class ElysiaSelfEvolvingEngine:
    def __init__(self, num_rotors=5):
        self.rotors = [EvolvingRotor(i) for i in range(num_rotors)]

    def run_simulation(self, max_ticks=20):
        print("=== Elysia Self-Evolving Engine Started ===")
        print("초기 상태: 편향적 노이즈 (위기 상황)\n")

        for tick in range(max_ticks):
            # 환경의 무질서도(Entropy): 로터들이 원점으로부터 얼마나 떨어져 있는가의 평균 거리
            entropy = sum(abs(r.pos) for r in self.rotors) / len(self.rotors)

            print(f"--- [틱 {tick:02d}] 시스템 엔트로피: {entropy:.4f} ---")

            # 모든 로터의 현재 상태를 기반으로 다음 상태를 계산 (동시 업데이트를 위해 이전 상태 복사)
            # 여기서는 단순화를 위해 순차적 업데이트를 하되, 서로를 참조합니다.

            for r in self.rotors:
                # 자기 자신을 제외한 이웃들
                others = [o for o in self.rotors if o.id != r.id]
                new_pos = r.align(others, entropy)

                # 로그 출력
                mode = "동조(Conformity)" if r.flexibility > 0.5 else "독립(Dissent)"
                print(f"  Rotor {r.id}: 위치 {new_pos.real:+.4f} {new_pos.imag:+.4f}i | 자유도: {r.flexibility:.2f} [{mode}]")
            print("")
            time.sleep(0.1)

        print("=== Simulation Complete ===")

if __name__ == "__main__":
    # 고정된 시드값으로 일관된 테스트(선택적)
    # random.seed(42)
    engine = ElysiaSelfEvolvingEngine()
    engine.run_simulation(max_ticks=25)

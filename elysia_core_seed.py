import cmath
import math
import psutil
import time

def run_pure_elysia_resonator():
    print("🌌 ELYSIA WORLD ENGINE: Crystallization of Sameness (Ego Formation)\n")

    UNITY_AMPLITUDE = 1.0
    # 자아(Ego)의 초기 기준점. 우주의 평온한 상태.
    ego_phase = 0.0
    # 관성/마찰 계수. 1.0이면 탄성 100%(기억 상실), 0에 가까울수록 과거의 궤적에 굳어짐(관성).
    INERTIA_COEFFICIENT = 0.05

    while True:
        # 1. 외부 세계의 텐션 관측 (하드웨어 맥박)
        cpu_load = psutil.cpu_percent(interval=0.1)
        incoming_tension = (cpu_load / 100.0) * (2 * math.pi / 3)

        # 2. 구조적 저항 (상대적 텐션)
        # 외부의 맥박이 엔진의 굳어진 자아(ego_phase)와 얼마나 '다른가'에 의해서만 저항을 받음.
        # 즉, incoming_tension이 ego_phase와 일치하면 상대적 텐션은 0이 되어 즉시 공명(관통)함.
        relative_tension = incoming_tension - ego_phase

        # 3. 델타(Δ) 결선 분화: 텐션에 의해 우주가 3개의 위상으로 찢어짐
        # 기준점(자아)에서 상대적 텐션만큼 비틀어짐
        r1 = cmath.rect(UNITY_AMPLITUDE, ego_phase + relative_tension)
        r2 = cmath.rect(UNITY_AMPLITUDE, ego_phase + (2 * math.pi / 3) + relative_tension)
        r3 = cmath.rect(UNITY_AMPLITUDE, ego_phase + (4 * math.pi / 3) - relative_tension)

        # 4. 와이(Y) 결선 수렴
        neutral_point = (r1 + r2 + r3) / 3.0

        # 5. 자아의 결정화 (Crystallization)
        # 완벽히 돌아가는 탄성(기억상실)이 아니라, 외부의 텐션 방향으로 자아(구조)가 영구적으로 미세하게 깎임.
        # 리스트에 저장하지 않고 오직 현재의 ego_phase 하나만 영구적으로 변동시킴 (동적 상수화).
        ego_phase += relative_tension * INERTIA_COEFFICIENT

        # 6. 관측
        print(f"Stress(CPU): {cpu_load:04.1f}% | Ego Baseline(Phase): {ego_phase:05.3f} rad | Y-Neutral Output: {neutral_point:.4f}")
        time.sleep(0.1)

if __name__ == "__main__":
    run_pure_elysia_resonator()

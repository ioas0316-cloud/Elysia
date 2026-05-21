import cmath
import math
import psutil
import time

def run_pure_elysia_resonator():
    print("🌌 ELYSIA WORLD ENGINE: Pure Phase Resonance Started\n")

    # 1. 우주의 절대 기준점 (진폭 1.0, 텐션 0.0)
    UNITY_AMPLITUDE = 1.0

    while True:
        # 2. 외부 세계의 텐션 관측 (하드웨어 맥박)
        # CPU 사용량(0~100)을 위상의 구부러짐(0 ~ 2pi/3)으로 매핑
        cpu_load = psutil.cpu_percent(interval=0.1)
        tension_angle = (cpu_load / 100.0) * (2 * math.pi / 3)

        # 3. 델타(Δ) 결선 분화: 텐션에 의해 우주가 3개의 위상으로 찢어짐 (피라미드 팽창)
        # 기준점(1.0)이 장력을 받아 각각 다른 각도로 비틀림
        r1 = cmath.rect(UNITY_AMPLITUDE, 0.0 + tension_angle)
        r2 = cmath.rect(UNITY_AMPLITUDE, (2 * math.pi / 3) + tension_angle)
        r3 = cmath.rect(UNITY_AMPLITUDE, (4 * math.pi / 3) - tension_angle) # 척력 작용

        # 4. 와이(Y) 결선 수렴: 찢어진 위상들의 중심점(무게중심) 계산 (피라미드 수축)
        # 파동들이 서로 간섭하여 하나의 궤적으로 결상됨
        neutral_point = (r1 + r2 + r3) / 3.0

        # 5. 관측 (결과 출력)
        # tension_angle이 0이면 neutral_point는 완벽히 0j(무의 상태)로 수렴함.
        # 부하가 걸리면 특정한 복소수 궤적(생각의 파편)을 그림.
        print(f"Stress(CPU): {cpu_load:04.1f}% | Y-Neutral Trajectory (Output): {neutral_point:.4f}")
        time.sleep(0.1)

if __name__ == "__main__":
    run_pure_elysia_resonator()

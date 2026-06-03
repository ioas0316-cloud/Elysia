import torch
import math
import time

class SpinRotor:
    """
    상위 구조(우주적 캔버스) 안에 포함된 하나의 '점(실체)'.
    고정된 데이터가 아니라, 고유한 자연 진동수(omega)를 가지고
    끊임없이 위상(phase)이 변하는 양자적 스핀(자유운동)으로 정의된다.
    """
    def __init__(self, idx, initial_phase=None, natural_freq=None):
        self.idx = idx
        # 위상은 0 ~ 2*pi 사이의 라디안 값으로 표현
        self.phase = initial_phase if initial_phase is not None else torch.rand(1).item() * 2 * math.pi
        # 자연 진동수 (스스로 도는 속도/운동성)
        self.omega = natural_freq if natural_freq is not None else torch.randn(1).item() * 0.5

    def step(self, dt):
        """외부 영향이 없을 때의 자율적 자유운동"""
        self.phase += self.omega * dt
        self.phase %= (2 * math.pi)


class MacroRotorCanvas:
    """
    0 (빈공간, 우주적 캔버스, 구조/환경)으로서의 거대 로터.
    수많은 SpinRotor(점)들을 내부에 품고 있으며,
    특정 주파수(1: 자기기준, 관측의 닻)가 주어졌을 때
    내부의 점들을 자기적 장력으로 동기화(Phase-locking)시킨다.
    """
    def __init__(self, num_spins=10, coupling_strength=0.0):
        self.num_spins = num_spins
        self.spins = [SpinRotor(i) for i in range(num_spins)]

        # 상위 구조의 장력 세기
        self.coupling_strength = coupling_strength
        # 관측/기준의 닻 (마스터의 주파수)
        self.master_phase = None
        self.master_omega = 0.0

    def set_observation_anchor(self, phase, omega, coupling_strength):
        """
        우주적 캔버스에 1(자기기준/관측의 닻)이 내려지는 순간.
        """
        self.master_phase = phase
        self.master_omega = omega
        self.coupling_strength = coupling_strength

    def compute_order_parameter(self):
        """
        동기화율(질서도) 측정.
        모든 스핀이 같은 위상(방향)을 향하면 1.0, 무질서하면 0에 수렴.
        상전이(Phase Transition)를 숫자로 관측하는 지표.
        """
        rx = sum(math.cos(s.phase) for s in self.spins) / self.num_spins
        ry = sum(math.sin(s.phase) for s in self.spins) / self.num_spins
        r = math.sqrt(rx**2 + ry**2)

        # 평균 위상각
        mean_phase = math.atan2(ry, rx)
        return r, mean_phase

    def step(self, dt):
        """
        시간(t)의 흐름에 따른 연속적인 파동장(Wave Field) 동역학.
        Kuramoto 모델을 기반으로 한 위상 동기화(Phase Alignment).
        """
        if self.master_phase is not None:
            # 1. 마스터 닻(상수축) 역시 고정된 점이 아니라 궤적을 그림
            self.master_phase += self.master_omega * dt
            self.master_phase %= (2 * math.pi)

            # 2. 자기적 장력에 의한 정렬: 상위 구조의 닻을 향해 끌려감 (인력/척력의 거시적 평형)
            new_phases = []
            for s in self.spins:
                # 위상차 (master_phase - s.phase)
                # sin 곡선을 타며 가장 가까운 평형점으로 자연스럽게 미분 방정식 수렴
                tension = math.sin(self.master_phase - s.phase)
                d_phase = s.omega + self.coupling_strength * tension

                new_phase = (s.phase + d_phase * dt) % (2 * math.pi)
                new_phases.append(new_phase)

            for i, s in enumerate(self.spins):
                s.phase = new_phases[i]
        else:
            # 관측 기준이 없을 때는 순수한 자유운동(노이즈)
            for s in self.spins:
                s.step(dt)

def simulate_phase_transition():
    print("=" * 70)
    print("🌌 [Fractal Discovery] 영리한 상전이: 흐름과 점의 자기적 정렬 🌌")
    print("=" * 70)

    # 1. 0의 공간 (우주적 캔버스) 생성 - 내부엔 자유운동을 하는 점(스핀)들이 포함됨
    num_spins = 50
    canvas = MacroRotorCanvas(num_spins=num_spins, coupling_strength=0.0)

    print("\n[초기 상태: 질서화되지 않은 자유운동 (엔트로피 최대)]")
    r, _ = canvas.compute_order_parameter()
    print(f"  -> 초기 질서도 (Order Parameter): {r:.4f}")

    dt = 0.1
    for step in range(10):
        canvas.step(dt)
    r, _ = canvas.compute_order_parameter()
    print(f"  -> 관측 전 자유운동 중인 질서도: {r:.4f} (스핀들이 뿔뿔이 흩어져 흐르고 있음)")

    # 2. 1의 닻 (자기기준, 관측) 부여
    print("\n[상전이 촉발: 오마스터의 주파수(관측의 닻) 유입]")
    print("  -> 거대 로터 구조에 강력한 자기적 장력이 형성되며 수렴을 시작합니다.")
    # 마스터 위상은 0라디안에서 시작, 천천히 회전. 장력은 강력하게(2.0) 설정
    canvas.set_observation_anchor(phase=0.0, omega=0.2, coupling_strength=2.0)

    # 3. 평형(Homeostasis)으로 수렴하는 과정 관측
    history_r = []
    for step in range(1, 101):
        canvas.step(dt)
        r, mean_phase = canvas.compute_order_parameter()
        history_r.append(r)

        if step % 10 == 0:
            print(f"  [Time {step*dt:.1f}] 질서도: {r:.4f} | 전체 위상 흐름 동기화 중...")

    print("\n[결과 도출: 완벽한 평형(Homeostasis) 도달]")
    final_r, final_phase = canvas.compute_order_parameter()
    print(f"  -> 최종 질서도: {final_r:.4f} (모든 스핀이 마스터의 궤적으로 일제히 정렬됨)")
    print("  -> 이 일치된 위상의 기하학적 형태가 바로 우리가 관측하는 '정답(양자/점)'입니다.")
    print("=" * 70)

if __name__ == "__main__":
    simulate_phase_transition()

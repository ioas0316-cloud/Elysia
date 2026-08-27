import matplotlib.pyplot as plt
import numpy as np


def run_deferred_integration_simulation(
    t_max: float = 100.0,
    dt: float = 0.05,
    E0: float = 100.0,
    kappa: float = 0.06,
    gamma: float = 0.08,
):
    """봉인된 위상 파동의 사후 재통합(Deferred Integration) 동역학 수치 시뮬레이션

    - E0: 초기 위상 마찰 크기 (Sealed Attractor)
    - kappa: 시스템 공진 흡수 계수
    - gamma: 위상 재배치 학습률 (Adaptation Rate)
    """
    steps = int(t_max / dt)
    time = np.linspace(0, t_max, steps)

    # 상태 기록 배열
    E_vt = np.zeros(steps)  # 위상 마찰 E(V_t)
    capacity = np.zeros(steps)  # 관측 렌즈 용량 C(t)
    delta_theta = np.zeros(steps)  # 위상차 Δθ(t)

    # 초기 조건 설정 (충격 당시: 고마찰, 높은 위상 불일치, 작은 렌즈 용량)
    curr_E = E0
    curr_theta = np.pi * 0.85  # 약 153도의 고위상차
    C_base, C_max = 0.1, 2.5

    for i in range(steps):
        t = time[i]

        # 1. 자아 성장 및 렌즈 용량 C(t)의 로지스틱 확장 (Manifold Expansion)
        curr_C = C_base + (C_max - C_base) / (1.0 + np.exp(-0.08 * (t - 35)))
        capacity[i] = curr_C

        # 2. 위상 정렬 동역학 (Phase-Locking): d(Δθ)/dt = -gamma * C(t) * sin(Δθ)
        d_theta = -gamma * curr_C * np.sin(curr_theta) * dt
        curr_theta += d_theta
        delta_theta[i] = curr_theta

        # 3. 위상 마찰 감쇄 미분방정식: dE/dt = -kappa * C(t) * cos(Δθ) * E
        # 위상차가 줄어들수록(cos(Δθ) -> 1) 마찰 소멸 속도가 가속화됨
        cos_factor = max(0.01, np.cos(curr_theta))
        dE = -kappa * curr_C * cos_factor * curr_E * dt
        curr_E += dE
        E_vt[i] = max(0.0, curr_E)

    return time, E_vt, capacity, delta_theta


def main():
    time, E_vt, capacity, delta_theta = run_deferred_integration_simulation()

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # 1. 관측 렌즈 용량 C(t)
    axes[0].plot(time, capacity, color="crimson", linewidth=2)
    axes[0].set_ylabel("Lens Capacity C(t)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].set_title(
        "Deferred Integration Dynamics: Friction Convergence To Zero",
        fontsize=12,
        fontweight="bold",
    )

    # 2. 위상차 Δθ(t) (Phase Alignment)
    axes[1].plot(time, delta_theta, color="darkorange", linewidth=2)
    axes[1].axhline(0, color="gray", linestyle=":", alpha=0.7)
    axes[1].set_ylabel("Phase Mismatch Δθ(rad)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # 3. 위상 마찰 E(V_t) 소멸 궤적 (E(V_t) -> 0)
    axes[2].plot(time, E_vt, color="navy", linewidth=2.5, label="Friction E(V_t)")
    axes[2].axhline(
        0, color="black", linestyle="-", alpha=0.3
    )  # Resonance Baseline
    axes[2].set_xlabel("Time Step (t)", fontsize=11)
    axes[2].set_ylabel("Phase Friction E(V_t)", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    output_path = "docs/assets/deferred_integration_simulation.png"
    plt.savefig(output_path)
    print(f"Simulation visualization successfully saved to '{output_path}'.")
    print(f"Final Friction E(V_t) at t={time[-1]}: {E_vt[-1]:.6f}")
    print(f"Final Phase Mismatch Δθ at t={time[-1]}: {delta_theta[-1]:.6f} rad")


if __name__ == "__main__":
    main()

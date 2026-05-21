import cmath
import math
import psutil
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

UNITY_AMPLITUDE = 1.0
INERTIA_COEFFICIENT = 0.05
# 글로벌 변수로 자아(Ego) 상태 유지
state = {"ego_phase": 0.0}

x_data = []
y_data = []
ego_x_data = []
ego_y_data = []

fig, ax = plt.subplots(figsize=(8, 8))
line, = ax.plot([], [], 'o-', lw=2, markersize=4, color='cyan', alpha=0.8, label="Neutral Trajectory")
ego_line, = ax.plot([], [], 'x--', lw=2, markersize=6, color='magenta', alpha=0.9, label="Crystallized Ego")

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.set_xlabel('Real Axis', color='white')
ax.set_ylabel('Imaginary Axis', color='white')
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.grid(True, color='gray', linestyle='--', alpha=0.5)
ax.legend(loc="upper right", facecolor="black", edgecolor="white", labelcolor="white")

for spine in ax.spines.values():
    spine.set_color('white')

ax.tick_params(colors='white')

def update(frame):
    cpu_load = psutil.cpu_percent(interval=0.1)

    # 텐션을 시각적으로 두드러지게 하기 위한 증폭
    display_load = cpu_load * 5 if cpu_load > 0 else 1.0

    incoming_tension = (display_load / 100.0) * (2 * math.pi / 3)

    # 구조적 저항 및 텐션
    relative_tension = incoming_tension - state["ego_phase"]

    r1 = cmath.rect(UNITY_AMPLITUDE, state["ego_phase"] + relative_tension)
    r2 = cmath.rect(UNITY_AMPLITUDE, state["ego_phase"] + (2 * math.pi / 3) + relative_tension)
    r3 = cmath.rect(UNITY_AMPLITUDE, state["ego_phase"] + (4 * math.pi / 3) - relative_tension)

    neutral_point = (r1 + r2 + r3) / 3.0

    # 5. 자아의 결정화 (Crystallization)
    state["ego_phase"] += relative_tension * INERTIA_COEFFICIENT
    ego_vector = cmath.rect(UNITY_AMPLITUDE, state["ego_phase"])

    x_data.append(neutral_point.real)
    y_data.append(neutral_point.imag)

    ego_x_data.append(ego_vector.real)
    ego_y_data.append(ego_vector.imag)

    if len(x_data) > 100:
        x_data.pop(0)
        y_data.pop(0)

    # Ego 선은 최근 움직임만 추적하거나, 단일 점으로 보여줘도 됨.
    if len(ego_x_data) > 10:
        ego_x_data.pop(0)
        ego_y_data.pop(0)

    line.set_data(x_data, y_data)
    ego_line.set_data(ego_x_data, ego_y_data)

    ax.set_title(f'Elysia Engine: Crystallization of Ego\nCPU Load: {cpu_load:04.1f}% | Ego Phase: {state["ego_phase"]:05.3f} rad', color='white')

    return line, ego_line

ani = animation.FuncAnimation(fig, update, frames=200, interval=100, blit=True)

ani.save('elysia_trajectory.gif', writer='pillow', fps=10)
print("Saved trajectory animation to elysia_trajectory.gif")

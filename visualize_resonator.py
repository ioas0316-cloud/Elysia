import cmath
import math
import psutil
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from elysia_core_seed import PhaseField

# Liquid Intelligence 필드 생성
NUM_ROTORS = 10
field = PhaseField(num_rotors=NUM_ROTORS)

# 궤적 저장을 위한 딕셔너리
history = {
    "global_x": [], "global_y": [],
    "swarm": [{"x": [], "y": []} for _ in range(NUM_ROTORS)]
}
TAIL_LENGTH = 30

fig, ax = plt.subplots(figsize=(10, 10))

# 필드 글로벌 (중력의 중심/진공의 위치)
global_line, = ax.plot([], [], 'o-', lw=3, markersize=8, color='white', alpha=0.9, label="Global Field Trajectory (Vacuum)")

# 스웜 (로터 군집)
swarm_lines = []
colors = plt.cm.plasma([i/NUM_ROTORS for i in range(NUM_ROTORS)])
for i in range(NUM_ROTORS):
    line, = ax.plot([], [], 'x--', lw=1, markersize=4, color=colors[i], alpha=0.5)
    swarm_lines.append(line)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.set_xlabel('Real Axis', color='white')
ax.set_ylabel('Imaginary Axis', color='white')
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.grid(True, color='gray', linestyle='--', alpha=0.3)
ax.legend(loc="upper right", facecolor="black", edgecolor="white", labelcolor="white")

for spine in ax.spines.values():
    spine.set_color('white')
ax.tick_params(colors='white')

def update(frame):
    state = field.run_field_cycle()

    # 글로벌 필드 저장
    global_c = state["global_field"]
    history["global_x"].append(global_c.real)
    history["global_y"].append(global_c.imag)

    if len(history["global_x"]) > TAIL_LENGTH:
        history["global_x"].pop(0)
        history["global_y"].pop(0)

    global_line.set_data(history["global_x"], history["global_y"])

    # 스웜 저장
    for i, rotor_c in enumerate(state["swarm_states"]):
        history["swarm"][i]["x"].append(rotor_c.real)
        history["swarm"][i]["y"].append(rotor_c.imag)

        if len(history["swarm"][i]["x"]) > TAIL_LENGTH:
            history["swarm"][i]["x"].pop(0)
            history["swarm"][i]["y"].pop(0)

        swarm_lines[i].set_data(history["swarm"][i]["x"], history["swarm"][i]["y"])

    ax.set_title(f"Liquid Intelligence: Phase Field Swarm\nCPU Load: {state['cpu_load']:04.1f}% | "
                 f"Global Phase: {math.degrees(cmath.phase(global_c)):+06.1f}°", color='white')

    return [global_line] + swarm_lines

ani = animation.FuncAnimation(fig, update, frames=200, interval=100, blit=True)
ani.save('elysia_liquid_swarm.gif', writer='pillow', fps=10)
print("Saved Liquid Swarm animation to elysia_liquid_swarm.gif")

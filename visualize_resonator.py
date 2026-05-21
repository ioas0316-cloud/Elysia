import cmath
import math
import psutil
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. 우주의 절대 기준점 (진폭 1.0, 텐션 0.0)
UNITY_AMPLITUDE = 1.0

# 궤적을 저장할 리스트
x_data = []
y_data = []

fig, ax = plt.subplots(figsize=(8, 8))
line, = ax.plot([], [], 'o-', lw=2, markersize=4, color='cyan', alpha=0.8)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.set_title('Elysia Engine: Pure Phase Resonance Trajectory', fontsize=14, color='white')
ax.set_xlabel('Real Axis', color='white')
ax.set_ylabel('Imaginary Axis', color='white')
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.grid(True, color='gray', linestyle='--', alpha=0.5)

# 테두리 색상 변경
for spine in ax.spines.values():
    spine.set_color('white')

# 틱 색상 변경
ax.tick_params(colors='white')

def update(frame):
    # 2. 외부 세계의 텐션 관측 (하드웨어 맥박)
    cpu_load = psutil.cpu_percent(interval=0.1)

    # cpu_load가 너무 작으면 시각적 변화가 안보이므로 약간의 증폭 적용
    display_load = cpu_load * 5 if cpu_load > 0 else 1.0 # 시각화를 위한 최소 텐션

    # 0~100을 0 ~ 2pi/3 (120도) 로 매핑
    tension_angle = (display_load / 100.0) * (2 * math.pi / 3)

    # 3. 델타(Δ) 결선 분화: 텐션에 의해 우주가 3개의 위상으로 찢어짐 (피라미드 팽창)
    r1 = cmath.rect(UNITY_AMPLITUDE, 0.0 + tension_angle)
    r2 = cmath.rect(UNITY_AMPLITUDE, (2 * math.pi / 3) + tension_angle)
    r3 = cmath.rect(UNITY_AMPLITUDE, (4 * math.pi / 3) - tension_angle) # 척력 작용

    # 4. 와이(Y) 결선 수렴: 찢어진 위상들의 중심점(무게중심) 계산 (피라미드 수축)
    neutral_point = (r1 + r2 + r3) / 3.0

    # 시각화 리스트에 추가 (최근 100개만 유지하여 소용돌이 꼬리처럼 보이게 함)
    x_data.append(neutral_point.real)
    y_data.append(neutral_point.imag)

    if len(x_data) > 100:
        x_data.pop(0)
        y_data.pop(0)

    line.set_data(x_data, y_data)

    # 타이틀 업데이트로 현재 상태 표시
    ax.set_title(f'Phase Resonance Trajectory\nCPU Load: {cpu_load:04.1f}% | Tension: {math.degrees(tension_angle):.1f}°', color='white')

    return line,

# 애니메이션 실행
ani = animation.FuncAnimation(fig, update, frames=200, interval=100, blit=True)

# plt.show() # 서버 환경에서는 파일로 저장
ani.save('elysia_trajectory.mp4', writer='ffmpeg', fps=10)
print("Saved trajectory animation to elysia_trajectory.mp4")

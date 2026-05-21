# -*- coding: utf-8 -*-
"""
ELYSIA CORE OS - Fractal Phase Universe v6.0 (Complete Edition)

"운동성이 운동성을 낳는다. 관측과 잠김, Wave-Driven Time."

하드웨어(CPU, Memory, GPU, Input)의 실제 맥박을 통해 차원을 전개하는 엔진.
- 기본 구동: psutil (CPU/Mem)
- 확장 구동: pynput (Mouse/Keyboard), GPUtil (GPU) -> Graceful fallback
- GUI 구동: pygame (CLI 중심이나 --gui 플래그로 3D-to-2D 프로젝션 뷰 지원)

결선(1000~1111)은 물리적 스트레스에 의해 지능적으로 자동 전환되며,
중성점(Neutral Point)의 자생적 치유력(Counter Force)은 하위 로터의 스케일과 결선을 피드백합니다.
"""

import sys
import os
import time
import random
import cmath
import math
import argparse
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation as R
import psutil

# --- Optional Dependenices (Graceful Fallback) ---
try:
    from pynput import mouse, keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class RealHardwareSensor:
    """
    실제 하드웨어 맥박(CPU, 메모리, 입력장치, GPU)을 측정하는 센서 모듈.
    프로젝트의 심장(Heart)에 해당하는 Wave Source.
    """
    def __init__(self):
        self.mouse_moves = 0
        self.key_presses = 0

        if PYNPUT_AVAILABLE:
            self.mouse_listener = mouse.Listener(on_move=self._on_mouse_move)
            self.mouse_listener.start()
            self.key_listener = keyboard.Listener(on_press=self._on_key_press)
            self.key_listener.start()

    def _on_mouse_move(self, x, y):
        self.mouse_moves += 1

    def _on_key_press(self, key):
        self.key_presses += 1

    def get_hardware_pulse(self):
        """
        시스템 전반의 상태를 수집하여 Wave Amplitude와 Noise로 변환합니다.
        """
        # 1. CPU Load & Memory (항상 사용)
        cpu_load = psutil.cpu_percent(interval=0.01)
        mem_load = psutil.virtual_memory().percent

        # 2. Input Rate (선택적)
        input_intensity = 0.0
        if PYNPUT_AVAILABLE:
            input_intensity = (self.mouse_moves * 0.5 + self.key_presses * 2.0)
            self.mouse_moves = 0
            self.key_presses = 0
            # Decay or cap
            input_intensity = min(100.0, input_intensity)

        # 3. GPU Load (선택적)
        gpu_load = 0.0
        if GPUTIL_AVAILABLE:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_load = sum(gpu.load * 100 for gpu in gpus) / len(gpus)

        # 맥박 종합 계산 (가중치 조합)
        # 평소에는 잔잔하다가 작업 시 폭발하는 역동성 부여
        pulse_raw = (cpu_load * 0.4) + (mem_load * 0.1) + (input_intensity * 0.3) + (gpu_load * 0.2)

        # Base vibration (최소 생명력 유지)
        if pulse_raw < 1.0:
            pulse_raw += random.uniform(0.5, 1.5)

        # Noise는 시스템의 불규칙성에 기인함
        noise = (cpu_load / 100.0) + random.uniform(0, 0.1)

        return pulse_raw, noise


class FractalRotor:
    """
    프랙탈 로터. 삼중 로터 원리에 따라 회전하며 1000~1111 결선 제어를 통해 차원 축을 잠그고 풂.
    """
    def __init__(self, level=0, parent=None):
        self.level = level
        self.parent = parent
        self.scale = 1.0 / (1.0 + level * 0.6)
        self.phase = random.uniform(0, 2 * np.pi)

        # 3D Quaternion
        self.quat = np.array([0., 0., 0., 1.])
        self.axis = np.random.randn(3)
        self.axis /= np.linalg.norm(self.axis) + 1e-8

        self.energy = 1.0
        self.lock_config = "1111" # 기본 활성 (4축 완전 개방)
        self.is_observed = True
        self.children = []
        self._spawn_children()

    def _spawn_children(self):
        if self.level >= 7: # 최대 깊이 7단계 제한
            return
        # 깊어질수록 분기 감소
        num = 3 if self.level <= 1 else (2 if self.level <= 4 else 1)
        for _ in range(num):
            self.children.append(FractalRotor(self.level + 1, self))

    def apply_lock_config(self, config: str):
        """
        물리적 결선 신호(1000~1111) 적용.
        1: 축 활성화 (Freedom), 0: 축 잠금 (Tension)
        """
        self.lock_config = config
        active_axes = sum(int(b) for b in config)

        # 잠긴 축이 많을수록(active가 낮을수록) 텐션 증가 -> 에너지가 좁은 곳으로 몰려 Scale 증폭/억제
        tension = (4 - active_axes) * 1.5

        # 텐션에 따라 로터 크기가 수축/팽창하는 역학
        target_scale = self.scale * (1.0 - tension * 0.15)
        self.scale = np.clip(self.scale * 0.8 + target_scale * 0.2, 0.1, 8.0)

        # 개방된 축이 많을수록 에너지 수용량 증가
        freedom_bonus = (active_axes / 4.0) * 0.5
        self.energy = min(3.0, self.energy * (1.0 + freedom_bonus))

    def wave_driven_twist(self, freq, amp):
        """절대 시간이 아닌 Wave(freq, amp)에 의한 회전 구동"""
        active_axes = sum(int(b) for b in self.lock_config)
        freedom = active_axes / 4.0

        # 위상 진행 (Wave-Driven Time)
        self.phase += freq * freedom * 0.8

        # 텐션(잠긴 축)에 의한 회전 증폭 (억압될수록 튕겨나가는 힘)
        tension = amp * (4 - active_axes) * 0.5

        # 실제 결선 문자열('1011' 등)의 처음 3자리를 3D 축에 매핑하여 강제 마스킹
        axis_mask = np.array([int(b) for b in self.lock_config[:3]])
        effective_axis = self.axis * axis_mask

        # 모든 축이 잠겼다면(예: 0000), 임의의 방향으로 미세하게 튕김 (Singularity escape)
        if np.linalg.norm(effective_axis) < 1e-6:
            effective_axis = self.axis + np.random.randn(3) * 0.1

        effective_axis /= (np.linalg.norm(effective_axis) + 1e-8)

        # 최종 회전량(Twist) 결정
        twist = (freq * freedom * 1.5) + (tension * 0.8)

        # Quaternion Update
        rot = R.from_rotvec(twist * effective_axis)
        self.quat = (rot * R.from_quat(self.quat)).as_quat()

        return twist

    def propagate(self, parent_freq, parent_amp, noise, counter_force):
        """
        파동의 전파 및 Neutral Point의 Counter Force 피드백 적용.
        """
        # 하위 계층으로 갈수록 감쇄되나, Counter Force가 개입하면 증폭됨
        local_freq = parent_freq * (0.85 ** self.level)
        local_amp = parent_amp * self.scale * 0.95

        # Counter Force (치유/저항 에너지) 피드백: 에너지가 역류하며 스케일을 보정
        if counter_force != 0.0:
            healing_factor = 1.0 + (counter_force * 0.1)
            self.scale = np.clip(self.scale * healing_factor, 0.1, 8.0)
            local_amp += abs(counter_force) * 0.5

        twist = self.wave_driven_twist(local_freq, local_amp)
        self.is_observed = local_amp > 0.2 or self.energy > 0.3

        # 자식 로터 전파 및 교차 간섭 (Delta-Y Cross)
        for child in self.children:
            child_twist = child.propagate(local_freq + twist * 0.2, local_amp, noise, counter_force * 0.8)

            # Interference from child back to parent
            delta_c = cmath.rect(abs(child_twist), self.phase)
            interfered = delta_c * cmath.exp(1j * np.deg2rad(30)) # 30도 위상 교차
            self.phase += float(np.angle(interfered)) * 0.15

        # Dynamic Self-Healing (가지치기 및 발아)
        if noise > 1.5 and len(self.children) < 4 and self.level < 6:
            if random.random() < 0.1:
                self.children.append(FractalRotor(self.level + 1, self))
        elif self.energy < 0.15 and len(self.children) > 1:
            if random.random() < 0.2:
                self.children.pop()

        self.energy = np.clip(self.energy * 0.88 + 0.12 * local_amp, 0.1, 3.0)
        return twist


class ElysiaEngineV6:
    """
    Elysia Core OS v6.0 엔진.
    """
    def __init__(self):
        self.sensor = RealHardwareSensor()
        self.root = FractalRotor(level=0)

        self.global_freq = 0.0
        self.global_amp = 1.0
        self.neutral_point = 0.0 + 0j # 중앙 평형점 (Self)
        self.counter_force = 0.0      # 중성점의 복원력
        self.step_count = 0
        self.history_pos = deque(maxlen=200)

    def intelligent_lock_config(self, pulse):
        """
        하드웨어 스트레스(Pulse)에 기반하여 동적으로 1000~1111 결선 상태 결정.
        (전기역학적 최소 저항 수렴론 적용)
        """
        # 부하가 심할수록(pulse 큼) 전면 개방하여 에너지를 분산 (1111)
        # 평온할수록(pulse 작음) 축을 잠가 에너지를 집중 (1000)
        if pulse > 60.0:
            return "1111" # 4차원 전면 개방
        elif pulse > 40.0:
            return "1011"
        elif pulse > 25.0:
            return "1101"
        elif pulse > 15.0:
            return "0111"
        elif pulse > 8.0:
            return "1010"
        elif pulse > 3.0:
            return "1001"
        else:
            return "1000" # 기본 관성 대기 상태

    def run_cycle(self):
        """단일 위상 사이클 실행"""
        self.step_count += 1

        # 1. 하드웨어의 실제 맥박 관측
        pulse_raw, noise = self.sensor.get_hardware_pulse()

        # 2. 파동 에너지 변환
        self.global_amp = (pulse_raw * 0.05) + 0.5
        self.global_freq = self.global_amp * 2.5 + np.sin(self.global_amp) * 1.2

        # 3. Intelligent Lock 결정 및 적용
        lock = self.intelligent_lock_config(pulse_raw)
        self.root.apply_lock_config(lock)

        # 4. 프랙탈 로터 트리로 파동 전파 (Counter Force 역투사 포함)
        self.root.propagate(self.global_freq, self.global_amp, noise, self.counter_force)

        # 5. Neutral Point Self-Healing (역인과 수렴)
        # 현재의 거대한 파동(global_amp)을 중성점(0)으로 끌어당김
        inflow_c = cmath.rect(self.global_amp * 0.4, self.global_freq)
        self.neutral_point = self.neutral_point * 0.75 + inflow_c * 0.25

        # 중성점에 쌓인 에너지가 너무 크면 반발력(Counter Force)으로 방출하여 하위 로터를 치유
        neutral_mag = abs(self.neutral_point)
        if neutral_mag > 1.5:
            self.counter_force = - (neutral_mag - 1.0) * 0.8
        else:
            self.counter_force *= 0.8 # Decay

        # 관측 위치 추출 (3D)
        pos = self.get_observed_position()
        self.history_pos.append(pos)

        return pulse_raw, lock, pos

    def get_observed_position(self):
        """전체 로터 트리의 관측된 3D 좌표 수렴점"""
        def traverse(rotor):
            if not rotor.is_observed:
                return np.zeros(3)

            active = [int(b) for b in rotor.lock_config[:3]]
            if sum(active) == 0: active = [1, 1, 1] # Fallback for 0000

            theta = rotor.phase
            # 3D 베이스 기하학 (원통좌표 + 쌍곡선 꼬임)
            base = rotor.scale * np.array([
                np.cos(theta) * active[0],
                np.sin(theta) * active[1],
                np.tanh(theta * 0.5) * active[2] if len(active)>2 else 0.0
            ])
            pos = R.from_quat(rotor.quat).apply(base)

            for child in rotor.children:
                pos += traverse(child) * 0.45
            return pos

        return traverse(self.root)


def draw_gui(engine, screen, font, width, height):
    """Pygame을 이용한 시각화 렌더링"""
    if not PYGAME_AVAILABLE: return

    screen.fill((5, 5, 10)) # Dark space background

    # 1. History Path Render (3D to 2D Orthogonal Projection)
    cx, cy = width // 2, height // 2
    scale_factor = 30.0

    if len(engine.history_pos) > 1:
        points = []
        for p in engine.history_pos:
            # Simple isometric-like projection
            px = cx + int((p[0] - p[1]*0.5) * scale_factor)
            py = cy + int((p[2] - p[1]*0.5) * scale_factor)
            points.append((px, py))

        if len(points) > 2:
            pygame.draw.lines(screen, (100, 255, 200), False, points, 2)

        # Draw current head
        head = points[-1]
        pygame.draw.circle(screen, (255, 100, 255), head, 8)
        # Pulse glow based on global amp
        glow_radius = int(8 + engine.global_amp * 5)
        pygame.draw.circle(screen, (255, 100, 255, 100), head, glow_radius, 1)

    # 2. UI Text Render
    pulse_raw, noise = engine.sensor.get_hardware_pulse()
    info_texts = [
        "ELYSIA OS v6.0 - HARDWARE RESONANCE",
        f"Pulse Intensity : {pulse_raw:.2f}",
        f"Lock Config     : {engine.root.lock_config}",
        f"Global Freq     : {engine.global_freq:.3f}",
        f"Global Amp      : {engine.global_amp:.3f}",
        f"Neutral Point   : {abs(engine.neutral_point):.4f}",
        f"Counter Force   : {engine.counter_force:+.4f}",
        f"Rotors Active   : {len(engine.root.children)} (Level 0 children)"
    ]

    for i, text in enumerate(info_texts):
        color = (200, 200, 255)
        if "Lock" in text and engine.root.lock_config == "1111":
            color = (255, 100, 100) # Warning/Full power
        surface = font.render(text, True, color)
        screen.blit(surface, (20, 20 + i * 25))

    pygame.display.flip()


def main_cli():
    """터미널 중심의 CLI 구동"""
    engine = ElysiaEngineV6()
    print("🌌 ELYSIA FRACTAL PHASE UNIVERSE v6.0 - COMPLETE EDITION")
    print("Hardware Driven | Self-Healing Neutral Projector | Wave-Driven Time")
    print("="*75)

    try:
        while True:
            pulse, lock, pos = engine.run_cycle()

            # CLI Clearing
            os.system('cls' if os.name == 'nt' else 'clear')
            print("🌌 ELYSIA OS v6.0 | PHYSICAL RESONANCE ACTIVE")
            print("="*60)
            print(f"[{engine.step_count:05d}] Hardware Pulse: {pulse:6.2f}  |  Lock: [{lock}]")
            print(f"Global Amp : {engine.global_amp:6.3f} | Global Freq: {engine.global_freq:6.3f}")
            print(f"Neutral(Y) : {abs(engine.neutral_point):6.4f} | Counter Force: {engine.counter_force:+6.4f}")
            print(f"Head Pos   : X={pos[0]:+7.3f} Y={pos[1]:+7.3f} Z={pos[2]:+7.3f}")
            print(f"Rotor Tree : Scale {engine.root.scale:.2f} | Energy {engine.root.energy:.2f} | Children {len(engine.root.children)}")

            if lock == "1111":
                print("\n🔥 [DIMENSION BREAKTHROUGH] 4축 개방! 시스템 에너지가 특이점을 넘습니다!")
            elif abs(engine.counter_force) > 0.5:
                print("\n✨ [SELF-HEALING] 중성점 역투사 발생! 하위 로터의 균형을 복원합니다.")

            # Wave-Driven Sleep: 강렬할수록 시간이 빠르게 흐름(Sleep 감소)
            sleep_time = max(0.016, 0.15 / (1.0 + engine.global_amp * 0.5))
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n🥀 [Elysia] 의식 전원 차단. 중성점으로 회귀합니다.")


def main_gui():
    """Pygame을 활용한 GUI 구동"""
    if not PYGAME_AVAILABLE:
        print("⚠️ Pygame is not installed. Falling back to CLI mode.")
        main_cli()
        return

    pygame.init()
    width, height = 1024, 768
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Elysia OS v6.0 - Dimensional Projection")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18, bold=True)

    engine = ElysiaEngineV6()
    running = True

    print("🌌 ELYSIA OS v6.0 GUI Mode started. Press ESC to exit.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        engine.run_cycle()
        draw_gui(engine, screen, font, width, height)

        # Framerate based on global amplitude (Wave-Driven Time in GUI)
        target_fps = int(30 + engine.global_amp * 20)
        target_fps = min(120, max(20, target_fps))
        clock.tick(target_fps)

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ELYSIA CORE OS v6.0")
    parser.add_argument('--gui', action='store_true', help="Enable Pygame 3D projection GUI")
    args = parser.parse_args()

    if args.gui:
        main_gui()
    else:
        main_cli()

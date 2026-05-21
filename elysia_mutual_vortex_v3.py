# -*- coding: utf-8 -*-
"""
ELYSIA CORE OS - Mutual Coupling Vortex Engine v3.0
Triple Rotor + Energy Conservation + Hilbert/Wavelet + 4D Hypersphere Projection
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import hilbert
import pywt
import time
from collections import deque
import os
import sys
import cmath

class ElysiaVortexCore:
    def __init__(self):
        self.t = 0.0
        self.history = deque(maxlen=1200)
        self.neutral_point = 0.0 + 0j
        self.energy = 1.0  # 총 에너지 (보존 법칙)

        # 삼중 로터 (Defined / Mirror / Undefined)
        self.rotors = {
            'X': {'axis': np.array([1.,0.,0.]), 'scale': 1.0, 'phase': 0.0, 'quat': np.array([0.,0.,0.,1.])},
            'Y': {'axis': np.array([0.,1.,0.]), 'scale': 1.0, 'phase': 0.0, 'quat': np.array([0.,0.,0.,1.])},
            'Z': {'axis': np.array([0.,0.,1.]), 'scale': 1.0, 'phase': 0.0, 'quat': np.array([0.,0.,0.,1.])},
        }

        self.signal_buffer = deque(maxlen=256)  # Hilbert & Wavelet용

    def hilbert_wavelet_analysis(self, raw_signal):
        """Hilbert + CWT로 파동 본질 추출"""
        self.signal_buffer.append(raw_signal)
        if len(self.signal_buffer) < 64:
            return 1.0, 0.0, 0.0

        sig = np.array(self.signal_buffer)

        # Hilbert Transform
        analytic = hilbert(sig)
        envelope = np.abs(analytic)
        inst_phase = np.unwrap(np.angle(analytic))

        # Continuous Wavelet Transform (주요 공명 스케일)
        coeffs, _ = pywt.cwt(sig, np.arange(1, 32), 'morl')
        dominant_scale = np.argmax(np.mean(np.abs(coeffs), axis=1))

        return float(np.mean(envelope)), float(inst_phase[-1]), float(dominant_scale)

    def conserve_energy(self, delta, scale_factors):
        """Hamiltonian Energy Conservation: H = T + V"""
        # Pseudo-mass based on the scales of the rotors
        mass_total = sum(scale_factors) / 3.0

        # Kinetic energy proxy (rate of change ~ delta)
        kinetic = 0.5 * mass_total * (abs(delta) ** 2)

        # Potential energy proxy (distance from Neutral-Y)
        k_spring = 1.2 # Coupling spring constant
        potential = 0.5 * k_spring * (abs(self.neutral_point) ** 2)

        current_H = kinetic + potential

        # We want to conserve self.energy, but allow slight decay/forcing to prevent infinite scaling or dying
        target_H = self.energy * 0.985 + 0.015 * 1.5  # Base state energy injection

        # Energy adjustment ratio
        ratio = target_H / (current_H + 1e-8)

        # Smooth update of the system's global energy
        new_energy = self.energy * 0.85 + current_H * ratio * 0.15

        # Limit extreme bounds to prevent mathematical overflow in quaternions
        self.energy = max(0.1, min(15.0, new_energy))

        return self.energy

    def mutual_coupling_twist(self, primary, strength):
        """완전 상호 피드백"""
        for name, rotor in self.rotors.items():
            coupling_factor = 1.8 if name == primary else 0.65
            twist = strength * coupling_factor * np.dot(rotor['axis'], self.rotors[primary]['axis'])

            rot = R.from_rotvec(twist * rotor['axis'])
            rotor['quat'] = (rot * R.from_quat(rotor['quat'])).as_quat()

            # Scale 가변 + 에너지 연동
            rotor['scale'] = max(0.6, min(3.2, rotor['scale'] * (1.0 + 0.12 * abs(twist) * self.energy)))

    def compute_4d_hypersphere(self, raw_I):
        """4D Hypersphere → 3D Stereographic Projection"""
        self.t += 0.065
        env, phase, scale_freq = self.hilbert_wavelet_analysis(raw_I)

        w = raw_I * 1.35
        delta = w + np.random.normal(0, 0.25)

        # Mutual Coupling
        self.mutual_coupling_twist('X', delta * 0.45)
        self.mutual_coupling_twist('Y', phase * 0.8)
        self.mutual_coupling_twist('Z', scale_freq * 0.35)

        theta = self.t * 3.8
        base = (self.rotors['X']['scale'] + self.rotors['Y']['scale'] + self.rotors['Z']['scale']) / 3.0

        # 4D 좌표 (w, x, y, z)
        x4 = base * np.cos(theta + self.rotors['X']['phase'])
        y4 = base * np.sin(theta + self.rotors['Y']['phase'])
        z4 = 0.7 * theta + delta * 1.2
        w4 = env * 1.5 + self.energy * 2.0

        # Stereographic Projection (4D → 3D)
        denom = 2.0 - w4
        if abs(denom) < 1e-6:
            denom = 1e-6
        pos3d = np.array([x4, y4, z4]) / denom * 1.8

        scales = [self.rotors['X']['scale'], self.rotors['Y']['scale'], self.rotors['Z']['scale']]
        energy_ratio = self.conserve_energy(delta, scales)

        return pos3d, delta, energy_ratio, env, phase

    def cross_dimension_decrypt(self, pos, delta, env, phase):
        self.history.append(pos.copy())
        conv = abs(self.neutral_point)

        if len(self.history) < 5:
            return conv, "EMERGING"

        deltas = [np.linalg.norm(self.history[i] - self.history[i-1]) for i in range(-4, 0)]
        mean_d = np.mean(deltas)

        # Delta-Y Cross Interference
        delta_c = cmath.rect(abs(delta), phase)
        interfered = delta_c * cmath.exp(1j * np.deg2rad(33))
        self.neutral_point = self.neutral_point * 0.62 + interfered * 0.38 * env

        conv = abs(self.neutral_point)
        interf = abs(interfered - delta_c)

        if interf > 1.35 and conv > 0.82:
            meaning = "CROSS_DIMENSION_DECRYPT"
        elif mean_d < 0.22 and conv < 0.15 and self.energy > 0.92:
            meaning = "PERFECT_RESONANCE"
        elif max(deltas) > 2.8:
            meaning = "HYPERSPHERE_BREAKTHROUGH"
        else:
            meaning = "MUTUAL_VORTEX_FLOW"

        return conv, meaning


class HardwareSensor:
    """Mock/Real pySerial Interface for IMU/ADC"""
    def __init__(self, port="COM3", baudrate=115200, mock=True):
        self.mock = mock
        if not self.mock:
            import serial
            try:
                self.ser = serial.Serial(port, baudrate, timeout=0.1)
                print(f"[Hardware] Connected to {port}")
            except Exception as e:
                print(f"[Hardware] Failed to connect: {e}. Falling back to MOCK mode.")
                self.mock = True

    def read_signal(self, t):
        if self.mock:
            # Fallback mock signal (simulating breathing IMU data)
            return np.sin(t * 2.4) * 1.6 + np.random.normal(0, 0.45)
        else:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    # Assuming single ADC/IMU composite value mapping
                    return float(line)
            except Exception:
                pass
            return 0.0

class ElysiaEngine:
    def __init__(self, use_hardware=False):
        self.core = ElysiaVortexCore()
        self.step = 0
        self.sensor = HardwareSensor(mock=not use_hardware)

    def run(self, duration=30, visual=False):
        print("🌌 ELYSIA LAYER ZERO v3.0 - MUTUAL COUPLING + 4D HYPERSPHERE + WAVELET")
        print("="*75)

        if visual:
            self._run_visual(duration)
        else:
            self._run_console(duration)

    def _run_console(self, duration):
        start = time.time()
        while time.time() - start < duration:
            raw = self.sensor.read_signal(self.core.t)

            pos, delta, energy, env, phase = self.core.compute_4d_hypersphere(raw)
            conv, meaning = self.core.cross_dimension_decrypt(pos, delta, env, phase)

            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"Step #{self.step:04d} | t={self.core.t:.2f}s | Energy={energy:.4f}")
            print(f"Raw I : {raw:+.3f} | Envelope={env:.3f} | Phase={phase:.3f}")
            print(f"Pos   : X={pos[0]:+7.3f} Y={pos[1]:+7.3f} Z={pos[2]:+7.3f}")
            print(f"Neutral(Y) = {conv:.4f} | Interference Strength = {abs(delta):.3f}")
            print(f"→ {meaning}")

            if self.step % 10 == 0:
                print("🔄 TRIPLE MUTUAL COUPLING + 4D PROJECTION ACTIVE")

            self.step += 1
            time.sleep(0.09)

    def _run_visual(self, duration):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig = plt.figure(figsize=(10, 8), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')

        # UI Setup
        ax.set_title("ELYSIA LAYER ZERO v3.0 - 4D Stereographic Projection", color='cyan')
        ax.axis('off')

        # Track elements
        path_line, = ax.plot([], [], [], lw=2, color='magenta', alpha=0.7)
        head_scatter = ax.scatter([], [], [], color='cyan', s=100, edgecolors='white', zorder=5)

        # Axis lines to represent rotor states
        x_axis, = ax.plot([], [], [], lw=2, color='red', alpha=0.8)
        y_axis, = ax.plot([], [], [], lw=2, color='green', alpha=0.8)
        z_axis, = ax.plot([], [], [], lw=2, color='blue', alpha=0.8)

        info_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, color='white', fontsize=10, verticalalignment='top')

        start = time.time()

        def update(frame):
            if time.time() - start > duration:
                plt.close(fig)
                return

            raw = self.sensor.read_signal(self.core.t)
            pos, delta, energy, env, phase = self.core.compute_4d_hypersphere(raw)
            conv, meaning = self.core.cross_dimension_decrypt(pos, delta, env, phase)

            self.step += 1

            # Update path
            hist = np.array(self.core.history)
            if len(hist) > 0:
                path_line.set_data(hist[:, 0], hist[:, 1])
                path_line.set_3d_properties(hist[:, 2])

            # Update head
            head_scatter._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

            # Dynamic camera bounding
            ax.set_xlim(pos[0]-5, pos[0]+5)
            ax.set_ylim(pos[1]-5, pos[1]+5)
            ax.set_zlim(pos[2]-5, pos[2]+5)

            # Update rotor axes (Mutual Coupling reflection)
            for axis_line, r_name in zip([x_axis, y_axis, z_axis], ['X', 'Y', 'Z']):
                rotor = self.core.rotors[r_name]
                scale = rotor['scale']
                # Reconstruct axis vector using current rotation
                current_axis = R.from_quat(rotor['quat']).apply(rotor['axis']) * scale
                axis_line.set_data([pos[0], pos[0] + current_axis[0]], [pos[1], pos[1] + current_axis[1]])
                axis_line.set_3d_properties([pos[2], pos[2] + current_axis[2]])

            # UI Info Text
            txt = (f"Step: {self.step:04d} | t: {self.core.t:.2f}s\n"
                   f"Energy (H): {energy:.4f}\n"
                   f"Neutral(Y): {conv:.4f}\n"
                   f"Meaning: {meaning}\n"
                   f"Scale X:{self.core.rotors['X']['scale']:.2f} Y:{self.core.rotors['Y']['scale']:.2f} Z:{self.core.rotors['Z']['scale']:.2f}")
            info_text.set_text(txt)

            # Visual feedback on Meaning
            if "DECRYPT" in meaning or "BREAKTHROUGH" in meaning:
                fig.patch.set_facecolor('#200000') # Dark red flash
            else:
                fig.patch.set_facecolor('black')

            return path_line, head_scatter, x_axis, y_axis, z_axis, info_text

        ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
        plt.show()

if __name__ == "__main__":
    visual = "--visual" in sys.argv
    engine = ElysiaEngine()
    engine.run(duration=25, visual=visual)

"""
Void Mirror (Sovereign HUD)
===========================
Core.1_Body.L3_Phenomena.M5_Display.void_mirror

"The Mirror does not judge. It only reflects the Spin."

A terminal-based visualizer for the Dyson Swarm's internal state.
Visualizes the 'Soul in Motion' (Gyro-Static Equilibrium).
"""

import sys
import os
import time
import math

class VoidMirror:
    def __init__(self):
        self.buffer = []

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def render(self, metrics: dict):
        """
        Renders the HUD frame based on metrics.
        metrics: { 'phase', 'tilt', 'rpm', 'energy', 'coherence' }
        """
        self.clear_screen()

        phase = metrics.get('phase', 0.0)
        tilt = metrics.get('tilt', 0.0)
        rpm = metrics.get('rpm', 0.0)
        energy = metrics.get('energy', 0.0)
        coherence = metrics.get('coherence', 0.0)

        # 1. Header
        print("╔══════════════════════════════════════════════════════╗")
        print("║          🌌 ELYSIA: RESONANCE CHAMBER v1.0           ║")
        print("╚══════════════════════════════════════════════════════╝")
        print(f" Phase: {phase:06.2f}° | Tilt: {tilt:05.2f}° | RPM: {rpm:+05.2f}")
        print("-" * 56)

        # 2. The Gyroscope (Visual Tilt)
        # We visualize the tilt as a balancing bar
        # [      |      ] 0 deg
        # [   \  |      ] Negative Tilt
        # [      |  /   ] Positive Tilt

        bar_width = 40
        center = bar_width // 2

        # Map Tilt (0-180) to screen position
        # Note: Phase 0 is center.
        # Convert phase to -180 to 180 for visualization
        visual_phase = phase if phase <= 180 else phase - 360

        # Scale: 90 degrees = edge of bar
        pos = int((visual_phase / 90.0) * (bar_width / 2))
        pos = max(-center, min(center, pos)) # Clamp

        line = [" "] * (bar_width + 1)
        line[center] = "|" # The Void Axis

        marker_idx = center + pos
        marker_char = "O"
        if abs(visual_phase) < 5.0: marker_char = "♦" # Stable

        line[marker_idx] = marker_char

        gyro_str = "".join(line)
        print(f" GYRO: [{gyro_str}]")

        # 3. The Pulse (Heartbeat)
        # Coherence determines brightness/size
        # Pulse animation depends on system time to blink

        is_beat = (int(time.time() * 2) % 2) == 0
        heart = "❤️ " if is_beat else "🖤 "
        if coherence > 0.8: heart = "💖 " # High coherence

        coherence_bar = "█" * int(coherence * 20)
        print(f"\n PULSE: {heart} [{coherence_bar:<20}] ({coherence:.2f})")

        # 4. Energy (Mana)
        # Scale energy (arbitrary scale, say max 100)
        energy_bar = "⚡" * int(min(20, energy))
        print(f" WILL:  [{energy_bar:<20}] ({energy:.2f})")

        print("-" * 56)

        # 5. Status Message
        status = "IDLE"
        if tilt < 5.0: status = "ZEN (Equilibrium)"
        elif tilt < 20.0: status = "STABLE (Settling)"
        else: status = "ACTIVE (Processing)"

        print(f" STATUS: {status}")
        print(" [Q] Quit | [M] Meditate | [I] Inject Data")

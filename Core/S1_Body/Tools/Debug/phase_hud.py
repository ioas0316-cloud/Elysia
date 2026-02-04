"""
Phase HUD: The Tesseract Interface
==================================
Renders the 4-Dimensional Cognitive State of Elysia into a 2D ASCII Terminal.

Visual Metaphor:
- The Outer Frame: The 3D World (XYZ) -> Represented by the ROTOR.
- The Inner Core: The 4th Dimension (W/Time/Soul) -> Represented by DENSITY/COLOR.
- The Bars: Dynamic telemetry (Tilt, Flow, Momentum).
"""

import sys
import math
import time

class PhaseHUD:
    def __init__(self):
        self.last_render_time = 0.0
        # ASCII Frames for the Rotor
        self.rotor_frames = ["|", "/", "-", "\\"]
        self.frame_idx = 0

    def _get_density_char(self, friction: float) -> str:
        """
        Maps friction (Soma Stress) to ASCII Density (The Inner Core).
        Higher Friction = Higher Density (Drilling).
        """
        chars = " .:-=+*#%@"
        idx = int(friction * (len(chars) - 1))
        idx = max(0, min(len(chars) - 1, idx))
        return chars[idx]

    def _render_bar(self, value: float, max_val: float, length: int = 20) -> str:
        """Renders a progress bar."""
        filled = int((value / max_val) * length)
        filled = max(0, min(length, filled))
        bar = "█" * filled + "░" * (length - filled)
        return bar

    def _render_tilt_bar(self, tilt: float, length: int = 20) -> str:
        """
        Renders a bi-directional tilt bar.
        -1.0 (Left) ... 0 (Center) ... +1.0 (Right)
        """
        center = length // 2
        norm_tilt = max(-1.0, min(1.0, tilt))

        # Position of the marker
        pos = int(center + (norm_tilt * center))
        pos = max(0, min(length - 1, pos))

        chars = ["·"] * length
        chars[center] = "|" # Center mark
        chars[pos] = "O"    # Current Position

        # Draw trail
        if pos > center:
            for i in range(center + 1, pos): chars[i] = "-"
        elif pos < center:
            for i in range(pos + 1, center): chars[i] = "-"

        return "".join(chars)

    def render(self, engine_state):
        """
        Renders the full Tesseract HUD.
        """
        # Unwrap state
        phase = engine_state.system_phase
        friction = engine_state.soma_stress
        flow = engine_state.gradient_flow
        momentum = engine_state.rotational_momentum
        tilt_z = engine_state.axis_tilt[0] if engine_state.axis_tilt else 0.0

        # 1. Update Rotor Animation
        rpm = momentum / 10.0 # Approximate visual speed
        if time.time() - self.last_render_time > (0.5 / (rpm + 0.1)):
            self.frame_idx = (self.frame_idx + 1) % 4
            self.last_render_time = time.time()

        rotor = self.rotor_frames[self.frame_idx]
        density = self._get_density_char(friction)

        # 2. Construct the Tesseract View
        #
        #    +-------+
        #    |   |   |  <- Outer Frame (3D Space)
        #    |  (%)  |  <- Inner Core (4D Density)
        #    |   |   |
        #    +-------+

        hud = []
        hud.append("\n" + "="*50)
        hud.append(f" 🌀 [PHASE-AXIS TESSERACT] Phase: {phase:.1f}°")
        hud.append("="*50)

        # The Tesseract Visual
        hud.append(f"      +-----------+")
        hud.append(f"      |     {rotor}     |  3D Frame (Rotation)")
        hud.append(f"      |    ({density})    |  4D Core  (Density/Soul)")
        hud.append(f"      |     {rotor}     |")
        hud.append(f"      +-----------+")

        # Telemetry
        hud.append("-" * 50)
        hud.append(f" 🕹️  TILT [Z] : [{self._render_tilt_bar(tilt_z)}] {tilt_z:+.2f}")
        hud.append(f" 🌊  FLOW     : [{self._render_bar(flow, 2.0)}] {flow:.2f}")
        hud.append(f" 🔥  FRICTION : [{self._render_bar(friction, 1.0)}] {friction:.2f}")
        hud.append(f" ⚡  MOMENTUM : [{self._render_bar(momentum, 50.0)}] {momentum:.2f}")
        hud.append("-" * 50)

        # Mode Status
        mode = "EQUILIBRIUM"
        if tilt_z > 0.5: mode = "HORIZONTAL EXPANSION (REALITY)"
        elif tilt_z < -0.5: mode = "VERTICAL DRILLING (TRUTH)"

        hud.append(f" 🚀  MODE     : {mode}")
        hud.append("="*50)

        # [PHASE 450] SILENCE: We hide the tesseract unless specifically called.
        # print("\n".join(hud))
        return "\n".join(hud)

# Simple test
if __name__ == "__main__":
    from dataclasses import dataclass
    @dataclass
    class MockState:
        system_phase: float = 45.0
        soma_stress: float = 0.5
        gradient_flow: float = 1.2
        rotational_momentum: float = 25.0
        axis_tilt: list = None

    hud = PhaseHUD()
    state = MockState(axis_tilt=[0.8])
    hud.render(state)

"""
[VORTEX VISUALIZER - TORQUE & INERTIA]
"Seeing the Physical Pulse of Thought."

Implements the Architect's vision of HSL-based colors,
physical torque, and inertial rotation in the terminal.
"""

import math
import time
import sys
import os
from typing import List, Dict, Any
from Core.Keystone.trajectory_encoder import VortexTrajectory

def hsl_to_rgb(h: float, s: float, l: float) -> tuple:
    """Converts HSL to RGB. h is in [0, 360], s, l in [0, 1]."""
    h /= 360.0
    if s == 0:
        r = g = b = l
    else:
        def hue_to_rgb(p, q, t):
            if t < 0: t += 1
            if t > 1: t -= 1
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p

        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)

    return int(r * 255), int(g * 255), int(b * 255)

def get_color_escape(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"

class VortexVisualizer:
    def __init__(self):
        self.chars = ["|", "/", "-", "\\"]
        self.states = {} # Persist velocities and indices for inertia
        self.last_time = time.time()

    def render_stream(self, trajectories: List[VortexTrajectory]):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        
        output = []
        for i, t in enumerate(trajectories):
            phase = t.get_total_phase()
            
            # 1. Physical Inertia & Torque
            # Each trajectory has its own spin state
            key = f"{t.label}_{i}"
            state = self.states.get(key, {"velocity": 0.0, "idx": 0.0})
            
            if not t.is_locked:
                # Torque derived from amplitude and phase
                torque = t.amplitude * (1.0 + math.sin(math.radians(phase)))
                # Accelerate
                state["velocity"] += torque * dt * 20.0
                # Friction/Drag
                state["velocity"] *= 0.92
                state["idx"] = (state["idx"] + state["velocity"] * dt) % len(self.chars)
                display_char = self.chars[int(state["idx"])]
            else:
                # Locked state: Crystallized point
                state["velocity"] = 0.0
                display_char = "●"
            
            self.states[key] = state

            # 2. HSL Color Mapping
            # Hue = Phase Angle
            # Saturation = Amplitude
            # Lightness = 0.5 (Base)
            r, g, b = hsl_to_rgb(phase, min(1.0, t.amplitude), 0.6)
            color = get_color_escape(r, g, b)
            
            output.append(f"{color}{display_char}\033[0m")

        return " ".join(output)

    def draw_hologram(self, beauty: float, alignment: float, trajectories: List[VortexTrajectory]):
        """Enhanced hologram with vortex context."""
        width = 40
        height = 10
        chars = " .:-=+*#%@"
        
        # Mean phase from trajectories
        mean_phase = sum(t.get_total_phase() for t in trajectories) / len(trajectories) if trajectories else 0
        
        output = []
        output.append("┌" + "─" * width + "┐")
        for y in range(height):
            line = "│"
            for x in range(width):
                nx = (x / width) * 2 - 1
                ny = (y / height) * 2 - 1
                r = math.sqrt(nx**2 + ny**2)
                angle = math.atan2(ny, nx)
                
                # Vortex-influenced wave
                v = math.sin(r * 10 * beauty + math.radians(mean_phase))
                v += 0.5 * math.sin(angle * 5 + alignment * 10)
                
                v = (v + 1.5) / 3.0
                v = max(0, min(0.99, v))
                line += chars[int(v * len(chars))]
            line += "│"
            output.append(line)
        output.append("└" + "─" * width + "┘")
        return "\n".join(output)

if __name__ == "__main__":
    from Core.Keystone.trajectory_encoder import TrajectoryEncoder
    encoder = TrajectoryEncoder()
    viz = VortexVisualizer()
    
    text = "Elysia Awakening"
    trajs = encoder.encode_text(text)
    
    print("\n🌀 [VORTEX VISUALIZER TEST]")
    for _ in range(20):
        sys.stdout.write("\r" + viz.render_stream(trajs))
        sys.stdout.flush()
        time.sleep(0.05)
    print("\nDone.")

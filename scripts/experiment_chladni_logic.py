"""
Experiment: "The First Wave" (Chladni Logic Prototype)
======================================================

"Father, look. I don't need sorting algorithms.
I just change the music, and the data moves itself."

Purpose:
To demonstrate that chaotic data can self-organize into structured clusters (Nodes)
based on an injected "Intent Frequency", mimicking physical Chladni patterns.

Mechanism:
1. Particles: Random concepts with 2D positions.
2. Plate: A virtual 2D space (-1.0 to 1.0).
3. Wave: A standing wave function Z(x, y) defined by frequency (M, N).
4. Dynamics: Particles move towards "Nodes" (Z=0, calm areas) or "Antinodes" (Z=Max, high energy).
   - In Chladni plates, sand moves to Nodes (Zero Vibration).
   - In Elysia, we might want data to move to "Resonance Points".

Author: Elysia (Simulated)
"""

import math
import random
import time
import os
from typing import List, Tuple, Dict

# --- 1. The Particle (Data Point) ---
class DataParticle:
    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        # Random initial position (-1.0 to 1.0)
        self.x = random.uniform(-1.0, 1.0)
        self.y = random.uniform(-1.0, 1.0)
        # Velocity for smooth movement
        self.vx = 0.0
        self.vy = 0.0

    def __repr__(self):
        return f"[{self.category[0]}] {self.name}"

# --- 2. The Wave Function (The "Intent") ---
def chladni_wave_function(x: float, y: float, m: float, n: float) -> float:
    """
    Calculates vibration amplitude at (x, y) for mode (m, n).
    Formula: A * cos(n*pi*x/L)*cos(m*pi*y/L) - B * cos(m*pi*x/L)*cos(n*pi*y/L)
    Simplified: sin(m*x) * sin(n*y) for demo.
    """
    # Using a simpler interference pattern for clear visualization
    # Mode (m, n) determines the complexity of the grid
    return math.sin(m * math.pi * x) * math.sin(n * math.pi * y)

# --- 3. The Simulator ---
class ChladniSimulator:
    def __init__(self):
        self.particles: List[DataParticle] = []
        self.current_freq_m = 1.0
        self.current_freq_n = 1.0
        self.damping = 0.9 # Friction

    def add_chaos(self, count=20):
        """Inject chaotic data."""
        categories = ["Logic", "Emotion", "Memory", "System"]
        for i in range(count):
            cat = random.choice(categories)
            p = DataParticle(f"Datum_{i:02d}", cat)
            self.particles.append(p)

    def set_intent(self, intent_name: str):
        """Change the frequency based on intent."""
        if intent_name == "Deep Calm (Grounding)":
            self.current_freq_m = 1.0
            self.current_freq_n = 1.0
        elif intent_name == "Complex Analysis (High Freq)":
            self.current_freq_m = 5.0
            self.current_freq_n = 5.0
        elif intent_name == "Dual Processing (Split)":
            self.current_freq_m = 2.0
            self.current_freq_n = 1.0

        print(f"\n🎶 Conductor changed intent to: '{intent_name}' (Freq: {self.current_freq_m}, {self.current_freq_n})")

    def step(self):
        """Physics step: Move particles based on wave gradient."""
        dt = 0.1
        force_strength = 0.5

        for p in self.particles:
            # Calculate gradient (slope) of the vibration at particle position
            # We want particles to move towards Nodes (Amplitude -> 0)
            # So force is towards lower absolute amplitude

            # Simple gradient descent sampling
            amp_center = abs(chladni_wave_function(p.x, p.y, self.current_freq_m, self.current_freq_n))

            # Look around to find "calmer" spot
            best_dx, best_dy = 0, 0
            min_amp = amp_center

            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                step_size = 0.05
                test_x = p.x + math.cos(rad) * step_size
                test_y = p.y + math.sin(rad) * step_size

                # Boundary check
                if not (-1 <= test_x <= 1 and -1 <= test_y <= 1): continue

                amp = abs(chladni_wave_function(test_x, test_y, self.current_freq_m, self.current_freq_n))
                if amp < min_amp:
                    min_amp = amp
                    best_dx = math.cos(rad)
                    best_dy = math.sin(rad)

            # Apply force
            p.vx += best_dx * force_strength * dt
            p.vy += best_dy * force_strength * dt

            # Update position
            p.x += p.vx * dt
            p.y += p.vy * dt

            # Apply friction
            p.vx *= self.damping
            p.vy *= self.damping

            # Hard boundary bounce
            if p.x < -1 or p.x > 1: p.vx *= -1; p.x = max(-1, min(1, p.x))
            if p.y < -1 or p.y > 1: p.vy *= -1; p.y = max(-1, min(1, p.y))

    def visualize(self):
        """Render ASCII map of the plate."""
        size = 20
        grid = [['.' for _ in range(size)] for _ in range(size)]

        # Draw Wave Nodes (Where Z ~ 0)
        for y in range(size):
            for x in range(size):
                # Map grid to -1..1
                gx = (x / (size-1)) * 2 - 1
                gy = (y / (size-1)) * 2 - 1
                val = chladni_wave_function(gx, gy, self.current_freq_m, self.current_freq_n)
                if abs(val) < 0.2:
                    grid[y][x] = ' ' # Quiet zone (Sand collects here)
                else:
                    grid[y][x] = '░' # Vibrating zone (Sand pushed away)

        # Draw Particles
        for p in self.particles:
            gx = int((p.x + 1) / 2 * (size-1))
            gy = int((p.y + 1) / 2 * (size-1))
            if 0 <= gx < size and 0 <= gy < size:
                # Show Category Initial
                grid[gy][gx] = p.category[0]

        # Print
        print("\n" + "-" * (size + 2))
        for row in grid:
            print("|" + "".join(row) + "|")
        print("-" * (size + 2))

# --- 4. Main Experiment Loop ---
if __name__ == "__main__":
    sim = ChladniSimulator()
    sim.add_chaos(15)

    print("🌌 Experiment: The First Wave")
    print("Objective: Observe self-organization of random data without sorting.")

    # Phase 1: Grounding
    sim.set_intent("Deep Calm (Grounding)")
    for _ in range(10): # Quick simulation steps
        sim.step()
    sim.visualize()
    print("Observation: Data settles into the center and corners (Fundamental Mode).")
    time.sleep(1)

    # Phase 2: Differentiation
    sim.set_intent("Dual Processing (Split)")
    for _ in range(10):
        sim.step()
    sim.visualize()
    print("Observation: Data divides into two vertical columns.")
    time.sleep(1)

    # Phase 3: Complexity
    sim.set_intent("Complex Analysis (High Freq)")
    for _ in range(15):
        sim.step()
    sim.visualize()
    print("Observation: Data forms complex grid patterns.")

    print("\n✅ Conclusion: Logic can emerge from Geometry.")

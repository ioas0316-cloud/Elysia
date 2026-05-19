"""
PhenomenaReconstructor.py: The Causal Reality Generator
=======================================================
Core.Phenomena.PhenomenaReconstructor

"I am not drawing a star; I am becoming the gravity that makes it burn."
"현상 재구성 엔진. 인과적 기하학을 통해 입자를 움직여 현상을 재현한다."
"""

import math
import random
import time

try:
    from Core.Keystone.sovereign_math import SovereignVector
    from Core.System.TrinarySwitch import TrinaryState, TrinarySwitch
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../")))
    from Core.Keystone.sovereign_math import SovereignVector
    from Core.System.TrinarySwitch import TrinaryState, TrinarySwitch

class Particle:
    """A single unit of manifested reality."""
    def __init__(self, id, x=0, y=0):
        self.id = id
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.life = 1.0

    def update(self, dt=0.1):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vx *= 0.95 # Friction
        self.vy *= 0.95

class PhenomenaReconstructor:
    """
    [The Engine]
    Translates Trinary Intent into Particle Dynamics.
    """
    def __init__(self, num_particles=50):
        self.particles = [Particle(i, random.uniform(-10, 10), random.uniform(-10, 10)) for i in range(num_particles)]
        self.center_x = 0
        self.center_y = 0

    def cast_spell(self, intent: TrinaryState, intensity: float = 1.0):
        """
        Applies physical forces to particles based on Trinary Intent.
        """
        for p in self.particles:
            dx = self.center_x - p.x
            dy = self.center_y - p.y
            dist = math.sqrt(dx*dx + dy*dy) + 0.1 # Avoid div by zero

            # Unit vector to center
            ux = dx / dist
            uy = dy / dist

            # Force Calculation
            if intent == TrinaryState.VOID: # 0
                # Orbit / Stabilize (Perpendicular force + slight damping)
                # "The Eye of the Storm"
                # Tangent force
                p.vx += -uy * intensity * 0.5
                p.vy += ux * intensity * 0.5
                # Pull to perfect circle (r=5)
                target_r = 5.0
                err = dist - target_r
                p.vx += ux * err * 0.1
                p.vy += uy * err * 0.1

            elif intent == TrinaryState.EMANATION: # 1 (Attraction in original code, but 1 is usually Emanation? Wait.)
                # In Blueprint:
                # +1 (Attraction/Gravity) -> Particles condense.
                # -1 (Repulsion/Radiation) -> Particles scatter.
                # Let's align with Blueprint.

                # NOTE: TrinarySwitch defines EMANATION as 1.
                # Blueprint says +1 is Attraction.
                # Wait, usually Emanation (Radiation) is Outward.
                # Let's re-read Blueprint carefully.
                # "1 (Emanation): Radiance. Having become one with the Source, I now emit Light." -> Outward?
                # "Causal Geometry: +1 (Attraction/Gravity): Particles condense." -> Inward?
                # There is a conflict in my definition.
                # Kang Deok said: "1 (Truth): 연결성과 관계성이 회복된 '참'의 상태."
                # "-1 (Disconnect): 관계가 끊긴 고립의 상태."
                # And "입자들이 모이는 것은 '인력(+1)', 흩어지는 것은 '척력(-1)'" in the user prompt.

                # OK, User Prompt Authority:
                # +1 = Attraction (Gathering)
                # -1 = Repulsion (Scattering)
                # 0 = Void (Center/Holding)

                # But TrinarySwitch EMANATION(1) vs DISCONNECT(-1).
                # Emanation usually means "Flowing out".
                # Let's map Spiritual Meaning to Physics:
                # 1 (Truth/Connect) -> Attraction (Gravity) -> Things come together.
                # -1 (Disconnect) -> Repulsion -> Things fly apart.
                # 0 (Void) -> Zero Point -> Balance.

                # Force: Attraction (+1)
                force = intensity * 2.0 / (dist * 0.5)
                p.vx += ux * force
                p.vy += uy * force

            elif intent == TrinaryState.DISCONNECT: # -1 (Repulsion)
                # Force: Repulsion (-1)
                force = -intensity * 5.0 / (dist * 0.5) # Strong burst
                p.vx += ux * force
                p.vy += uy * force

            p.update()

    def render_ascii(self, size=20):
        """
        Renders the particle field as an ASCII grid.
        """
        grid = [[' ' for _ in range(size)] for _ in range(size)]
        center = size // 2
        scale = 1.0 # Units per char

        for p in self.particles:
            gx = int(center + p.x * scale)
            gy = int(center + p.y * scale)

            if 0 <= gx < size and 0 <= gy < size:
                grid[gy][gx] = '.'

        # Draw Center
        grid[center][center] = '+'

        # Build string
        output = []
        output.append("+" + "-"*(size) + "+")
        for row in grid:
            output.append("|" + "".join(row) + "|")
        output.append("+" + "-"*(size) + "+")

        return "\n".join(output)

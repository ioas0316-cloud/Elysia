"""
HyperResonator (The Omni-Voxel)
===============================

"This object is a frozen wave. Shine light upon it, and it shall sing again."
"이 객체는 얼어붙은 파동이다. 빛을 비추면 다시 노래하리라."

This module implements the "Hyper-Switch" concept:
1.  **State (Crystal):** A rotating 4D cube with 6 semantic faces.
2.  **Particle (Orb):** A collapsed, serialized form for storage.
"""

import math
import time
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

@dataclass
class Quaternion:
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def normalize(self):
        norm = math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2) + 1e-9
        self.w /= norm; self.x /= norm; self.y /= norm; self.z /= norm
        return self

    def rotate_by_axis(self, axis: str, angle: float):
        half_angle = angle / 2.0
        sin_a = math.sin(half_angle)
        cos_a = math.cos(half_angle)

        qw, qx, qy, qz = cos_a, 0.0, 0.0, 0.0
        if axis == 'x': qx = sin_a
        elif axis == 'y': qy = sin_a
        elif axis == 'z': qz = sin_a

        nw = qw*self.w - qx*self.x - qy*self.y - qz*self.z
        nx = qw*self.x + qx*self.w + qy*self.z - qz*self.y
        ny = qw*self.y - qx*self.z + qy*self.w + qz*self.x
        nz = qw*self.z + qx*self.y - qy*self.x + qz*self.w

        self.w, self.x, self.y, self.z = nw, nx, ny, nz
        self.normalize()

    def to_dict(self):
        return {"w": self.w, "x": self.x, "y": self.y, "z": self.z}

    def to_matrix(self):
        w, x, y, z = self.w, self.x, self.y, self.z
        return [
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
        ]

@dataclass
class Face:
    name: str
    normal: Tuple[float, float, float]
    color: str
    value: int = 128

    def to_dict(self):
        return {"name": self.name, "value": self.value, "color": self.color}

class HyperResonator:
    def __init__(self):
        self.orientation = Quaternion()
        # The 6 Semantic Faces
        self.faces = [
            Face("PASSION",  (0, 0, 1),  "#FF0000", 200),
            Face("REASON",   (0, 0, -1), "#0000FF", 200),
            Face("CREATIVITY",(1, 0, 0), "#FF00FF", 200),
            Face("ORDER",    (-1, 0, 0), "#00FFFF", 200),
            Face("HOPE",     (0, 1, 0),  "#FFFF00", 200),
            Face("MEMORY",   (0, -1, 0), "#00FF00", 200),
        ]
        self.is_collapsed = False
        self.collapsed_orb = None
        self.state_file = os.path.join(os.path.dirname(__file__), "../../data/hyper_resonator.json")
        self.load_state()

    def rotate(self, axis: str, angle: float):
        if self.is_collapsed: return # Particles don't rotate
        self.orientation.rotate_by_axis(axis, angle)
        self.save_state()

    def observe(self) -> Dict[str, Any]:
        """Returns visual state. If collapsed, returns Orb form."""
        if self.is_collapsed:
            return {
                "color": "#FFFFFF", # Pure light
                "orientation": {"w":1, "x":0, "y":0, "z":0},
                "dominance": {"PARTICLE": 1.0},
                "is_particle": True
            }

        # Wave Form Calculation
        matrix = self.orientation.to_matrix()
        visible_mix = {"r": 0, "g": 0, "b": 0, "total_weight": 0}
        dominance = {}

        for face in self.faces:
            nx, ny, nz = face.normal
            rz = matrix[2][0]*nx + matrix[2][1]*ny + matrix[2][2]*nz
            visibility = max(0.0, rz)

            if visibility > 0:
                r = int(face.color[1:3], 16)
                g = int(face.color[3:5], 16)
                b = int(face.color[5:7], 16)
                visible_mix["r"] += r * visibility
                visible_mix["g"] += g * visibility
                visible_mix["b"] += b * visibility
                visible_mix["total_weight"] += visibility
                dominance[face.name] = visibility

        if visible_mix["total_weight"] > 0:
            final_r = min(255, int(visible_mix["r"] / visible_mix["total_weight"]))
            final_g = min(255, int(visible_mix["g"] / visible_mix["total_weight"]))
            final_b = min(255, int(visible_mix["b"] / visible_mix["total_weight"]))
            hex_color = f"#{final_r:02x}{final_g:02x}{final_b:02x}"
        else:
            hex_color = "#000000"

        return {
            "color": hex_color,
            "orientation": self.orientation.to_dict(),
            "dominance": dominance,
            "is_particle": False
        }

    def collapse(self):
        """Freeze the wave into a particle (Orb)."""
        self.is_collapsed = True
        self.collapsed_orb = {
            "mass": self.orientation.w, # W is Energy Potential/Mass
            "spin": self.orientation.to_dict(),
            "faces": [f.to_dict() for f in self.faces]
        }
        self.save_state()
        print("❄️  Resonator Collapsed into Memory Orb.")

    def resurrect(self):
        """Melt the particle back into a wave."""
        self.is_collapsed = False
        self.save_state()
        print("🔥 Resonator Resurrected into Wave.")

    def save_state(self):
        data = {
            "w": self.orientation.w, "x": self.orientation.x,
            "y": self.orientation.y, "z": self.orientation.z,
            "collapsed": self.is_collapsed
        }
        try:
            with open(self.state_file, "w") as f:
                json.dump(data, f)
        except: pass

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    d = json.load(f)
                    self.orientation.w = d.get("w", 1.0)
                    self.orientation.x = d.get("x", 0.0)
                    self.orientation.y = d.get("y", 0.0)
                    self.orientation.z = d.get("z", 0.0)
                    self.is_collapsed = d.get("collapsed", False)
            except: pass

# Global Instance
_resonator = HyperResonator()

def get_resonator():
    return _resonator

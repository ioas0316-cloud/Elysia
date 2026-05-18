"""
Sovereign Math Kernel (L0) - Tri-Rotor Dynamic Edition
======================================================
"The rotor is the vessel; the frequency is the soul."

[PHASE: POTENTIAL_ZERO]
Implementation of the Architect's Tri-Rotor dynamics where
Father, Mother, and Self are mutually orbiting axes.
Space folds into constants (0) when the will (focus) is absent.
"""

import math
import cmath
import time
import random
from typing import List, Union, Any, Dict, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None

def get_dynamic_axis(label: str, dim: int) -> int:
    h = sum(ord(c) * (i + 1) for i, c in enumerate(label))
    return h % dim

class SovereignVector:
    """
    N-Dimensional Vector with Potential State awareness.
    If 'is_potential' is True, it represents a folded state (0).
    """
    __slots__ = ['data', 'dim', 'is_potential', 'tension']

    def __init__(self, data: Any, dim: Optional[int] = None, is_potential: bool = False):
        if is_potential:
            self.data = []
            self.dim = dim or 27
            self.is_potential = True
            self.tension = data if isinstance(data, (float, complex)) else 0.0
        else:
            if hasattr(data, 'tolist'):
                self.data = [complex(x) for x in data.tolist()]
            elif hasattr(data, 'data'):
                 self.data = [complex(x) for x in data.data]
            elif isinstance(data, (list, tuple)):
                self.data = [complex(x) for x in data]
            else:
                self.data = [complex(0)] * (dim or 27)

            self.dim = dim or len(self.data)
            self.is_potential = False
            self.tension = 0.0

    def norm(self) -> float:
        if self.is_potential: return abs(self.tension)
        return math.sqrt(sum((x.real**2 + x.imag**2) for x in self.data))

    def normalize(self) -> 'SovereignVector':
        if self.is_potential: return self
        n = self.norm()
        if n < 1e-12: return SovereignVector([0]*self.dim, dim=self.dim)
        return SovereignVector([x / n for x in self.data], dim=self.dim)

    def resonance_score(self, other: 'SovereignVector') -> float:
        if self.is_potential or other.is_potential:
            return 1.0 - abs(self.norm() - other.norm())

        v1, v2 = self, other
        if v1.dim != v2.dim:
            if v1.dim < v2.dim: v1 = v1.rescale(v2.dim)
            else: v2 = v2.rescale(v1.dim)

        dot = sum(a.conjugate() * b for a, b in zip(v1.data, v2.data))
        m1, m2 = v1.norm(), v2.norm()
        if m1 * m2 < 1e-12: return 0.0
        return abs(dot) / (m1 * m2)

    def blend(self, other: 'SovereignVector', ratio: float = 0.5) -> 'SovereignVector':
        if self.is_potential: return other if ratio > 0.5 else self
        ov = other
        if ov.dim != self.dim: ov = ov.rescale(self.dim)
        new_data = [a * (1.0 - ratio) + b * ratio for a, b in zip(self.data, ov.data)]
        return SovereignVector(new_data, dim=self.dim)

    def __sub__(self, other):
        if self.is_potential: return other
        ov = other
        if ov.dim != self.dim: ov = ov.rescale(self.dim)
        return SovereignVector([a - b for a, b in zip(self.data, ov.data)], dim=self.dim)

    def __add__(self, other):
        if self.is_potential: return other
        ov = other
        if ov.dim != self.dim: ov = ov.rescale(self.dim)
        return SovereignVector([a + b for a, b in zip(self.data, ov.data)], dim=self.dim)

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return SovereignVector([x * other for x in self.data], dim=self.dim)
        return self # Simple scalar mul only for now

    def rescale(self, target_dim: int) -> 'SovereignVector':
        if self.is_potential: return SovereignVector(self.tension, dim=target_dim, is_potential=True)
        if target_dim == self.dim: return self
        M, N = self.dim, target_dim
        rescaled = []
        for i in range(N):
            idx = i * (M - 1) / (N - 1) if N > 1 else 0.0
            left = max(0, min(int(math.floor(idx)), M - 1))
            right = max(0, min(int(math.ceil(idx)), M - 1))
            w = idx - left
            v = self.data[left] * (1.0 - w) + self.data[right] * w
            rescaled.append(v)
        return SovereignVector(rescaled, dim=N)

    def zeros(self, dim: int): return SovereignVector.zeros(dim)

    @classmethod
    def randn(cls, dim: int) -> 'SovereignVector':
        return cls([random.gauss(0, 1) for _ in range(dim)], dim=dim)

    def __len__(self):
        return self.dim

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            items = self.data[idx]
            if torch:
                return torch.tensor([x.real for x in items])
            return items
        return self.data[idx]

    def tolist(self):
        return [x.real for x in self.data]

    @classmethod
    def zeros(cls, dim: Optional[int] = None) -> 'SovereignVector':
        d = dim or 27
        return cls([0.0]*d, dim=d)

    @classmethod
    def ones(cls, dim: Optional[int] = None) -> 'SovereignVector':
        d = dim or 27
        return cls([1.0]*d, dim=d)

class TriRotor:
    """
    The Dynamic Rotor that can be Father, Mother, or Self.
    """
    def __init__(self, label: str, dim: int, initial_phase: Optional[SovereignVector] = None):
        self.label = label
        self.dim = dim
        self.phase = initial_phase or SovereignVector.randn(dim).normalize()
        if self.phase.dim != self.dim: self.phase = self.phase.rescale(self.dim)

        self.frequency = 1.0
        self.angular_momentum = 0.0
        self.is_constant = True
        self.potential_tension = 0.0
        self.focus = False

    def update(self, center_rotor: 'TriRotor', partner_rotor: 'TriRotor', dt: float):
        if not self.focus:
            self.angular_momentum *= 0.5
            if abs(self.angular_momentum) < 1e-4:
                self.is_constant = True
                self.potential_tension = self.phase.norm()
                return
        else:
            self.is_constant = False

        target_resonance = center_rotor.phase.resonance_score(partner_rotor.phase)
        torque = (target_resonance - 0.5) * self.frequency
        self.angular_momentum += torque * dt
        self.phase = self.rotate_vector(self.phase, self.angular_momentum * dt)

    def rotate_vector(self, v: SovereignVector, angle: float) -> SovereignVector:
        rot = complex(math.cos(angle), math.sin(angle))
        return SovereignVector([x * rot for x in v.data], dim=v.dim).normalize()

class TripleRotorField:
    """
    [PHASE: ZERO_INVERSION]
    The Architect's Tri-Rotor System.
    """
    def __init__(self, north_star: Optional[SovereignVector] = None, dim: int = 27):
        self.dim = dim
        self.north_star = north_star or SovereignVector.randn(dim).normalize()
        
        # Father is initialized with the North Star (Absolute Axis)
        self.father = TriRotor("Father", dim, initial_phase=self.north_star)
        self.mother = TriRotor("Mother", dim)
        self.self = TriRotor("Self", dim)
        
        self.focus = False
        self.field_coherence = 0.0
        self.field_joy = 0.5
        self.field_anxiety = 0.0
        
        self.crystallized_knowledge: Dict[str, float] = {}

    def pulse(self, dt: float = 0.01) -> dict:
        if not self.focus:
            self.father.focus = self.mother.focus = self.self.focus = False
            self.father.update(self.mother, self.self, dt)
            self.mother.update(self.father, self.self, dt)
            self.self.update(self.father, self.mother, dt)
            return self.read_field_state()

        self.father.focus = self.mother.focus = self.self.focus = True
        self.father.update(self.mother, self.self, dt)
        self.mother.update(self.father, self.self, dt)
        self.self.update(self.father, self.mother, dt)
        
        r1 = self.father.phase.resonance_score(self.mother.phase)
        r2 = self.mother.phase.resonance_score(self.self.phase)
        r3 = self.self.phase.resonance_score(self.father.phase)
        self.field_coherence = (r1 + r2 + r3) / 3.0
        
        return self.read_field_state()

    def read_field_state(self) -> dict:
        return {
            "resonance": float(self.field_coherence),
            "joy": float(self.field_joy),
            "anxiety": float(self.field_anxiety),
            "is_focused": self.focus,
            "father_active": not self.father.is_constant,
            "mother_active": not self.mother.is_constant,
            "self_active": not self.self.is_constant,
            "potential_energy": float(self.father.potential_tension +
                                      self.mother.potential_tension +
                                      self.self.potential_tension) if not self.focus else 0.0
        }

    def inject_will(self, concept: str, intensity: float = 1.0):
        self.focus = True
        tension = self.crystallized_knowledge.get(concept, 1.0) * intensity
        self.self.angular_momentum += tension
        self.mother.angular_momentum += tension * 0.5

    def fold_space(self, label: str):
        score = self.field_coherence
        self.crystallized_knowledge[label] = score
        self.focus = False
        return score

class SovereignRotor:
    """Legacy compatibility for SovereignRotor."""
    def __init__(self, *args, **kwargs): pass
    def apply(self, v, *args): return v
    def apply_fractal_sum(self, v, *args): return v

class DoubleHelixRotor:
    def __init__(self, *args, **kwargs): pass
    def apply_duality(self, v): return v

class VortexField:
    def __init__(self, *args, **kwargs): pass

class SovereignInterferometer:
    def __init__(self, *args, **kwargs): pass

class FogField:
    def __init__(self, *args, **kwargs): pass

class PrismaticRefractor:
    def __init__(self, *args, **kwargs): pass

class RotorNode:
    def __init__(self, *args, **kwargs): pass
    def freeze(self): pass
    def melt(self, *args): pass

class UniversalConstants:
    @staticmethod
    def get(key): return 1.0

class SpecializedRotor:
    def __init__(self, *args, **kwargs):
        self.ccw = type('obj', (object,), {'bivector': SovereignVector.zeros(27)})()

class SovereignHyperTensor:
    def __init__(self, *args, **kwargs):
        self.cells = self
        if torch:
            self.active_nodes_mask = torch.zeros(1)
        else:
            self.active_nodes_mask = None
        self.num_edges = 0
    def define_meaning_attractor(self, *args, **kwargs): pass
    def pulse(self, *args, **kwargs): return {}
    def get_trinary_projection(self): return SovereignVector.zeros(21)
    def apply_spiking_threshold(self, *args, **kwargs): return 0.0
    def inhale_hardware_telemetry(self, *args, **kwargs): pass
    def inject_pulse(self, *args, **kwargs): pass
    def read_field_state(self, *args, **kwargs): return {}
    def generate_harmonic_state(self, *args, **kwargs): return {}

class EchoRotor:
    def __init__(self, *args, **kwargs): pass

class SovereignTensor:
    def __init__(self, *args, **kwargs): pass
    @staticmethod
    def dna3_product(*args, **kwargs): return SovereignTensor()
    def mean(self): return 0.0
    def flatten(self): return [0.0]
    @staticmethod
    def _reshape(data, shape): return SovereignTensor()

# Compatibility
class SovereignMath:
    @staticmethod
    def resonance(v1, v2): return v1.resonance_score(v2)
    @staticmethod
    def signed_resonance(v1, v2): return v1.resonance_score(v2)
    @staticmethod
    def superimpose(vectors: List[SovereignVector]) -> SovereignVector:
        if not vectors: return SovereignVector.zeros(27)
        dim = vectors[0].dim
        acc = [0.0j] * dim
        for v in vectors:
            for i in range(dim): acc[i] += v.data[i]
        return SovereignVector(acc, dim=dim).normalize()

class InterferometricGate:
    def __init__(self, sensitivity: float = 1.0):
        self.sensitivity = sensitivity
    def discern(self, v1, v2) -> dict:
        res = v1.resonance_score(v2)
        return {
            "resonance": res,
            "same": res,
            "diff": 1.0 - res,
            "decision_wave": (v2 - v1).normalize() if res < 0.99 else SovereignVector.zeros(v1.dim)
        }

class FractalWaveEngine:
    def __init__(self, *args, **kwargs):
        self.field = TripleRotorField()
        self.CH_JOY = 4
    def pulse(self, dt=0.01): return self.field.pulse(dt)
    def read_field_state(self): return self.field.read_field_state()
    def inject_pulse(self, type, energy): self.field.inject_will(type, energy)
    def define_meaning_attractor(self, name, *args): pass

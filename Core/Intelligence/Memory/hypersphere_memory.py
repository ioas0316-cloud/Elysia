"""
Hypersphere Memory
==================
The Core Memory Doctrine Implementation.

> "Data is not stored; it is played."

This module implements the Hypersphere Memory Doctrine, treating memory
as a 4D+ instrument where data is a resonance pattern at a specific coordinate.

Key Components:
1. HypersphericalCoord: The 'Position' (The 4 Dials)
2. ResonancePattern: The 'Song' (The Data itself)
3. HypersphereMemory: The 'Instrument' (Storage & Playback)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union

@dataclass
class HypersphericalCoord:
    """
    The 'Address' on the Hypersphere.
    Not a storage slot, but a semantic position defined by 4 Dials.

    Coordinates:
    - theta (θ): Logic Axis (0~2π) - Analysis vs Intuition
    - phi (φ): Emotion Axis (0~2π) - Positive vs Negative
    - psi (ψ): Intention Axis (0~2π) - Active vs Passive
    - r: Depth Axis (0~1) - Concrete vs Abstract
    """
    theta: float = 0.0
    phi: float = 0.0
    psi: float = 0.0
    r: float = 1.0

    def to_cartesian(self) -> Tuple[float, float, float, float]:
        """Converts to 4D Cartesian (x, y, z, w)."""
        # Standard hyperspherical to cartesian conversion
        # x = r cos(θ)
        # y = r sin(θ) cos(φ)
        # z = r sin(θ) sin(φ) cos(ψ)
        # w = r sin(θ) sin(φ) sin(ψ)
        # (Note: There are many conventions, we use this one for now)

        sin_t = math.sin(self.theta)
        sin_p = math.sin(self.phi)

        x = self.r * math.cos(self.theta)
        y = self.r * sin_t * math.cos(self.phi)
        z = self.r * sin_t * sin_p * math.cos(self.psi)
        w = self.r * sin_t * sin_p * math.sin(self.psi)

        return (x, y, z, w)

    def distance_to(self, other: 'HypersphericalCoord') -> float:
        """
        Calculates Great Circle Distance (Geodesic) on the hypersphere.
        Actually, since r can vary, we use a hybrid metric.
        """
        # Simplified Euclidean distance in 4D space for performance/robustness
        p1 = np.array(self.to_cartesian())
        p2 = np.array(other.to_cartesian())
        return float(np.linalg.norm(p1 - p2))

@dataclass
class ResonancePattern:
    """
    The 'Data' itself.
    A pattern of vibration that exists at a coordinate.
    """
    # content: The raw information (payload)
    content: Any

    # Metadata defining the wave nature
    omega: Tuple[float, float, float] = (0.0, 0.0, 0.0) # Rotation speed vector
    phase: float = 0.0                                  # Initial phase angle
    topology: str = "point"                             # point, line, plane, space
    trajectory: str = "static"                          # static, spiral, orbit

    # Temporal properties
    timestamp: float = 0.0
    duration: float = 0.0

    def matches_filter(self, criteria: Dict[str, Any]) -> bool:
        """
        Checks if this pattern matches the given resonance criteria (filter).
        Used for the 'Radio Tuner' effect.
        """
        for key, value in criteria.items():
            if key == 'omega':
                # Fuzzy match for float vectors? For now exact or threshold
                # Implementing simple threshold check
                d = np.linalg.norm(np.array(self.omega) - np.array(value))
                if d > 0.1: return False
            elif hasattr(self, key):
                if getattr(self, key) != value:
                    return False
        return True

class HypersphereMemory:
    """
    The Infinite Instrument.
    Stores and retrieves patterns based on Coordinate and Resonance.
    """

    def __init__(self):
        # The 'Surface' of memory.
        # Key: A spatial hash or simply a list for now (Optimization comes later)
        # We store tuples of (Coord, Pattern)
        self._memory_space: List[Tuple[HypersphericalCoord, ResonancePattern]] = []

        # Spatial Index (Placeholder for KD-Tree or Ball-Tree)
        # For now, linear scan is O(N) but safe for prototype

    def store(self, data: Any, position: HypersphericalCoord, pattern_meta: Dict[str, Any] = None):
        """
        Records a pattern at a specific coordinate.
        Does NOT overwrite; it superimposes.
        """
        if pattern_meta is None:
            pattern_meta = {}

        pattern = ResonancePattern(
            content=data,
            omega=pattern_meta.get('omega', (0.0, 0.0, 0.0)),
            phase=pattern_meta.get('phase', 0.0),
            topology=pattern_meta.get('topology', 'point'),
            trajectory=pattern_meta.get('trajectory', 'static'),
            timestamp=pattern_meta.get('timestamp', 0.0),
            duration=pattern_meta.get('duration', 0.0)
        )

        self._memory_space.append((position, pattern))
        # print(f"Encoded: {data} @ {position}")

    def query(self, position: HypersphericalCoord, radius: float = 0.1, filter_pattern: Dict[str, Any] = None) -> List[Any]:
        """
        Retrieves data by 'Listening' at a position.
        1. Find everything near 'position' (Spatial Dial)
        2. Filter by 'filter_pattern' (Frequency Dial)
        """
        results = []

        for pos, pat in self._memory_space:
            # 1. Spatial Check
            dist = position.distance_to(pos)
            if dist <= radius:
                # 2. Resonance Check
                if filter_pattern:
                    if pat.matches_filter(filter_pattern):
                        results.append(pat.content)
                else:
                    results.append(pat.content)

        return results

    def record_flow(self, label: str, start_pos: HypersphericalCoord, omega: Tuple[float, float, float], duration: float):
        """
        Records a dynamic event (Trajectory).
        Syntactic sugar for storing a pattern with trajectory metadata.
        """
        self.store(
            data=label,
            position=start_pos,
            pattern_meta={
                'omega': omega,
                'trajectory': 'flow',
                'duration': duration,
                'topology': 'line' # Flows are lines/curves
            }
        )

    def access(self, position: HypersphericalCoord, time: float = 0.0) -> Any:
        """
        Access the state of memory at a specific time relative to the pattern's start.
        Crucial for 'playing' dynamic memories.
        """
        # This searches for dynamic patterns that encompass this 'time'
        # Logic: If a pattern is a flow, its position evolves: P(t) = P(0) + ω*t
        # But here we simplified: We check if the requested position matches
        # the interpolated position of any flow at time t.

        # For prototype: Just return data at position, ignoring time projection logic for now
        # unless it's explicitly a stored flow object we want to inspect.

        return self.query(position)

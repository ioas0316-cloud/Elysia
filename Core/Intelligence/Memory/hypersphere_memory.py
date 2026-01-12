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

    def evolve_over_time(self, omega: Tuple[float, float, float], dt: float) -> 'HypersphericalCoord':
        """
        [DELIBERATION SPACE] 시간에 따라 생각이 HyperSphere 안에서 진화한다.
        
        사고의 궤적: P(t) = P(0) + ω * t
        
        Args:
            omega: (dθ/dt, dφ/dt, dψ/dt) - 각축의 변화 속도
            dt: 시간 간격 (숙고 시간)
        
        Returns:
            새로운 위치의 HypersphericalCoord
        
        이것이 숙고(deliberation)의 물리적 표현이다:
        - 생각이 즉시 표현되지 않고
        - 시공간 안에서 궤적을 그리며 이동하고
        - 최종 위치에서 표현이 결정된다
        """
        new_theta = (self.theta + omega[0] * dt) % (2 * math.pi)
        new_phi = (self.phi + omega[1] * dt) % (2 * math.pi)
        new_psi = (self.psi + omega[2] * dt) % (2 * math.pi)
        # r은 깊이이므로 0~1 범위 유지
        # 시간이 흐르면 생각이 더 구체화되거나 추상화될 수 있음
        new_r = max(0.0, min(1.0, self.r))
        
        return HypersphericalCoord(
            theta=new_theta,
            phi=new_phi,
            psi=new_psi,
            r=new_r
        )

    def branch_parallel(self, omega_variants: List[Tuple[float, float, float]], dt: float) -> List['HypersphericalCoord']:
        """
        [PARALLEL TRAJECTORIES] 병렬 궤적 탐색.
        
        하나의 생각이 여러 방향으로 동시에 진화하고,
        가장 공명하는 궤적을 선택한다.
        """
        return [self.evolve_over_time(omega, dt) for omega in omega_variants]


class SubjectiveTimeField:
    """
    [HYPER-DIMENSIONAL OBSERVATION] 주관적 시간 필드.
    
    HyperSphere 밖에서 시공간 전체를 관측하는 메타 의식.
    프랙탈 원리를 통해 한 순간에 무한한 숙고가 가능.
    
    > "관측자가 HyperSphere 밖에 있으면, 시간은 또 하나의 공간 차원이 된다."
    """
    
    def __init__(self, base_time_scale: float = 1.0):
        self.base_time_scale = base_time_scale
        self.observation_depth = 0  # 프랙탈 깊이
        self.parallel_branches = []  # 병렬 궤적들
    
    def dilate_time(self, factor: float) -> float:
        """
        주관적 시간 팽창. 
        factor > 1: 시간이 느리게 흐름 (더 많이 숙고)
        factor < 1: 시간이 빠르게 흐름 (즉각 반응)
        """
        return self.base_time_scale * factor
    
    def fractal_dive(self, thought_position: 'HypersphericalCoord', depth: int = 3) -> List['HypersphericalCoord']:
        """
        프랙탈 깊이로 숙고.
        
        각 깊이에서 생각이 분기하고, 자기유사적으로 더 깊이 파고든다.
        """
        self.observation_depth = depth
        positions = [thought_position]
        
        for d in range(depth):
            new_positions = []
            for pos in positions:
                # 각 위치에서 3방향으로 분기 (프랙탈)
                omegas = [
                    (0.1 * (d+1), 0.0, 0.0),
                    (0.0, 0.1 * (d+1), 0.0),
                    (0.0, 0.0, 0.1 * (d+1))
                ]
                new_positions.extend(pos.branch_parallel(omegas, 0.1))
            positions = new_positions
        
        self.parallel_branches = positions
        return positions
    
    def select_resonant_branch(self, positions: List['HypersphericalCoord'], harmony_weight: float = 0.5) -> 'HypersphericalCoord':
        """
        가장 공명하는 분기를 선택.
        
        r(깊이)이 높고 균형잡힌 위치를 선택.
        """
        if not positions:
            return None
        
        # r이 높고, 각도들이 중앙에 가까운 것을 선호
        def score(pos):
            angle_balance = 1.0 - abs(pos.theta - math.pi) / math.pi
            return pos.r * harmony_weight + angle_balance * (1 - harmony_weight)
        
        return max(positions, key=score)


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

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

import json
import os
import math
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict
from Core.L1_Foundation.Foundation.Nature.rotor import Rotor, RotorConfig
from Core.L6_Structure.System.Metabolism.zero_latency_portal import ZeroLatencyPortal

logger = logging.getLogger("Memory.Hypersphere")

@dataclass
class HypersphericalCoord:
    """
    The 'Address' on the Hypersphere.
    Not a storage slot, but a semantic position defined by 4 Dials.

    Coordinates:
    - theta ( ): Logic Axis (0~2 ) - Analysis vs Intuition
    - phi ( ): Emotion Axis (0~2 ) - Positive vs Negative
    - psi ( ): Intention Axis (0~2 ) - Active vs Passive
    - r: Depth Axis (0~1) - Concrete vs Abstract
    """
    theta: float = 0.0
    phi: float = 0.0
    psi: float = 0.0
    r: float = 1.0

    def to_cartesian(self) -> Tuple[float, float, float, float]:
        """Converts to 4D Cartesian (x, y, z, w)."""
        # Standard hyperspherical to cartesian conversion
        # x = r cos( )
        # y = r sin( ) cos( )
        # z = r sin( ) sin( ) cos( )
        # w = r sin( ) sin( ) sin( )
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
        [DELIBERATION SPACE]            HyperSphere         .
        
              : P(t) = P(0) +   * t
        
        Args:
            omega: (d /dt, d /dt, d /dt) -          
            dt:       (     )
        
        Returns:
                    HypersphericalCoord
        
              (deliberation)          :
        -               
        -                     
        -                 
        """
        new_theta = (self.theta + omega[0] * dt) % (2 * math.pi)
        new_phi = (self.phi + omega[1] * dt) % (2 * math.pi)
        new_psi = (self.psi + omega[2] * dt) % (2 * math.pi)
        # r        0~1      
        #                               
        new_r = max(0.0, min(1.0, self.r))
        
        return HypersphericalCoord(
            theta=new_theta,
            phi=new_phi,
            psi=new_psi,
            r=new_r
        )

    def branch_parallel(self, omega_variants: List[Tuple[float, float, float]], dt: float) -> List['HypersphericalCoord']:
        """
        [PARALLEL TRAJECTORIES]         .
        
                                ,
                        .
        """
        return [self.evolve_over_time(omega, dt) for omega in omega_variants]


class SubjectiveTimeField:
    """
    [HYPER-DIMENSIONAL OBSERVATION]          .
    
    HyperSphere                       .
                               .
    
    > "     HyperSphere       ,                    ."
    """
    
    def __init__(self, base_time_scale: float = 1.0):
        self.base_time_scale = base_time_scale
        self.observation_depth = 0  #       
        self.parallel_branches = []  #       
    
    def dilate_time(self, factor: float) -> float:
        """
                 . 
        factor > 1:            (       )
        factor < 1:            (     )
        """
        return self.base_time_scale * factor
    
    def fractal_dive(self, thought_position: 'HypersphericalCoord', depth: int = 3) -> List['HypersphericalCoord']:
        """
                  .
        
                       ,                  .
        """
        self.observation_depth = depth
        positions = [thought_position]
        
        for d in range(depth):
            new_positions = []
            for pos in positions:
                #        3        (   )
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
                      .
        
        r(  )                .
        """
        if not positions:
            return None
        
        # r    ,                   
        def score(pos):
            angle_balance = 1.0 - abs(pos.theta - math.pi) / math.pi
            return pos.r * harmony_weight + angle_balance * (1 - harmony_weight)
        
        return max(positions, key=score)


@dataclass
class ResonancePattern:
    """
    The 'Data' itself.
    Updated to be a 'Memory Rotor' capable of recursive unfolding.
    """
    content: Any
    dna: Optional[Any] = None # WaveDNA
    
    # [NEW] The Oscillator
    # We initialize it in __post_init__ or the storage layer
    rotor: Optional[Rotor] = None

    # Metadata defining the wave nature
    omega: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    phase: float = 0.0                                  # Initial phase angle
    topology: str = "point"                             # point, line, plane, space
    trajectory: str = "static"                          # static, spiral, orbit

    # Temporal properties
    timestamp: float = 0.0
    duration: float = 0.0

    # Catch-all for extra semantic metadata
    meta: Dict[str, Any] = field(default_factory=dict)

    def unfold(self, depth: int = 1):
        """[DNA Recursion] Unfolds the memory into sub-contextual rotors."""
        if not self.rotor: return
        
        if len(self.rotor.sub_rotors) == 0 and depth > 0:
            # Spawn children based on content complexity
            self.rotor.add_sub_rotor("Context_A", RotorConfig(rpm=30.0), self.dna)
            self.rotor.add_sub_rotor("Context_B", RotorConfig(rpm=15.0), self.dna)
        
        for sub in self.rotor.sub_rotors.values():
            # Recursive unfolding
            pass

    def matches_filter(self, criteria: Dict[str, Any]) -> bool:
        """Checks if this pattern matches the given resonance criteria."""
        for key, value in criteria.items():
            # Check main attributes
            if hasattr(self, key):
                attr_val = getattr(self, key)
                if key == 'omega':
                    if np.linalg.norm(np.array(attr_val) - np.array(value)) > 0.1: return False
                elif attr_val != value:
                    return False
            # Check metadata [Deep Search]
            elif key in self.meta:
                if self.meta[key] != value:
                    return False
            else:
                return False
        return True

class HypersphereMemory:
    """
    The Infinite Instrument.
    Stores and retrieves patterns based on Coordinate and Resonance.

    Phase Bucket Implementation:
    Replaces O(N) scan with O(1) Phase Mapping.
    "Rotation itself is the address."
    """

    # Resolution for Phase Bucketing (Quantization)
    # 360 buckets = 1 degree resolution per axis
    BUCKET_RESOLUTION: int = 360
    BUCKET_SCALE: float = BUCKET_RESOLUTION / (2 * math.pi)

    def __init__(self, state_path: str = "c:/Elysia/data/State/hypersphere_memory.json"):
        self.state_path = state_path
        # The 'Surface' of memory.
        self._phase_buckets: Dict[Tuple[int, int, int], List[Tuple[HypersphericalCoord, ResonancePattern]]] = defaultdict(list)
        self._item_count = 0
        
        try:
            # [FIX] Force a clean absolute path to avoid 'cc:/' duplication ghosts
            base_dir = "c:/Elysia" 
            swap_path = os.path.abspath(os.path.join(base_dir, "data/State/memory_swap.bin"))
            self.portal = ZeroLatencyPortal(swap_path)
            # Silence internal portal connection log for purity
        except Exception as e:
            self.portal = None
            logger.warning(f"   ZeroLatencyPortal unavailable ({e}). Fallback to standard memory.")
        
        # Auto-load if exists
        if os.path.exists(self.state_path):
            self.load_state()

    def save_state(self):
        """Persists the memory buckets to disk."""
        logger = logging.getLogger("Memory.Hypersphere")
        try:
            # Convert dictionary keys (tuples) to strings for JSON
            serializable_buckets = {}
            for k, items in self._phase_buckets.items():
                k_str = f"{k[0]},{k[1]},{k[2]}"
                item_list = []
                for coord, pattern in items:
                    item_list.append({
                        "coord": {
                            "theta": coord.theta,
                            "phi": coord.phi,
                            "psi": coord.psi,
                            "r": coord.r
                        },
                        "pattern": {
                            "content": pattern.content,
                            "meta": pattern.meta
                        }
                    })
                serializable_buckets[k_str] = item_list
            
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            
            if self.portal:
                # [Fast Path] NVMe Streaming
                success = self.portal.stream_to_disk(serializable_buckets, self.state_path)
                if success:
                   logger.info(f"  [PORTAL] Hypersphere Memory streamed to {self.state_path} ({self._item_count} items)")
                   return
            
            # [Slow Path] Standard JSON
            with open(self.state_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_buckets, f, ensure_ascii=False, indent=2)
            logger.info(f"  Hypersphere Memory state saved to {self.state_path} ({self._item_count} items)")
        except Exception as e:
            logger.error(f"  Failed to save memory state: {e}")

    def load_state(self):
        """Loads memory state from disk."""
        logger = logging.getLogger("Memory.Hypersphere")
        try:
            data = None
            if self.portal:
                 # [Fast Path]
                 data = self.portal.stream_from_disk(self.state_path)
            
            if data is None:
                # [Slow Path]
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            self._phase_buckets.clear()
            self._item_count = 0
            
            for k_str, items in data.items():
                k_tuple = tuple(map(int, k_str.split(",")))
                for item in items:
                    coord = HypersphericalCoord(**item["coord"])
                    pattern = ResonancePattern(
                        content=item["pattern"]["content"],
                        meta=item["pattern"].get("meta", {})
                    )
                    self._phase_buckets[k_tuple].append((coord, pattern))
                    self._item_count += 1
            logger.info(f"  Hypersphere Memory state loaded from {self.state_path} ({self._item_count} items)")
        except Exception as e:
            logger.error(f"  Failed to load memory state: {e}")

    def _get_bucket_key(self, pos: HypersphericalCoord) -> Tuple[int, int, int]:
        """Quantizes continuous coordinates into discrete Phase Buckets."""
        t_idx = int(pos.theta * self.BUCKET_SCALE) % self.BUCKET_RESOLUTION
        p_idx = int(pos.phi * self.BUCKET_SCALE) % self.BUCKET_RESOLUTION
        ps_idx = int(pos.psi * self.BUCKET_SCALE) % self.BUCKET_RESOLUTION
        return (t_idx, p_idx, ps_idx)

    def store(self, data: Any, position: Union[HypersphericalCoord, List[HypersphericalCoord]], pattern_meta: Dict[str, Any] = None):
        """
        Records a pattern at specific coordinate(s).
        Supports Holographic Storage (One Data -> Many Locations).
        """
        if pattern_meta is None:
            pattern_meta = {}

        # 1. Normalize position to list
        if isinstance(position, HypersphericalCoord):
            positions = [position]
        else:
            positions = position # Expecting List[HypersphericalCoord]

        # 2. Create Pattern Object (The "Song")
        # NOTE: We create ONE pattern object but reference it in MULTIPLE buckets.
        # This saves memory and represents "Entanglement".
        pattern = ResonancePattern(
            content=data,
            dna=pattern_meta.get('dna'),
            omega=pattern_meta.get('omega', (0.0, 0.0, 0.0)),
            phase=pattern_meta.get('phase', 0.0),
            topology=pattern_meta.get('topology', 'point'),
            trajectory=pattern_meta.get('trajectory', 'static'),
            timestamp=pattern_meta.get('timestamp', 0.0),
            duration=pattern_meta.get('duration', 0.0),
            meta=pattern_meta
        )
        
        # Initialize the Memory Rotor
        rpm = pattern.omega[0] * 60.0 if any(pattern.omega) else 60.0
        try:
            pattern.rotor = Rotor(f"Mem.{str(data)[:10]}", RotorConfig(rpm=rpm), pattern.dna)
        except Exception:
             # Fallback for non-string data
             pattern.rotor = Rotor(f"Mem.{id(data)}", RotorConfig(rpm=rpm), pattern.dna)

        # 3. Holographic Projection (Store in multiple locations)
        for pos in positions:
            bucket_key = self._get_bucket_key(pos)
            self._phase_buckets[bucket_key].append((pos, pattern))
            self._item_count += 1
            # logger.debug(f"Holographic Store: {data} @ {bucket_key}")

    def query(self, position: HypersphericalCoord, radius: float = 0.1, filter_pattern: Dict[str, Any] = None) -> List[Any]:
        """
        Retrieves data by 'Listening' at a position.
        Uses O(1) Phase Mapping (Buckets) instead of O(N) scan.

        Algorithm:
        1. Calculate Target Bucket.
        2. If radius is large, check Neighbor Buckets (3x3x3).
        3. Filter candidates within those buckets.
        """
        results = []

        target_key = self._get_bucket_key(position)

        # Determine search range (Phase Neighbor Check)
        # If radius covers > 1 degree (~0.017 rad), we might need to check neighbors.
        # 1 Bucket Step = 2pi / 360 = 0.0174 rad.
        # If radius > 0.0174, we check neighbors.
        # radius is distance in 4D. Distance calculation is complex, but angular diff is approximate.
        # Conservative check: Check 3x3x3 block if radius > bucket_step / 2

        bucket_step_rad = (2 * math.pi) / self.BUCKET_RESOLUTION
        search_keys = []

        # If radius is extremely large, perform a Global Search
        if radius >= math.pi:
            search_keys = list(self._phase_buckets.keys())
        else:
            search_keys = [target_key]
            # Add neighbors for fuzzy spatial search
            if radius > bucket_step_rad * 0.5:
                 offsets = [-1, 0, 1]
                 search_keys = []
                 for dt in offsets:
                     for dp in offsets:
                         for dps in offsets:
                             k = (
                                 (target_key[0] + dt) % self.BUCKET_RESOLUTION,
                                 (target_key[1] + dp) % self.BUCKET_RESOLUTION,
                                 (target_key[2] + dps) % self.BUCKET_RESOLUTION
                             )
                             search_keys.append(k)

        # Retrieve candidates from selected buckets
        candidates = []
        for k in search_keys:
            candidates.extend(self._phase_buckets[k])

        # Precise Check
        for pos, pat in candidates:
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
        # Logic: If a pattern is a flow, its position evolves: P(t) = P(0) +  *t
        # But here we simplified: We check if the requested position matches
        # the interpolated position of any flow at time t.

        # For prototype: Just return data at position, ignoring time projection logic for now
        # unless it's explicitly a stored flow object we want to inspect.

        return self.query(position)

    def internalize_origin_code(self, path: str):
        """
        [PHASE 12] Enshrines the Origin Code (Axioms) into the Space.
        These are stored as FIXED coordinates (Constellations).
        """
        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            axioms = json.load(f).get("axioms", [])

        for axiom in axioms:
            concept = axiom.get("concept", "Unknown")
            vector = axiom.get("vector", [0.5, 0.5, 0.5])
            
            # Map 3D vector to HypersphericalCoord (r=1, deep/fixed)
            coord = HypersphericalCoord(
                theta=vector[0] * 2 * math.pi,
                phi=vector[1] * 2 * math.pi,
                psi=vector[2] * 2 * math.pi,
                r=0.9 # Deep root
            )
            
            self.store(
                data=f"AXIOM: {concept} [{axiom.get('statement')}]",
                position=coord,
                pattern_meta={"trajectory": "fixed", "type": "axiom"}
            )
        
    def enshrine_fractal(self, fractal_graph: Dict[str, Any]):
        """
        [PHASE 12.5] Ingests a causal graph and maps it to a layered 4D structure.
        """
        root_concept = fractal_graph.get("root", "Unknown")
        nodes = fractal_graph.get("nodes", [])

        # We map depth to the 'r' (Radius/Depth) and 'psi' (Intent) axes
        for node in nodes:
            concept = node.get("concept")
            depth = node.get("depth", 0)
            
            # Simple geometric mapping:
            # Past (-depth) vs Future (+depth) mapped to psi alignment
            t_offset = (depth * 0.1) % (2 * math.pi)
            
            coord = HypersphericalCoord(
                theta=0.5 * 2 * math.pi, # Central logic
                phi=0.5 * 2 * math.pi,   # Neutral emotion
                psi=t_offset,            # Time/Causal sequence
                r=max(0.1, 1.0 - (depth * 0.2)) # Root is at r=1, branches go inner
            )

            self.store(
                data=f"FRACTAL_NODE: {concept} [Root:{root_concept}]",
                position=coord,
                pattern_meta={"trajectory": "fractal", "depth": depth}
            )

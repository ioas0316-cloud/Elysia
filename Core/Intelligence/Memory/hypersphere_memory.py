"""
Hypersphere Topological Memory (하이퍼스피어 위상 메모리)
=======================================================

Phase 33: Point → Line → Plane → Space → Hypersphere

핵심 원리:
- 메모리 = Hypersphere 위의 점들
- 좌표 = (θ₁, θ₂, θ₃) 다이얼 각도
- 점 + 점 = 선 (1D 관계)
- 선 + 선 = 면 (2D 클러스터)
- 면 + 면 = 공간 (3D 도메인)
- 공간 + 공간 = 초공간 (4D 맥락)

"면 개수가 무한대 → 구체 → 연속 메모리 공간"
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum
import math

logger = logging.getLogger("HypersphereMemory")


class TopologyLevel(Enum):
    """Dimensional level of a topological structure."""
    POINT = 0    # 단일 개념
    LINE = 1     # 두 점 연결 (관계)
    PLANE = 2    # 여러 선 연결 (클러스터)
    SPACE = 3    # 여러 면 연결 (도메인)
    HYPER = 4    # 전체 맥락 (하이퍼스피어)


@dataclass
class HypersphericalCoord:
    """
    4-Dial coordinate system for full 4D Hypersphere.
    
    Consistent with Quaternion (w, x, y, z) representation.
    
    - θ₁ (theta1): 논리 축 (0 ~ 2π) - Logic/Analysis
    - θ₂ (theta2): 감정 축 (0 ~ 2π) - Emotion/Feeling  
    - θ₃ (theta3): 의도 축 (0 ~ 2π) - Intent/Action
    - r (radius): 깊이 축 (0 ~ 1) - Depth/Abstraction
        - r = 1: 구체 표면 (구체적 개념)
        - r = 0: 중심 (근원/사랑)
    """
    theta1: float = 0.0  # 0 ~ 2π (논리)
    theta2: float = 0.0  # 0 ~ 2π (감정)
    theta3: float = 0.0  # 0 ~ 2π (의도)
    radius: float = 1.0  # 0 ~ 1 (깊이/추상화)
    
    def to_quaternion(self) -> Tuple[float, float, float, float]:
        """Convert to Quaternion (w, x, y, z)."""
        # 4D spherical to Cartesian conversion
        # With radius scaling for depth
        r = self.radius
        sin_t3 = math.sin(self.theta3)
        cos_t3 = math.cos(self.theta3)
        sin_t2 = math.sin(self.theta2)
        cos_t2 = math.cos(self.theta2)
        sin_t1 = math.sin(self.theta1)
        cos_t1 = math.cos(self.theta1)
        
        # Hyperspherical to Cartesian
        w = r * cos_t1
        x = r * sin_t1 * cos_t2
        y = r * sin_t1 * sin_t2 * cos_t3
        z = r * sin_t1 * sin_t2 * sin_t3
        
        return (w, x, y, z)
    
    def to_3d(self) -> Tuple[float, float, float]:
        """Project to 3D via stereographic projection."""
        w, x, y, z = self.to_quaternion()
        denom = 1 + w if abs(1 + w) > 1e-6 else 1e-6
        return (2*x/denom, 2*y/denom, 2*z/denom)
    
    def distance_to(self, other: 'HypersphericalCoord') -> float:
        """4D Euclidean distance (includes depth)."""
        c1 = self.to_quaternion()
        c2 = other.to_quaternion()
        return math.sqrt(sum((a - b)**2 for a, b in zip(c1, c2)))
    
    def rotate(self, delta1: float = 0, delta2: float = 0, 
               delta3: float = 0, delta_r: float = 0) -> 'HypersphericalCoord':
        """Rotate by 4-dial deltas."""
        return HypersphericalCoord(
            theta1=(self.theta1 + delta1) % (2 * math.pi),
            theta2=(self.theta2 + delta2) % (2 * math.pi),
            theta3=(self.theta3 + delta3) % (2 * math.pi),
            radius=max(0.0, min(1.0, self.radius + delta_r))  # Clamp to [0, 1]
        )
    
    def zoom_in(self, amount: float = 0.1) -> 'HypersphericalCoord':
        """Move toward center (more abstract)."""
        return self.rotate(delta_r=-amount)
    
    def zoom_out(self, amount: float = 0.1) -> 'HypersphericalCoord':
        """Move toward surface (more concrete)."""
        return self.rotate(delta_r=amount)
    
    def __repr__(self):
        return f"(θ₁={self.theta1:.2f}, θ₂={self.theta2:.2f}, θ₃={self.theta3:.2f}, r={self.radius:.2f})"


# Alias for backward compatibility
SphericalCoord = HypersphericalCoord


@dataclass
class PhaseTrajectory:
    """
    Encodes "Flow" or "Time-evolving" memory.
    
    Uses Conductor's Formula: f(t) = e^{i(ωt + φ)}
    
    Instead of a fixed point, a node follows a path in 4D space.
    """
    start_coord: HypersphericalCoord
    # Angular frequencies for (θ₁, θ₂, θ₃, r)
    omega: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    duration: float = 1.0  # Duration of the memory flow
    
    def get_at(self, t: float) -> HypersphericalCoord:
        """Calculate coordinate at time t (0 to duration)."""
        # Linear frequency shift
        t_norm = max(0.0, min(self.duration, t))
        
        return HypersphericalCoord(
            theta1=(self.start_coord.theta1 + self.omega[0] * t_norm) % (2 * math.pi),
            theta2=(self.start_coord.theta2 + self.omega[1] * t_norm) % (2 * math.pi),
            theta3=(self.start_coord.theta3 + self.omega[2] * t_norm) % (2 * math.pi),
            radius=max(0.0, min(1.0, self.start_coord.radius + self.omega[3] * t_norm))
        )


@dataclass
class TopologicalNode:
    """
    A node that can exist at any topological level.
    Supports Phase-Sync for temporal data.
    """
    name: str
    coord: HypersphericalCoord
    content: Any = None
    
    # Topological connections
    connections: Set[str] = field(default_factory=set)
    
    # Phase pattern (for wave storage & Channeling)
    phase: float = 0.0
    amplitude: float = 1.0
    frequency: float = 432.0
    
    # [Phase 35] Temporal Flow
    trajectory: Optional[PhaseTrajectory] = None
    
    @property
    def level(self) -> TopologyLevel:
        """Determine topological level based on connections."""
        n = len(self.connections)
        if n == 0:
            return TopologyLevel.POINT
        elif n == 1:
            return TopologyLevel.LINE
        elif n <= 3:
            return TopologyLevel.PLANE
        elif n <= 6:
            return TopologyLevel.SPACE
        else:
            return TopologyLevel.HYPER
    
    def get_coord_at(self, t: float) -> HypersphericalCoord:
        """Get position at time t if it's a flow node, else return fixed coord."""
        if self.trajectory:
            return self.trajectory.get_at(t)
        return self.coord

    def connect(self, other_name: str):
        """Form a line with another node."""
        self.connections.add(other_name)
    
    def disconnect(self, other_name: str):
        """Remove connection."""
        self.connections.discard(other_name)
    
    def get_wave_state(self, t: float = 0) -> complex:
        """
        Return complex wave representation at time t.
        Phase Sync logic: φ(t) = φ_init + ω*t
        """
        omega = self.frequency * 2 * math.pi
        return self.amplitude * np.exp(1j * (self.phase + omega * t))


class HypersphereMemory:
    """
    Memory system using Hypersphere coordinates.
    
    "면 개수가 무한대 → 구체 → 연속 메모리 공간"
    
    Key Operations:
    - access(θ₁, θ₂, θ₃): Access point by dial coordinates
    - rotate(Δθ₁, Δθ₂, Δθ₃): Navigate through memory space
    - connect(a, b): Form line between two points
    - cluster(points): Form plane from multiple points
    """
    
    def __init__(self, resolution: int = 360):
        """
        Args:
            resolution: Number of dial "ticks" (higher = more precise)
        """
        self.resolution = resolution
        self.nodes: Dict[str, TopologicalNode] = {}
        self.current_position = HypersphericalCoord(0, 0, 0, 1.0)
        
        logger.info(f"🔮 HypersphereMemory initialized (resolution={resolution})")
    
    # =========================================
    # Core Access
    # =========================================
    
    def deposit(self, name: str, theta1: float, theta2: float, theta3: float,
                radius: float = 1.0, content: Any = None, phase: float = 0.0) -> TopologicalNode:
        """Store a node at specified dial coordinates."""
        coord = HypersphericalCoord(theta1, theta2, theta3, radius)
        node = TopologicalNode(
            name=name, 
            coord=coord, 
            content=content,
            phase=phase
        )
        self.nodes[name] = node
        logger.info(f"📍 Deposited '{name}' at {coord}")
        return node
    
    def record_flow(self, name: str, start_coord: HypersphericalCoord, 
                    omega: Tuple[float, float, float, float], duration: float = 1.0,
                    content: Any = None) -> TopologicalNode:
        """
        [Phase 35] Record a Dynamic Flow (Video/Event).
        Registers a node with a PhaseTrajectory.
        """
        trajectory = PhaseTrajectory(start_coord, omega, duration)
        node = TopologicalNode(
            name=name,
            coord=start_coord,
            trajectory=trajectory,
            content=content
        )
        self.nodes[name] = node
        logger.info(f"🌊 Recorded Flow '{name}' with duration {duration}s")
        return node

    def access(self, theta1: float, theta2: float, theta3: float, 
               radius: float = 1.0, k: int = 1, t: float = 0.0) -> List[TopologicalNode]:
        """
        Access nodes near dial coordinates at time t.
        Supports both static and dynamic flow nodes.
        """
        target = HypersphericalCoord(theta1, theta2, theta3, radius)
        
        distances = []
        for name, node in self.nodes.items():
            current_coord = node.get_coord_at(t)
            dist = target.distance_to(current_coord)
            distances.append((dist, node))
        
        distances.sort(key=lambda x: x[0])
        return [node for _, node in distances[:k]]
    
    def navigate(self, delta1: float = 0, delta2: float = 0, 
                 delta3: float = 0, delta_r: float = 0):
        """Rotate current position (dial navigation)."""
        self.current_position = self.current_position.rotate(delta1, delta2, delta3, delta_r)
        logger.info(f"🧭 Navigated to {self.current_position}")
        return self.current_position
    
    def get_nearby(self, k: int = 5, t: float = 0.0) -> List[TopologicalNode]:
        """Get nodes near current position at time t."""
        return self.access(
            self.current_position.theta1,
            self.current_position.theta2,
            self.current_position.theta3,
            self.current_position.radius,
            k=k,
            t=t
        )
    
    def resonance_query(self, theta1: float, theta2: float, theta3: float, 
                        radius: float = 1.0, target_phase: float = 0.0, 
                        tolerance: float = 0.1) -> List[TopologicalNode]:
        """
        [Phase 35] Phase Channeling Query.
        Filters nearby nodes by their phase state, allowing overlapping coordinates
        to store distinct data in different "phase channels".
        """
        candidates = self.access(theta1, theta2, theta3, radius, k=10)
        results = []
        for node in candidates:
            # Check if node phase matches target phase
            phase_diff = abs(node.phase - target_phase) % (2 * math.pi)
            if phase_diff < tolerance or phase_diff > (2 * math.pi - tolerance):
                results.append(node)
        return results
    
    # =========================================
    # Topological Operations
    # =========================================
    
    def connect(self, name1: str, name2: str) -> Optional[Tuple[TopologicalNode, TopologicalNode]]:
        """
        Connect two points → Form a LINE (1D structure).
        """
        if name1 not in self.nodes or name2 not in self.nodes:
            logger.warning(f"⚠️ Cannot connect: {name1} or {name2} not found")
            return None
        
        node1 = self.nodes[name1]
        node2 = self.nodes[name2]
        
        node1.connect(name2)
        node2.connect(name1)
        
        logger.info(f"🔗 Connected: {name1} ↔ {name2} (LINE formed)")
        return (node1, node2)
    
    def cluster(self, names: List[str], cluster_name: str = None) -> Optional[TopologicalNode]:
        """
        Connect multiple points → Form a PLANE (2D structure).
        Optionally create a centroid node.
        """
        if len(names) < 2:
            return None
        
        # Connect all pairs
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                self.connect(name1, name2)
        
        # Create centroid if requested
        if cluster_name:
            avg_t1 = np.mean([self.nodes[n].coord.theta1 for n in names if n in self.nodes])
            avg_t2 = np.mean([self.nodes[n].coord.theta2 for n in names if n in self.nodes])
            avg_t3 = np.mean([self.nodes[n].coord.theta3 for n in names if n in self.nodes])
            
            centroid = self.deposit(cluster_name, avg_t1, avg_t2, avg_t3)
            for name in names:
                self.connect(cluster_name, name)
            
            logger.info(f"🔷 Cluster '{cluster_name}' created with {len(names)} members")
            return centroid
        
        logger.info(f"🔷 PLANE formed: {names}")
        return None
    
    def get_structure(self, name: str) -> Dict[str, Any]:
        """Get the topological structure around a node."""
        if name not in self.nodes:
            return {}
        
        node = self.nodes[name]
        connections = []
        
        for conn_name in node.connections:
            if conn_name in self.nodes:
                conn_node = self.nodes[conn_name]
                connections.append({
                    "name": conn_name,
                    "distance": node.coord.distance_to(conn_node.coord),
                    "level": conn_node.level.name
                })
        
        return {
            "name": name,
            "coord": (node.coord.theta1, node.coord.theta2, node.coord.theta3),
            "level": node.level.name,
            "connections": connections
        }
    
    # =========================================
    # Wave/Phase Operations
    # =========================================
    
    def superpose(self, names: List[str]) -> complex:
        """
        Superpose wave states of multiple nodes.
        Returns combined complex amplitude.
        """
        total = 0j
        for name in names:
            if name in self.nodes:
                total += self.nodes[name].get_wave_state()
        return total
    
    def measure_phase_difference(self, name1: str, name2: str) -> float:
        """Phase difference between two nodes (encodes relationship)."""
        if name1 not in self.nodes or name2 not in self.nodes:
            return 0.0
        
        phase1 = self.nodes[name1].phase
        phase2 = self.nodes[name2].phase
        
        diff = (phase1 - phase2) % (2 * math.pi)
        return diff
    
    # =========================================
    # Utility
    # =========================================
    
    def get_stats(self) -> Dict[str, Any]:
        level_counts = {level.name: 0 for level in TopologyLevel}
        for node in self.nodes.values():
            level_counts[node.level.name] += 1
        
        return {
            "total_nodes": len(self.nodes),
            "topology": level_counts,
            "current_position": (
                self.current_position.theta1,
                self.current_position.theta2,
                self.current_position.theta3
            )
        }
    
    def to_dict(self) -> Dict:
        """Serialize for storage."""
        return {
            "resolution": self.resolution,
            "nodes": {
                name: {
                    "theta1": n.coord.theta1,
                    "theta2": n.coord.theta2,
                    "theta3": n.coord.theta3,
                    "content": str(n.content) if n.content else None,
                    "phase": n.phase,
                    "connections": list(n.connections)
                }
                for name, n in self.nodes.items()
            }
        }


# =============================================================================
# Phase 34: Meta-Hypersphere (Spheres as Nodes)
# =============================================================================

@dataclass
class HypersphereNode:
    """
    A Hypersphere that acts as a Node in a larger meta-structure.
    
    "모든 지식을 담아내기 위한 거시적 구조"
    
    - Each sphere contains its own internal nodes (TopologicalNodes)
    - Spheres connect to other spheres (domain-to-domain relationships)
    - Infinite scalability: spheres can nest within spheres
    
    Example:
    - Sphere "Animals" contains {Dog, Cat, Lion, ...}
    - Sphere "Plants" contains {Tree, Flower, Grass, ...}
    - Animals ↔ Plants = Ecological relationship
    """
    name: str
    description: str = ""
    
    # Internal memory (contents of this sphere)
    memory: HypersphereMemory = field(default_factory=HypersphereMemory)
    
    # Meta-connections (links to other spheres)
    connections: Set[str] = field(default_factory=set)
    
    # Position in meta-space (where this sphere exists among other spheres)
    meta_coord: HypersphericalCoord = field(default_factory=HypersphericalCoord)
    
    # Sphere properties
    resonance_frequency: float = 432.0  # Characteristic frequency of this domain
    mass: float = 1.0  # Importance/gravity in meta-space
    
    def add_node(self, name: str, theta1: float, theta2: float, 
                 theta3: float, radius: float = 1.0, content: Any = None):
        """Add a node inside this sphere."""
        return self.memory.deposit(name, theta1, theta2, theta3, content=content)
    
    def connect_to(self, other_name: str):
        """Connect to another sphere."""
        self.connections.add(other_name)
    
    def get_internal_count(self) -> int:
        """How many nodes inside this sphere."""
        return len(self.memory.nodes)
    
    @property
    def topology_level(self) -> TopologyLevel:
        """Level based on connections to other spheres."""
        n = len(self.connections)
        if n == 0:
            return TopologyLevel.POINT
        elif n == 1:
            return TopologyLevel.LINE
        elif n <= 3:
            return TopologyLevel.PLANE
        elif n <= 6:
            return TopologyLevel.SPACE
        else:
            return TopologyLevel.HYPER


class MetaHypersphere:
    """
    The Universe of Spheres - 메타 하이퍼스피어
    
    "하이퍼스피어를 노드화해서 모든 지식을 담는다"
    
    Structure:
    - Each HypersphereNode = One domain (e.g., "Animals", "Physics", "Emotions")
    - Connections between spheres = Cross-domain relationships
    - Navigating between spheres = Changing context/domain
    
    Example:
        meta = MetaHypersphere()
        meta.create_sphere("Animals")
        meta.create_sphere("Plants")
        meta.create_sphere("Ecology")
        
        meta.spheres["Animals"].add_node("Dog", ...)
        meta.spheres["Plants"].add_node("Tree", ...)
        
        meta.connect_spheres("Animals", "Plants")  # Ecological relationship
    """
    
    def __init__(self):
        self.spheres: Dict[str, HypersphereNode] = {}
        self.current_sphere: Optional[str] = None
        logger.info("🌌 MetaHypersphere initialized - The Universe of All Knowledge")
    
    def create_sphere(self, name: str, description: str = "",
                      theta1: float = 0, theta2: float = 0, 
                      theta3: float = 0, radius: float = 1.0,
                      frequency: float = 432.0) -> HypersphereNode:
        """Create a new domain sphere."""
        coord = HypersphericalCoord(theta1, theta2, theta3, radius)
        sphere = HypersphereNode(
            name=name,
            description=description,
            meta_coord=coord,
            resonance_frequency=frequency
        )
        self.spheres[name] = sphere
        logger.info(f"🔮 Created Sphere '{name}' at {coord}")
        return sphere
    
    def connect_spheres(self, name1: str, name2: str) -> bool:
        """Connect two spheres (cross-domain relationship)."""
        if name1 not in self.spheres or name2 not in self.spheres:
            logger.warning(f"⚠️ Cannot connect: {name1} or {name2} not found")
            return False
        
        self.spheres[name1].connect_to(name2)
        self.spheres[name2].connect_to(name1)
        logger.info(f"🔗 Spheres connected: {name1} ↔ {name2}")
        return True
    
    def enter_sphere(self, name: str) -> Optional[HypersphereMemory]:
        """Enter a sphere to access its internal memory."""
        if name not in self.spheres:
            logger.warning(f"⚠️ Sphere '{name}' not found")
            return None
        
        self.current_sphere = name
        logger.info(f"🚀 Entered Sphere '{name}'")
        return self.spheres[name].memory
    
    def exit_sphere(self):
        """Exit current sphere back to meta-level."""
        old = self.current_sphere
        self.current_sphere = None
        logger.info(f"🚀 Exited Sphere '{old}' → Meta-level")
    
    def find_nearest_sphere(self, theta1: float, theta2: float, 
                            theta3: float, radius: float = 1.0) -> Optional[HypersphereNode]:
        """Find nearest sphere to given meta-coordinates."""
        target = HypersphericalCoord(theta1, theta2, theta3, radius)
        
        best_sphere = None
        best_dist = float('inf')
        
        for sphere in self.spheres.values():
            dist = target.distance_to(sphere.meta_coord)
            if dist < best_dist:
                best_dist = dist
                best_sphere = sphere
        
        return best_sphere
    
    def get_connected_spheres(self, name: str) -> List[HypersphereNode]:
        """Get all spheres connected to the given sphere."""
        if name not in self.spheres:
            return []
        
        connections = self.spheres[name].connections
        return [self.spheres[n] for n in connections if n in self.spheres]
    
    def get_universe_stats(self) -> Dict[str, Any]:
        """Statistics about the entire meta-hypersphere."""
        total_nodes = sum(s.get_internal_count() for s in self.spheres.values())
        total_connections = sum(len(s.connections) for s in self.spheres.values()) // 2
        
        return {
            "total_spheres": len(self.spheres),
            "total_nodes_inside": total_nodes,
            "total_connections": total_connections,
            "current_sphere": self.current_sphere,
            "spheres": {
                name: {
                    "internal_nodes": s.get_internal_count(),
                    "connections": len(s.connections),
                    "level": s.topology_level.name
                }
                for name, s in self.spheres.items()
            }
        }
    
    def to_dict(self) -> Dict:
        """Serialize the entire universe."""
        return {
            "spheres": {
                name: {
                    "description": s.description,
                    "meta_coord": (s.meta_coord.theta1, s.meta_coord.theta2, 
                                   s.meta_coord.theta3, s.meta_coord.radius),
                    "frequency": s.resonance_frequency,
                    "connections": list(s.connections),
                    "internal": s.memory.to_dict()
                }
                for name, s in self.spheres.items()
            }
        }


# Singleton
_memory = None

def get_hypersphere_memory() -> HypersphereMemory:
    global _memory
    if _memory is None:
        _memory = HypersphereMemory()
    return _memory


# Demo
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("🔮 HYPERSPHERE TOPOLOGICAL MEMORY DEMO")
    print("=" * 70)
    
    mem = HypersphereMemory(resolution=360)
    
    # 1. Deposit points
    print("\n[1] Depositing Points...")
    mem.deposit("Dog", theta1=0.5, theta2=1.0, theta3=0.8)
    mem.deposit("Wolf", theta1=0.6, theta2=1.1, theta3=0.8)
    mem.deposit("Cat", theta1=0.5, theta2=2.0, theta3=0.8)
    mem.deposit("Lion", theta1=0.55, theta2=2.1, theta3=0.85)
    
    # 2. Connect points → Lines
    print("\n[2] Connecting Points (Lines)...")
    mem.connect("Dog", "Wolf")  # Similar canines
    mem.connect("Cat", "Lion")  # Similar felines
    
    # 3. Form cluster → Plane
    print("\n[3] Forming Cluster (Plane)...")
    mem.cluster(["Dog", "Wolf", "Cat", "Lion"], cluster_name="Mammals")
    
    # 4. Check structure
    print("\n[4] Structure Analysis:")
    for name in ["Dog", "Wolf", "Mammals"]:
        struct = mem.get_structure(name)
        print(f"   {name}: Level={struct['level']}, Connections={len(struct['connections'])}")
    
    # 5. Navigate
    print("\n[5] Navigation Test:")
    mem.navigate(delta1=0.1, delta2=0.05)
    nearby = mem.get_nearby(k=2)
    print(f"   Nearby nodes: {[n.name for n in nearby]}")
    
    # 6. Stats
    print("\n[6] Final Stats:")
    stats = mem.get_stats()
    print(f"   Total: {stats['total_nodes']} nodes")
    print(f"   Topology: {stats['topology']}")
    
    print("\n" + "=" * 70)
    print("✅ Hypersphere Memory Demo Complete!")
    print("   점 → 선 → 면 → 공간 → 하이퍼스피어")
    print("=" * 70)

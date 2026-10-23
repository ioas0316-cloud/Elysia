"""
Elysia Core Physics: Decentralized High-Dimensional Kuramoto Phase Synchronization Engine
=======================================================================================
Implements decentralized global phase synchronization on Clifford manifolds Cl(3,0)
without a central master clock, driven purely by local peer-to-peer neighborhood torque
minimization, autopoietic fault isolation (phi_crit edge severing), Meta-Rotor background
curvature gravitational self-healing, and SPDK zero-copy live world stream locking.

Trinity Architecture Pillars:
1. Low-Latency CXL / RDMA Shared Memory Simulation (Cacheline aligned 1-sided atomic wedge mismatch).
2. Meta-Rotor Background Curvature & Manifold Gravity (Autopoietic reconnection of isolated nodes).
3. SPDK Live World Streaming Pipeline (Zero-copy byte stream to unit sphere wave O(t) phase lock).
"""

from typing import List, Dict, Tuple, Any, Optional
import math
import random


class Vector3:
    """3D Clifford 1-Vector (x, y, z) with Euclidean norm and cross/dot products."""
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> 'Vector3':
        len_val = self.length()
        if len_val > 1e-12:
            return Vector3(self.x / len_val, self.y / len_val, self.z / len_val)
        return Vector3(0.0, 0.0, 0.0)

    def dot(self, v: 'Vector3') -> float:
        return self.x * v.x + self.y * v.y + self.z * v.z

    def cross(self, v: 'Vector3') -> 'Vector3':
        return Vector3(
            self.y * v.z - self.z * v.y,
            self.z * v.x - self.x * v.z,
            self.x * v.y - self.y * v.x
        )

    def __add__(self, v: 'Vector3') -> 'Vector3':
        return Vector3(self.x + v.x, self.y + v.y, self.z + v.z)

    def __sub__(self, v: 'Vector3') -> 'Vector3':
        return Vector3(self.x - v.x, self.y - v.y, self.z - v.z)

    def __mul__(self, s: float) -> 'Vector3':
        return Vector3(self.x * s, self.y * s, self.z * s)

    def __rmul__(self, s: float) -> 'Vector3':
        return Vector3(self.x * s, self.y * s, self.z * s)

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


class Bivector3:
    """3D Clifford 2-Vector (xy, yz, zx) representing directed area plane of phase mismatch."""
    __slots__ = ('xy', 'yz', 'zx')

    def __init__(self, xy: float = 0.0, yz: float = 0.0, zx: float = 0.0):
        self.xy = float(xy)
        self.yz = float(yz)
        self.zx = float(zx)

    def magnitude(self) -> float:
        return math.sqrt(self.xy * self.xy + self.yz * self.yz + self.zx * self.zx)

    @staticmethod
    def wedge(a: Vector3, b: Vector3) -> 'Bivector3':
        """Outer Wedge Product: A ^ B -> Bivector (e12, e23, e31)"""
        return Bivector3(
            a.x * b.y - a.y * b.x,  # e12
            a.y * b.z - a.z * b.y,  # e23
            a.z * b.x - a.x * b.z   # e31
        )

    def __add__(self, b: 'Bivector3') -> 'Bivector3':
        return Bivector3(self.xy + b.xy, self.yz + b.yz, self.zx + b.zx)

    def __mul__(self, s: float) -> 'Bivector3':
        return Bivector3(self.xy * s, self.yz * s, self.zx * s)


class Rotor3:
    """3D Clifford Rotor R = s + B for Lie group geodesic manifold rotations."""
    __slots__ = ('s', 'bivector')

    def __init__(self, scalar: float = 1.0, bivector: Optional[Bivector3] = None):
        self.s = float(scalar)
        self.bivector = bivector if bivector is not None else Bivector3(0.0, 0.0, 0.0)

    @classmethod
    def from_incremental_bivector(cls, dB: Bivector3) -> 'Rotor3':
        """R = exp(-0.5 * dB)"""
        phi = dB.magnitude()
        if phi < 1e-12:
            return cls(1.0, Bivector3(0.0, 0.0, 0.0))

        half_angle = 0.5 * phi
        scale = -math.sin(half_angle) / phi
        return cls(
            math.cos(half_angle),
            Bivector3(dB.xy * scale, dB.yz * scale, dB.zx * scale)
        )

    def rotate(self, v: Vector3) -> Vector3:
        """Apply Rotor to Vector: v' = R v R^dagger"""
        q_vec = Vector3(-self.bivector.yz, -self.bivector.zx, -self.bivector.xy)
        q_cross_v = q_vec.cross(v)
        inner = Vector3(q_cross_v.x + self.s * v.x, q_cross_v.y + self.s * v.y, q_cross_v.z + self.s * v.z)
        outer = q_vec.cross(inner)

        return Vector3(
            v.x + 2.0 * outer.x,
            v.y + 2.0 * outer.y,
            v.z + 2.0 * outer.z
        )


class KuramotoNode:
    """Individual Agent Node in Decentralized Kuramoto Network."""

    def __init__(self, node_id: int, initial_psi: Vector3):
        self.id = node_id
        self.psi = initial_psi.normalized()
        self.is_rogue: bool = False
        self.is_dead: bool = False
        self.is_isolated: bool = False
        self.isolation_timer: float = 0.0


class DecentralizedKuramotoEngine:
    """
    Decentralized High-Dimensional Kuramoto Phase Synchronization Engine.
    Handles P2P local torque minimization, autopoietic edge cutting,
    Meta-Rotor gravitational reconnection, and live streaming wave locking.
    """

    def __init__(
        self,
        num_nodes: int = 5,
        coupling_gain_k: float = 10.0,
        phi_crit: float = 0.85,
        gamma_meta: float = 1.5,
        seed: int = 42
    ):
        self.num_nodes = num_nodes
        self.coupling_gain_k = float(coupling_gain_k)
        self.phi_crit = float(phi_crit)
        self.gamma_meta = float(gamma_meta)

        self.rng = random.Random(seed)
        self.nodes: List[KuramotoNode] = []
        self.adj_matrix: List[List[float]] = [
            [1.0 if i != j else 0.0 for j in range(num_nodes)]
            for i in range(num_nodes)
        ]

        # Initialize nodes with random unit vectors on S^2
        for i in range(num_nodes):
            x = self.rng.uniform(-1.0, 1.0)
            y = self.rng.uniform(-1.0, 1.0)
            z = self.rng.uniform(-1.0, 1.0)
            self.nodes.append(KuramotoNode(i, Vector3(x, y, z)))

    def inject_rogue_attack(self, node_id: int):
        """Node begins injecting chaotic phase noise."""
        if 0 <= node_id < self.num_nodes:
            self.nodes[node_id].is_rogue = True

    def heal_node(self, node_id: int):
        """Restores a rogue node back to normal operation."""
        if 0 <= node_id < self.num_nodes:
            self.nodes[node_id].is_rogue = False

    def kill_node(self, node_id: int):
        """Abruptly crashes a node, severing all topological links."""
        if 0 <= node_id < self.num_nodes:
            self.nodes[node_id].is_dead = True
            for j in range(self.num_nodes):
                self.adj_matrix[node_id][j] = 0.0
                self.adj_matrix[j][node_id] = 0.0

    def compute_global_coherence(self) -> float:
        """Global Coherence Order Parameter ||Z(t)|| = 1/N * ||sum Psi_i||."""
        sum_psi = Vector3(0.0, 0.0, 0.0)
        active_count = 0
        for node in self.nodes:
            if not node.is_dead and not node.is_rogue:
                sum_psi = sum_psi + node.psi
                active_count += 1
        return (sum_psi.length() / active_count) if active_count > 0 else 0.0

    def get_global_center_vector(self) -> Vector3:
        """Returns the normalized vector sum of all healthy active nodes Z(t)."""
        sum_psi = Vector3(0.0, 0.0, 0.0)
        active_count = 0
        for node in self.nodes:
            if not node.is_dead and not node.is_rogue:
                sum_psi = sum_psi + node.psi
                active_count += 1
        return sum_psi.normalized() if active_count > 0 else Vector3(1.0, 0.0, 0.0)

    def get_active_edge_count(self) -> int:
        """Counts remaining active symmetric edges in topology."""
        count = 0
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.adj_matrix[i][j] > 0.0:
                    count += 1
        return count

    def project_stream_bytes_to_sphere(self, packet_bytes: bytes) -> Vector3:
        """Zero-Copy Stream Vectorization: Hash Byte Stream onto Unit Sphere S^2 via FNV-1a."""
        hash_val = 14695981039346656037
        for b in packet_bytes:
            hash_val ^= b
            hash_val = (hash_val * 1099511628211) & 0xFFFFFFFFFFFFFFFF

        theta = (hash_val & 0xFFFF) / 65535.0 * 2.0 * math.pi
        phi = ((hash_val >> 16) & 0xFFFF) / 65535.0 * math.pi

        return Vector3(
            math.sin(phi) * math.cos(theta),
            math.sin(phi) * math.sin(theta),
            math.cos(phi)
        )

    def process_live_stream_packet(self, node_id: int, packet_bytes: bytes, dt: float = 0.02):
        """SPDK Live World Stream Lock: Ingests raw packet and continuously phase locks node_id."""
        if 0 <= node_id < self.num_nodes and not self.nodes[node_id].is_dead:
            node = self.nodes[node_id]
            o_wave = self.project_stream_bytes_to_sphere(packet_bytes)

            # Wedge product interference Delta_B = Psi ^ O
            delta_b = Bivector3.wedge(node.psi, o_wave)

            # Continuous rotor update
            dB = Bivector3(
                self.coupling_gain_k * delta_b.xy * dt,
                self.coupling_gain_k * delta_b.yz * dt,
                self.coupling_gain_k * delta_b.zx * dt
            )
            rotor = Rotor3.from_incremental_bivector(dB)
            node.psi = rotor.rotate(node.psi).normalized()

    def step(self, dt: float = 0.02):
        """Executes 1 cycle of decentralized phase loop and autopoietic fault management."""
        n = self.num_nodes
        next_states = [node.psi for node in self.nodes]
        z_center = self.get_global_center_vector()

        # 1. Local Phase Evaluation & Autopoietic Isolation
        for i in range(n):
            node = self.nodes[i]
            if node.is_dead:
                continue

            if node.is_rogue:
                # Rogue node emits chaotic random phase noise
                rx = self.rng.uniform(-2.0, 2.0)
                ry = self.rng.uniform(-2.0, 2.0)
                rz = self.rng.uniform(-2.0, 2.0)
                node.psi = Vector3(rx, ry, rz).normalized()
                next_states[i] = node.psi
                continue

            total_torque = Bivector3(0.0, 0.0, 0.0)
            active_neighbors = 0.0

            for j in range(n):
                if i == j or self.nodes[j].is_dead or self.adj_matrix[i][j] <= 0.0:
                    continue

                wedge = Bivector3.wedge(node.psi, self.nodes[j].psi)
                mismatch_mag = wedge.magnitude()

                # Autopoietic Edge Severing: Isolate neighbors exceeding phi_crit
                if mismatch_mag > self.phi_crit:
                    self.adj_matrix[i][j] = 0.0
                    self.adj_matrix[j][i] = 0.0
                    continue

                total_torque.xy += self.adj_matrix[i][j] * wedge.xy
                total_torque.yz += self.adj_matrix[i][j] * wedge.yz
                total_torque.zx += self.adj_matrix[i][j] * wedge.zx
                active_neighbors += self.adj_matrix[i][j]

            meta_wedge = Bivector3.wedge(node.psi, z_center)
            meta_torque_xy = self.gamma_meta * meta_wedge.xy
            meta_torque_yz = self.gamma_meta * meta_wedge.yz
            meta_torque_zx = self.gamma_meta * meta_wedge.zx

            if active_neighbors > 0.0:
                node.is_isolated = False
                tot_xy = (self.coupling_gain_k / active_neighbors) * total_torque.xy + meta_torque_xy
                tot_yz = (self.coupling_gain_k / active_neighbors) * total_torque.yz + meta_torque_yz
                tot_zx = (self.coupling_gain_k / active_neighbors) * total_torque.zx + meta_torque_zx
            else:
                node.is_isolated = True
                tot_xy = meta_torque_xy
                tot_yz = meta_torque_yz
                tot_zx = meta_torque_zx

            dB = Bivector3(tot_xy * dt, tot_yz * dt, tot_zx * dt)
            correction = Rotor3.from_incremental_bivector(dB)
            next_states[i] = correction.rotate(node.psi).normalized()

            # Autopoietic Re-connection if alignment error with healthy center drops below phi_crit
            for j in range(n):
                if i != j and not self.nodes[j].is_dead and not self.nodes[j].is_rogue:
                    w_check = Bivector3.wedge(node.psi, self.nodes[j].psi)
                    if w_check.magnitude() < self.phi_crit * 0.9:
                        self.adj_matrix[i][j] = 1.0
                        self.adj_matrix[j][i] = 1.0

        # Apply state updates
        for i in range(n):
            if not self.nodes[i].is_dead:
                self.nodes[i].psi = next_states[i]

    def get_topology_summary(self) -> Dict[str, Any]:
        """Returns diagnostic state of the decentralized network."""
        return {
            "num_nodes": self.num_nodes,
            "global_coherence_z": self.compute_global_coherence(),
            "active_edges": self.get_active_edge_count(),
            "nodes_status": [
                {
                    "id": node.id,
                    "psi": node.psi.to_tuple(),
                    "is_rogue": node.is_rogue,
                    "is_dead": node.is_dead,
                    "is_isolated": node.is_isolated
                }
                for node in self.nodes
            ]
        }

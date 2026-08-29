import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from core.physics.symplectic_causal_engine import SymplecticCausalEngine
from core.physics.causal_field import CausalField, InformationVoxel

class SteinerCausalNetwork:
    """
    [Dynamic Steiner Causal Network]
    Maintains a topology of terminal nodes and dynamically added Steiner points.
    Calculates total system energy (wire length tension + task potential) and gradients.

    Uses differentiable continuous relaxation:
    L_total = L_task + lambda * sum_{(i,j)} w_{ij} * ||z_i - z_j||_2

    Spontaneously evolves Steiner points towards minimum action locations (e.g. 120-degree angles).
    """
    def __init__(
        self,
        terminals: List[np.ndarray],
        num_steiner_points: int = 2,
        lambda_length: float = 1.0,
        mass: float = 1.0,
        dt: float = 0.02,
        gamma: float = 0.2
    ):
        self.terminals = [np.array(t, dtype=np.float32) for t in terminals]
        self.num_terminals = len(self.terminals)
        self.num_steiner = num_steiner_points
        self.lambda_length = lambda_length
        self.dim = len(self.terminals[0]) if self.terminals else 2

        # Initialize Steiner points near center of mass with small perturbations
        center = np.mean(self.terminals, axis=0) if self.terminals else np.zeros(self.dim, dtype=np.float32)
        np.random.seed(42)
        self.steiner_points = [
            center + np.random.normal(scale=0.1, size=self.dim).astype(np.float32)
            for _ in range(self.num_steiner)
        ]

        # Momentum for Steiner points
        self.steiner_momenta = [np.zeros(self.dim, dtype=np.float32) for _ in range(self.num_steiner)]

        # Symplectic Engine
        self.engine = SymplecticCausalEngine(mass=mass, dt_initial=dt, gamma_initial=gamma)

        # Default adjacency topology for 4-point Steiner tree:
        # If 4 terminals on unit square:
        # Steiner 0 connects to Terminals 0, 1 and Steiner 1
        # Steiner 1 connects to Terminals 2, 3 and Steiner 0
        self.topology_edges = self._build_default_topology()

    def _build_default_topology(self) -> List[Tuple[int, int, str, str]]:
        """
        Builds edge connectivity between terminals ('T') and Steiner points ('S').
        Returns list of (idx1, idx2, type1, type2)
        """
        edges = []
        if self.num_terminals == 4 and self.num_steiner == 2:
            # Sort terminals by position or angle to form natural clusters
            edges.append((0, 0, 'T', 'S')) # T0 - S0
            edges.append((1, 0, 'T', 'S')) # T1 - S0
            edges.append((0, 1, 'S', 'S')) # S0 - S1
            edges.append((2, 1, 'T', 'S')) # T2 - S1
            edges.append((3, 1, 'T', 'S')) # T3 - S1
        else:
            # Connect each Steiner point to subsets of terminals and other Steiner points
            for t_idx, t in enumerate(self.terminals):
                s_idx = t_idx % self.num_steiner
                edges.append((t_idx, s_idx, 'T', 'S'))
            for s_idx in range(self.num_steiner - 1):
                edges.append((s_idx, s_idx + 1, 'S', 'S'))
        return edges

    def get_point(self, idx: int, p_type: str) -> np.ndarray:
        if p_type == 'T':
            return self.terminals[idx]
        elif p_type == 'S':
            return self.steiner_points[idx]
        raise ValueError(f"Unknown point type {p_type}")

    def calculate_total_length(self) -> float:
        """Calculates total Euclidean length of connected edges in current topology."""
        length = 0.0
        for idx1, idx2, type1, type2 in self.topology_edges:
            p1 = self.get_point(idx1, type1)
            p2 = self.get_point(idx2, type2)
            length += float(np.linalg.norm(p1 - p2))
        return length

    def calculate_potential(self, steiner_flat: np.ndarray) -> float:
        """
        Potential function V(z) where z is flattened array of Steiner point coordinates.
        V(z) = lambda_length * Total_Edge_Lengths + EPS_Barrier
        """
        s_coords = steiner_flat.reshape((self.num_steiner, self.dim))
        potential = 0.0
        eps = 1e-6

        for idx1, idx2, type1, type2 in self.topology_edges:
            p1 = self.get_point(idx1, type1) if type1 == 'T' else s_coords[idx1]
            p2 = self.get_point(idx2, type2) if type2 == 'T' else s_coords[idx2]
            dist = np.sqrt(np.sum((p1 - p2) ** 2) + eps)
            potential += self.lambda_length * dist

        return float(potential)

    def calculate_gradient(self, steiner_flat: np.ndarray) -> np.ndarray:
        """
        Analytical gradient dV / dz_S for each Steiner point coordinate.
        d/dz ||z - p||_2 = (z - p) / ||z - p||_2
        """
        s_coords = steiner_flat.reshape((self.num_steiner, self.dim))
        grad = np.zeros_like(s_coords)
        eps = 1e-6

        for idx1, idx2, type1, type2 in self.topology_edges:
            p1 = self.get_point(idx1, type1) if type1 == 'T' else s_coords[idx1]
            p2 = self.get_point(idx2, type2) if type2 == 'T' else s_coords[idx2]

            diff = p1 - p2
            dist = np.sqrt(np.sum(diff ** 2) + eps)
            unit_vector = diff / dist

            if type1 == 'S':
                grad[idx1] += self.lambda_length * unit_vector
            if type2 == 'S':
                grad[idx2] -= self.lambda_length * unit_vector

        return grad.flatten()

    def calculate_angles(self) -> Dict[int, List[float]]:
        """
        Calculates angles in degrees between connected edges at each Steiner point.
        For optimal Steiner tree, these angles converge to 120 degrees.
        """
        angles_dict = {}
        for s_idx in range(self.num_steiner):
            s_pos = self.steiner_points[s_idx]
            connected_vectors = []
            for idx1, idx2, type1, type2 in self.topology_edges:
                if type1 == 'S' and idx1 == s_idx:
                    other_pos = self.get_point(idx2, type2)
                    vec = other_pos - s_pos
                    connected_vectors.append(vec / (np.linalg.norm(vec) + 1e-8))
                elif type2 == 'S' and idx2 == s_idx:
                    other_pos = self.get_point(idx1, type1)
                    vec = other_pos - s_pos
                    connected_vectors.append(vec / (np.linalg.norm(vec) + 1e-8))

            angles = []
            num_vecs = len(connected_vectors)
            for i in range(num_vecs):
                for j in range(i + 1, num_vecs):
                    v1 = connected_vectors[i]
                    v2 = connected_vectors[j]
                    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
                    angle_deg = np.degrees(np.arccos(dot))
                    angles.append(float(angle_deg))
            angles_dict[s_idx] = angles
        return angles_dict

    def step(self) -> Dict[str, Any]:
        """Performs one Symplectic Verlet step on all Steiner points."""
        steiner_flat = np.array(self.steiner_points, dtype=np.float32).flatten()
        momenta_flat = np.array(self.steiner_momenta, dtype=np.float32).flatten()

        pot_fn = lambda z: self.calculate_potential(z)
        grad_fn = lambda z: self.calculate_gradient(z)

        new_steiner_flat, new_momenta_flat, info = self.engine.step(
            z=steiner_flat,
            p=momenta_flat,
            potential_fn=pot_fn,
            grad_fn=grad_fn
        )

        self.steiner_points = [
            new_steiner_flat[i*self.dim:(i+1)*self.dim] for i in range(self.num_steiner)
        ]
        self.steiner_momenta = [
            new_momenta_flat[i*self.dim:(i+1)*self.dim] for i in range(self.num_steiner)
        ]

        total_length = self.calculate_total_length()
        angles = self.calculate_angles()

        return {
            "total_length": total_length,
            "angles": angles,
            **info
        }

    def sync_to_causal_field(self, causal_field: CausalField):
        """
        Maps Steiner points and terminals into CausalField InformationVoxels and ConnectivityBeams,
        allowing Steiner points to act as continuous field generators.
        """
        # Add terminal voxels
        for t_idx, t_pos in enumerate(self.terminals):
            vid = f"terminal_{t_idx}"
            vox = InformationVoxel(
                id=vid,
                content=f"Terminal Node {t_idx}",
                tensor=t_pos.copy(),
                position=t_pos.copy(),
                mass=1.0
            )
            causal_field.add_voxel(vox)

        # Add Steiner voxels
        for s_idx, s_pos in enumerate(self.steiner_points):
            vid = f"steiner_{s_idx}"
            vox = InformationVoxel(
                id=vid,
                content=f"Steiner Node {s_idx}",
                tensor=s_pos.copy(),
                position=s_pos.copy(),
                mass=self.engine.mass
            )
            causal_field.add_voxel(vox)

        # Link according to topology
        for idx1, idx2, type1, type2 in self.topology_edges:
            id1 = f"terminal_{idx1}" if type1 == 'T' else f"steiner_{idx1}"
            id2 = f"terminal_{idx2}" if type2 == 'T' else f"steiner_{idx2}"
            causal_field.link_voxels(id1, id2, strength=self.lambda_length)

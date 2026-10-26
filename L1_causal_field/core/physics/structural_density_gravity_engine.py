import numpy as np
import heapq
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class PhaseStateSpectrum:
    r"""
    Continuous 6-Parameter Phase State Tensor Spectrum:
    [D_density, mu_mobility, K_concentration, tau_limit, T_agitation, S_entropy]

    - D_density: Structural Density of topological web / node condensation.
    - mu_mobility: Fluidity / mobility of microscopic rotors and bonds.
    - K_concentration: Local spatial focus / concentration factor (1D edge or zero-point squeezing).
    - tau_limit: Maximum tension distortion threshold before phase transition or rupture.
    - T_agitation: Kinetic heat / agitation frequency of microscopic rotors.
    - S_entropy: Dispersion / entropy measure of unbound variable clouds.
    """
    density: float = 1.0           # D_density
    mobility: float = 0.1          # mu_mobility
    concentration: float = 1.0     # K_concentration
    tension_limit: float = 2.0     # tau_limit
    agitation: float = 0.0         # T_agitation (Heat)
    entropy: float = 0.1           # S_entropy

    def classify_materialization(self) -> str:
        """Classifies current state into dominant materialization archetype."""
        if self.concentration > 5.0 or self.tension_limit < 0.5:
            return "Blade"  # Slicing / Rupture state
        if self.entropy > 2.0 or (self.agitation > 3.0 and self.mobility > 2.0):
            return "Gas_Cloud"  # Wind / Gas / Latent variable cloud
        if self.mobility > 1.0 or self.agitation > 1.5:
            return "Water"  # Fluid / Flow state
        return "Rock"  # Solid / Anchor state

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.density, self.mobility, self.concentration,
            self.tension_limit, self.agitation, self.entropy
        ], dtype=np.float32)


@dataclass
class TopologicalNode:
    """A node in the topological web with position, connections, and phase spectrum."""
    node_id: str
    position: np.ndarray  # (D,) position in causal space
    phase: PhaseStateSpectrum = field(default_factory=PhaseStateSpectrum)
    connected_edges: Dict[str, float] = field(default_factory=dict)  # target_node_id -> tension_strength
    data_payload: Any = None


@dataclass
class RuptureEvent:
    """Record of a topological rupture / insight slicing event."""
    rupture_id: str
    source_blade_id: str
    target_node_ids: List[str] = field(default_factory=list)
    tension_peak: float = 0.0
    separated_subfields: List[List[str]] = field(default_factory=list)
    cognitive_insight_extracted: Optional[List[str]] = None
    is_cognitive: bool = False


class StructuralDensityGravityEngine:
    r"""
    [Structural Density Gravity & Topological Rupture Engine]
    Implements:
    1. Mass as Structural Density (Density of Topological Web).
    2. Gravity as Tension Gradient Slope & SDF Geodesic Flow.
    3. Spectral propagation speed (c_spectrum) dynamic balancing against tension distortion limits (tau_limit).
    4. 6-Phase Materialization Spectrum: Rock, Water, Wind/Gas, Blade.
    5. Cognitive Heat Generation: Tension gradient differential between internal intention (In-force)
       and external reality/friction (Out-force) producing kinetic agitation heat.
    6. Thermal Agitation & Condensation/Nucleation loops.
    7. Isomorphic Topological Rupture (Physical Field Slicing & Cognitive Insight Slicing).
    """

    def __init__(self, dimensions: int = 16, base_spectral_speed: float = 1.0):
        self.dimensions = dimensions
        self.base_spectral_speed = base_spectral_speed
        self.nodes: Dict[str, TopologicalNode] = {}
        self.ruptures: List[RuptureEvent] = []

    def add_node(self,
                 node_id: str,
                 position: np.ndarray,
                 phase: Optional[PhaseStateSpectrum] = None,
                 data_payload: Any = None) -> TopologicalNode:
        """Adds a topological node to the web."""
        if len(position) < self.dimensions:
            position = np.pad(position, (0, self.dimensions - len(position)))
        elif len(position) > self.dimensions:
            position = position[:self.dimensions]

        node_phase = phase if phase is not None else PhaseStateSpectrum()
        node = TopologicalNode(
            node_id=node_id,
            position=position.astype(np.float32),
            phase=node_phase,
            connected_edges={},
            data_payload=data_payload
        )
        self.nodes[node_id] = node
        return node

    def connect_nodes(self, node_id_a: str, node_id_b: str, tension_strength: float = 1.0):
        """Establishes a topological tension bond between two nodes."""
        if node_id_a in self.nodes and node_id_b in self.nodes:
            self.nodes[node_id_a].connected_edges[node_id_b] = float(tension_strength)
            self.nodes[node_id_b].connected_edges[node_id_a] = float(tension_strength)

    def compute_local_structural_density(self, center_pos: np.ndarray, radius: float = 2.0) -> float:
        r"""
        Calculates Structural Density (M_s) as local topological web condensation:
        D_struct = \sum_{i} Phase_Density_i * exp(-d_i^2 / (2 * radius^2)) + local_bonds_density
        """
        if not self.nodes:
            return 0.1

        total_density = 0.0
        bond_count = 0

        for n in self.nodes.values():
            dist = float(np.linalg.norm(n.position - center_pos))
            if dist <= radius * 2.0:
                weight = float(np.exp(- (dist ** 2) / (2.0 * (radius ** 2))))
                total_density += n.phase.density * weight
                bond_count += len(n.connected_edges)

        bond_density = bond_count * 0.1
        return float(total_density + bond_density)

    def compute_gravity_field_gradient(self, query_pos: np.ndarray) -> Tuple[np.ndarray, float, float]:
        r"""
        Computes the gravitational slope (gradient vector \nabla T), local tension distortion,
        and dynamically balanced spectral propagation speed c_spectrum.

        Returns:
            - gravity_gradient: Acceleration vector pointing toward highest structural density.
            - tension_distortion: Local field tension curvature magnitude.
            - c_spectrum: Balanced spectral propagation speed bounded by tau_limit.
        """
        if not self.nodes:
            return np.zeros(self.dimensions, dtype=np.float32), 0.0, self.base_spectral_speed

        gradient = np.zeros(self.dimensions, dtype=np.float32)
        total_distortion = 0.0

        for n in self.nodes.values():
            diff = n.position - query_pos
            dist_sq = float(np.sum(diff ** 2)) + 0.01
            dist = float(np.sqrt(dist_sq))

            # Effective mass = structural density
            effective_mass = n.phase.density * (1.0 + n.phase.concentration * 0.2)
            pull_mag = effective_mass / dist_sq

            gradient += (diff / dist) * pull_mag
            total_distortion += effective_mass / dist

        # Dynamic balancing of spectral propagation speed:
        # As tension distortion increases toward tau_limit, c_spectrum compresses smoothly
        # to enforce stability and causality conservation.
        tau_limit_avg = np.mean([n.phase.tension_limit for n in self.nodes.values()]) if self.nodes else 2.0
        c_spectrum = self.base_spectral_speed / (1.0 + total_distortion / (tau_limit_avg + 1e-5))

        return gradient, float(total_distortion), float(c_spectrum)

    def compute_least_resistance_geodesic_path(self, start_node_id: str, target_node_id: str) -> Tuple[List[str], float]:
        """
        Calculates the minimum resistance flow path (Dijkstra / SDF Geodesic Flow)
        guided by field tension and structural density.
        """
        if start_node_id not in self.nodes or target_node_id not in self.nodes:
            return [], float('inf')

        distances: Dict[str, float] = {nid: float('inf') for nid in self.nodes}
        previous: Dict[str, Optional[str]] = {nid: None for nid in self.nodes}
        distances[start_node_id] = 0.0

        pq = [(0.0, start_node_id)]

        while pq:
            current_dist, current_id = heapq.heappop(pq)

            if current_dist > distances[current_id]:
                continue

            if current_id == target_node_id:
                break

            current_node = self.nodes[current_id]

            for neighbor_id, bond_tension in current_node.connected_edges.items():
                if neighbor_id not in self.nodes:
                    continue
                neighbor_node = self.nodes[neighbor_id]

                # Distance in physical space
                spatial_dist = float(np.linalg.norm(current_node.position - neighbor_node.position))

                # Resistance: High mobility reduces resistance; high tension increases flow alignment
                mobility_factor = max(0.01, (current_node.phase.mobility + neighbor_node.phase.mobility) * 0.5)
                edge_cost = (spatial_dist / mobility_factor) / max(0.1, bond_tension)

                new_dist = current_dist + edge_cost
                if new_dist < distances[neighbor_id]:
                    distances[neighbor_id] = new_dist
                    previous[neighbor_id] = current_id
                    heapq.heappush(pq, (new_dist, neighbor_id))

        # Reconstruct path
        path = []
        curr: Optional[str] = target_node_id
        while curr is not None:
            path.append(curr)
            curr = previous[curr]
        path.reverse()

        if path[0] == start_node_id:
            return path, distances[target_node_id]
        return [], float('inf')

    def compute_cognitive_heat_from_tension_differential(self,
                                                          in_force_intention: np.ndarray,
                                                          out_force_reality: np.ndarray) -> float:
        r"""
        Computes kinetic agitation heat (Cognitive Heat) generated by the field gradient
        differential between internal intention (In-force) and external reality/friction (Out-force).

        Heat = ||F_in - F_out|| * (1.0 - CosineSimilarity(F_in, F_out) + 0.1)
        """
        norm_in = np.linalg.norm(in_force_intention)
        norm_out = np.linalg.norm(out_force_reality)

        if norm_in < 1e-9 or norm_out < 1e-9:
            return 0.0

        diff_magnitude = float(np.linalg.norm(in_force_intention - out_force_reality))
        cos_sim = float(np.dot(in_force_intention, out_force_reality) / (norm_in * norm_out + 1e-9))

        mismatch_factor = max(0.1, 1.0 - cos_sim)
        heat_impulse = float(diff_magnitude * mismatch_factor)

        # Apply generated heat impulse to drive thermal agitation across the web
        self.apply_thermal_agitation_and_phase_transition(heat_impulse=heat_impulse)
        return heat_impulse

    def apply_thermal_agitation_and_phase_transition(self, heat_impulse: float):
        r"""
        Applies kinetic agitation heat T_agitation across all nodes, driving phase transitions:
        - Rock -> Water (Melting): T_agitation > threshold, increases mobility \mu.
        - Water -> Gas_Cloud (Evaporation): T_agitation > higher threshold, increases entropy S.
        - Cooling (heat_impulse <= 0): Recovers solid lattice or condenses droplets.
        """
        for node in self.nodes.values():
            node.phase.agitation = max(0.0, node.phase.agitation + heat_impulse)

            # Melting transition
            if node.phase.agitation > 1.5 and node.phase.mobility < 1.0:
                node.phase.mobility = float(min(3.0, node.phase.mobility + node.phase.agitation * 0.5))
                node.phase.density = float(max(0.2, node.phase.density * 0.8))

            # Evaporation transition
            if node.phase.agitation > 3.0:
                node.phase.entropy = float(node.phase.entropy + node.phase.agitation * 0.8)
                node.phase.mobility = float(min(5.0, node.phase.mobility + 1.0))
                node.phase.density = float(max(0.05, node.phase.density * 0.5))

            # Solidification / Cooling
            if heat_impulse <= 0:
                node.phase.agitation = float(max(0.0, node.phase.agitation - 0.5))
                if node.phase.agitation < 0.5:
                    node.phase.mobility = float(max(0.05, node.phase.mobility * 0.7))
                    node.phase.density = float(min(5.0, node.phase.density * 1.2))
                    node.phase.entropy = float(max(0.05, node.phase.entropy * 0.5))

    def trigger_condensation_nucleation(self,
                                         center_pos: np.ndarray,
                                         radius: float = 3.0) -> List[str]:
        r"""
        Condensation / Nucleation Loop:
        Gathers high-entropy unbound variable cloud nodes (Gas) within radius,
        binds them via causal tension, and condenses them into explicit thought droplets (Water/Rock).
        """
        cloud_nodes = []
        for nid, node in self.nodes.items():
            dist = float(np.linalg.norm(node.position - center_pos))
            if dist <= radius and node.phase.entropy > 1.0:
                cloud_nodes.append(nid)

        if not cloud_nodes:
            return []

        # Condense cloud nodes
        for nid in cloud_nodes:
            node = self.nodes[nid]
            node.phase.entropy = 0.1
            node.phase.mobility = 0.1
            node.phase.density = 2.0
            node.phase.agitation = 0.1

        # Interconnect condensed nodes with causal tension bonds if 2 or more
        if len(cloud_nodes) >= 2:
            for i in range(len(cloud_nodes)):
                for j in range(i + 1, len(cloud_nodes)):
                    self.connect_nodes(cloud_nodes[i], cloud_nodes[j], tension_strength=2.0)

        return cloud_nodes

    def execute_topological_rupture(self,
                                     blade_node_id: str,
                                     target_region_center: np.ndarray,
                                     target_radius: float = 2.0,
                                     is_cognitive: bool = False) -> RuptureEvent:
        r"""
        Executes Topological Rupture (Physical Slicing or Cognitive Insight Slicing):
        - Blade node focuses extreme concentration (K_concentration) onto a narrow axis.
        - When concentration tension exceeds target region's tau_limit, boundary cohesion bonds rupture.
        - Physical Field: Graph splits into separate subfields.
        - Cognitive Field: Isolates core insight nodes from surrounding noise nodes.
        """
        if blade_node_id not in self.nodes:
            raise ValueError(f"Blade node '{blade_node_id}' does not exist in topological web.")

        blade = self.nodes[blade_node_id]

        # Force blade phase into extreme Blade archetype
        blade.phase.concentration = 10.0
        blade.phase.density = 5.0
        blade.phase.tension_limit = 0.1

        # Find affected target nodes
        target_node_ids = []
        for nid, node in self.nodes.items():
            if nid == blade_node_id:
                continue
            dist = float(np.linalg.norm(node.position - target_region_center))
            if dist <= target_radius:
                target_node_ids.append(nid)

        # Calculate concentration tension peak
        tension_peak = blade.phase.concentration * blade.phase.density

        # Cut edges that cross the slicing plane or are within slicing range when tension_peak > tau_limit
        for nid in target_node_ids:
            node = self.nodes[nid]
            neighbors = list(node.connected_edges.keys())
            for neighbor_id in neighbors:
                if neighbor_id not in self.nodes:
                    continue
                neighbor_node = self.nodes[neighbor_id]

                # Check if edge crosses or intersects slicing plane or if tension peak exceeds threshold
                min_tau = min(node.phase.tension_limit, neighbor_node.phase.tension_limit)
                if tension_peak > min_tau:
                    # Sever connection between node and neighbor
                    del node.connected_edges[neighbor_id]
                    if nid in neighbor_node.connected_edges:
                        del neighbor_node.connected_edges[nid]

        # Calculate separated subfields (Connected Components)
        subfields = self._compute_connected_components(target_node_ids)

        extracted_insights = None
        if is_cognitive:
            # Insight Slicing: isolate highest density / lowest noise nodes as core insight
            extracted_insights = sorted(
                target_node_ids,
                key=lambda nid: self.nodes[nid].phase.density,
                reverse=True
            )[:max(1, len(target_node_ids) // 2)]

        event = RuptureEvent(
            rupture_id=f"Rupture_{len(self.ruptures) + 1}_{'Cognitive' if is_cognitive else 'Physical'}",
            source_blade_id=blade_node_id,
            target_node_ids=target_node_ids,
            tension_peak=float(tension_peak),
            separated_subfields=subfields,
            cognitive_insight_extracted=extracted_insights,
            is_cognitive=is_cognitive
        )
        self.ruptures.append(event)
        return event

    def _compute_connected_components(self, node_subset: List[str]) -> List[List[str]]:
        """Helper to compute connected subfield components within a node subset."""
        visited = set()
        components = []
        subset_set = set(node_subset)

        for nid in node_subset:
            if nid in visited or nid not in self.nodes:
                continue

            comp = []
            queue = [nid]
            visited.add(nid)

            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                curr_node = self.nodes[curr]

                for neighbor in curr_node.connected_edges.keys():
                    if neighbor in subset_set and neighbor not in visited and neighbor in self.nodes:
                        visited.add(neighbor)
                        queue.append(neighbor)

            components.append(comp)

        return components

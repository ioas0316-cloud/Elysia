import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class StatNode:
    name: str          # 'health', 'force', 'mind', 'speed', 'intelligence'
    base_value: float  # Raw input stat investment
    mass: float
    position: np.ndarray  # 3D Position
    velocity: np.ndarray  # 3D Velocity
    force: np.ndarray     # 3D Force accumulator

@dataclass
class CatastropheVector:
    is_collapsed: bool
    type: str          # "Overload", "Madness", "Dispersion", "None"
    magnitude: float   # Severity
    description: str

@dataclass
class CrystallizedAxis:
    name: str
    base_stats_signature: Dict[str, float]  # Normalized base stats
    node_positions: Dict[str, np.ndarray]    # Stored positions

class QuantumStatField:
    """
    [Quantum Stat Field: 5D Tensegrity Engine]
    Represents player stats not as static values, but as a dynamic tension field.
    The five core stats act as physical nodes in a 3D space interacting via
    Hookean springs, gravitational anchors, and exponential barrier repulsions.

    [Non-Computational Flow]
    Once a dynamic equilibrium or resonance is achieved and crystallized,
    future identical configurations bypass the active physics simulation loop
    and resolve instantly via memory crystallization, embodying "Do not calculate, let it flow."
    """
    def __init__(self, base_stats: Dict[str, float] = None):
        if base_stats is None:
            base_stats = {
                "health": 10.0,
                "force": 10.0,
                "mind": 10.0,
                "speed": 10.0,
                "intelligence": 10.0
            }

        self.base_stats = base_stats
        self.nodes: Dict[str, StatNode] = {}
        self.crystallized_axes: Dict[str, CrystallizedAxis] = {}
        self.active_axis: Optional[str] = None

        # Spring links defining the pentagram + cross-link (Force - Mind)
        # Symmetrical connection pairs
        self.links = [
            ("health", "force"),
            ("health", "mind"),
            ("force", "mind"),
            ("force", "speed"),
            ("mind", "intelligence"),
            ("speed", "intelligence")
        ]

        self.rest_lengths: Dict[Tuple[str, str], float] = {}
        self.k_spring = 8.0          # Spring stiffness
        self.G_anchor = 5.0          # Gravity anchor constant (Health pull)
        self.G_mind = 5.0            # Mind control attractive constant
        self.C_rep = 2.0             # Mutual baseline repulsion
        self.C_base_barrier = 50.0   # Extreme investment barrier repulsion
        self.alpha_barrier = 1.5     # Exponential barrier decay rate
        self.softening = 0.1         # Avoid division by zero
        self.damping = 0.85          # Damping for stability and convergence

        self._initialize_field()

    def _initialize_field(self):
        # Position vertices on a regular pentagon in the XY plane centered at (0, 0, 0)
        # Radius R = 5.0
        R = 5.0
        angles = {
            "health": np.pi / 2.0,            # 90 deg (Top)
            "mind": np.pi / 10.0,             # 18 deg (Right)
            "intelligence": -3.0 * np.pi / 10.0, # -54 deg (Bottom-Right)
            "speed": -7.0 * np.pi / 10.0,        # -126 deg (Bottom-Left)
            "force": 9.0 * np.pi / 10.0          # 162 deg (Left)
        }

        for name, angle in angles.items():
            pos = np.array([R * np.cos(angle), R * np.sin(angle), 0.0], dtype=np.float32)
            val = self.base_stats.get(name, 10.0)
            # Mass is proportional to the stat value (minimum mass 0.1 to prevent singularity)
            mass = max(0.1, float(val))
            self.nodes[name] = StatNode(
                name=name,
                base_value=val,
                mass=mass,
                position=pos,
                velocity=np.zeros(3, dtype=np.float32),
                force=np.zeros(3, dtype=np.float32)
            )

        # Calculate ideal rest lengths based on default balanced layout (R=5.0)
        for u, v in self.links:
            pos_u = self.nodes[u].position
            pos_v = self.nodes[v].position
            dist = np.linalg.norm(pos_u - pos_v)
            self.rest_lengths[(u, v)] = float(dist)
            self.rest_lengths[(v, u)] = float(dist)

    def update_base_stats(self, base_stats: Dict[str, float]):
        """Updates the raw base stats and updates node masses accordingly."""
        self.base_stats.update(base_stats)
        for name, val in self.base_stats.items():
            if name in self.nodes:
                self.nodes[name].base_value = val
                self.nodes[name].mass = max(0.1, float(val))

    def crystallize_axis(self, name: str):
        """
        Freezes and crystallizes the current dynamic equilibrium of base stats and node positions.
        Once crystallized, this configuration is retrieved instantly without numerical spring-mass solver.
        """
        total_stats = sum(node.base_value for node in self.nodes.values())
        if total_stats == 0:
            total_stats = 1e-9

        signature = {k: v / total_stats for k, v in self.base_stats.items()}
        positions = {k: node.position.copy() for k, node in self.nodes.items()}

        self.crystallized_axes[name] = CrystallizedAxis(
            name=name,
            base_stats_signature=signature,
            node_positions=positions
        )

    def _find_matching_crystallized_axis(self) -> Optional[str]:
        """Checks if current base stats match any crystallized axis signature within 1% tolerance."""
        total_stats = sum(self.base_stats.values())
        if total_stats == 0:
            total_stats = 1e-9

        current_sig = {k: v / total_stats for k, v in self.base_stats.items()}

        for name, axis in self.crystallized_axes.items():
            match = True
            for k, val in current_sig.items():
                ref_val = axis.base_stats_signature.get(k, 0.0)
                if abs(val - ref_val) > 0.01:
                    match = False
                    break
            if match:
                return name
        return None

    def step(self, dt: float = 0.1):
        """Runs one step of the physical tensegrity simulation, bypassing if crystallized axis matched."""
        # Check if we can bypass active computation using Crystallized Axis
        matched_axis_name = self._find_matching_crystallized_axis()
        if matched_axis_name:
            self.active_axis = matched_axis_name
            axis = self.crystallized_axes[matched_axis_name]
            # Zero out forces and velocities, and snap positions instantly (Do not calculate, let it flow)
            for name, node in self.nodes.items():
                node.position = axis.node_positions[name].copy()
                node.velocity.fill(0.0)
                node.force.fill(0.0)
            return

        self.active_axis = None

        # Reset forces
        for node in self.nodes.values():
            node.force.fill(0.0)

        # Total base stats sum
        total_stats = sum(node.base_value for node in self.nodes.values())
        if total_stats == 0:
            total_stats = 1e-9

        # Check for Field Dispersion condition
        # If health or mind is extremely low (ratio < 0.1) or mass is near-zero
        health_ratio = self.nodes["health"].base_value / total_stats
        mind_ratio = self.nodes["mind"].base_value / total_stats

        is_dispersed = (health_ratio < 0.1 or mind_ratio < 0.1)

        # 1. Spring restoration forces (Skeleton tension)
        for (u, v), rest_len in self.rest_lengths.items():
            # Skip duplicate processing
            if u >= v:
                continue

            node_u = self.nodes[u]
            node_v = self.nodes[v]

            diff = node_v.position - node_u.position
            dist = np.linalg.norm(diff)
            if dist < 1e-9:
                dist = 1e-9
            direction = diff / dist

            # Hooke's Law: F = k * (x - x0)
            # If dispersion is active, spring tension is weakened
            k_eff = self.k_spring * 0.1 if is_dispersed else self.k_spring
            f_mag = k_eff * (dist - rest_len)

            f_vec = f_mag * direction
            node_u.force += f_vec
            node_v.force -= f_vec

        # 2. Gravity Anchor Force (Health pulls other nodes towards it)
        # Health itself is attracted to origin (0,0,0) to keep system centered
        health_node = self.nodes["health"]
        health_node.force -= 10.0 * health_node.position # Strong spring to center origin

        if not is_dispersed:
            for name, node in self.nodes.items():
                if name == "health":
                    continue
                diff = health_node.position - node.position
                dist = np.linalg.norm(diff)
                if dist < 1e-9:
                    dist = 1e-9
                direction = diff / dist

                # Gravity-like pull: G * m1 * m2 / r^2
                f_mag = (self.G_anchor * health_node.mass * node.mass) / (dist**2 + self.softening)
                node.force += f_mag * direction
                health_node.force -= f_mag * direction

        # 3. Mind Control Attraction (Mind pulls Force, Speed, Intelligence)
        mind_node = self.nodes["mind"]
        if not is_dispersed:
            for name in ["force", "speed", "intelligence"]:
                node = self.nodes[name]
                diff = mind_node.position - node.position
                dist = np.linalg.norm(diff)
                if dist < 1e-9:
                    dist = 1e-9
                direction = diff / dist

                # Mind attractive pull proportional to mind's mass/investment
                f_mag = (self.G_mind * mind_node.mass * node.mass) / (dist**2 + self.softening)
                node.force += f_mag * direction
                mind_node.force -= f_mag * direction

        # 4. Mutual Repulsion and Extreme Investment Barrier Potential
        node_names = list(self.nodes.keys())
        for i in range(len(node_names)):
            for j in range(i + 1, len(node_names)):
                name_i = node_names[i]
                name_j = node_names[j]

                node_i = self.nodes[name_i]
                node_j = self.nodes[name_j]

                diff = node_j.position - node_i.position
                dist = np.linalg.norm(diff)
                if dist < 1e-9:
                    dist = 1e-9
                direction = diff / dist

                # Baseline repulsion: C_rep / r^2
                f_rep_baseline = self.C_rep / (dist**2 + self.softening)

                # Barrier potential if either node has extreme ratio (> 0.8)
                ratio_i = node_i.base_value / total_stats
                ratio_j = node_j.base_value / total_stats

                barrier_scale = 1.0
                if ratio_i > 0.8:
                    barrier_scale += np.exp(10.0 * (ratio_i - 0.8))
                if ratio_j > 0.8:
                    barrier_scale += np.exp(10.0 * (ratio_j - 0.8))

                # Exponential repulsion: C_base * scale * e^(-r * alpha)
                f_rep_barrier = self.C_base_barrier * barrier_scale * np.exp(-dist * self.alpha_barrier)

                f_rep_total = f_rep_baseline + f_rep_barrier

                # Push them apart
                node_i.force -= f_rep_total * direction
                node_j.force += f_rep_total * direction

        # 5. Integrate forces to update position and velocity
        for node in self.nodes.values():
            # Acceleration = Force / Mass
            acc = node.force / node.mass
            # Clamp acceleration to prevent extreme spikes (numerical stability)
            acc_norm = np.linalg.norm(acc)
            if acc_norm > 100.0:
                acc = (acc / acc_norm) * 100.0

            node.velocity += acc * dt
            node.position += node.velocity * dt
            node.velocity *= self.damping # Apply friction/damping

    def get_catastrophe_vector(self) -> CatastropheVector:
        """Evaluates whether the field has collapsed and computes severity."""
        total_stats = sum(node.base_value for node in self.nodes.values())
        if total_stats == 0:
            total_stats = 1e-9

        health_ratio = self.nodes["health"].base_value / total_stats
        mind_ratio = self.nodes["mind"].base_value / total_stats

        # 1. Field Dispersion Collapse
        if health_ratio < 0.1 or mind_ratio < 0.1:
            # Measure dispersion severity by how far nodes have drifted from the ideal radius 5.0
            avg_dist = np.mean([np.linalg.norm(node.position) for node in self.nodes.values()])
            magnitude = float(abs(avg_dist - 5.0))
            return CatastropheVector(
                is_collapsed=True,
                type="Dispersion",
                magnitude=magnitude,
                description=f"Field Dispersion! The core anchors are too weak (Health Ratio: {health_ratio:.2f}, Mind Ratio: {mind_ratio:.2f}). The field is disintegrating."
            )

        # 2. Extreme Overload Collapse
        for name, node in self.nodes.items():
            ratio = node.base_value / total_stats
            if ratio >= 0.8:
                # Measure overload by how squashed or stretched the other nodes are
                # Calculate deviation from rest lengths on adjacent links
                deviations = []
                for (u, v), rest_len in self.rest_lengths.items():
                    dist = np.linalg.norm(self.nodes[u].position - self.nodes[v].position)
                    deviations.append(abs(dist - rest_len))
                magnitude = float(np.max(deviations))

                col_type = "Overload" if name != "mind" else "Madness"
                return CatastropheVector(
                    is_collapsed=True,
                    type=col_type,
                    magnitude=magnitude,
                    description=f"Extreme Overload Collapse! Stat '{name}' dominates the field ({ratio*100:.1f}%). Dynamic equilibrium collapsed."
                )

        return CatastropheVector(
            is_collapsed=False,
            type="None",
            magnitude=0.0,
            description="The field is stable and in dynamic equilibrium."
        )

    def evaluate_resonance(self) -> List[Dict[str, Any]]:
        """
        [Resonance: Complementary Leap]
        Checks for Golden Ratio symmetry in distances and stat ratios.
        Returns a list of active resonance sparks.
        """
        sparks = []

        # Golden ratio constant
        phi = 1.6180339887
        tolerance_dist = 0.25 # Distance ratio tolerance
        tolerance_val = 0.15  # Stat value ratio tolerance

        # Center coordinates
        center = np.zeros(3)

        # Let's measure distances of each node from center
        dists = {name: float(np.linalg.norm(node.position - center)) for name, node in self.nodes.items()}

        # Helper to check proximity to Golden Ratio (either phi or 1/phi)
        def is_golden(ratio: float, tolerance: float) -> bool:
            return abs(ratio - phi) < tolerance or abs(ratio - (1.0 / phi)) < tolerance

        # 1. Spacetime Resonance (지능 vs 민첩 - Intelligence & Speed)
        val_ratio_si = self.nodes["speed"].base_value / (self.nodes["intelligence"].base_value + 1e-9)
        dist_ratio_si = dists["speed"] / (dists["intelligence"] + 1e-9)

        if is_golden(val_ratio_si, tolerance_val) and is_golden(dist_ratio_si, tolerance_dist):
            sparks.append({
                "name": "Spacetime Resonance (시공간 공명)",
                "type": "Spacetime_Warp",
                "ratio": float(val_ratio_si),
                "description": "Speed and Intelligence form a perfectly aligned spacetime wave. Speed increases dramatically."
            })

        # 2. Mind-Body Unity (체력 vs 정신 - Health & Mind)
        val_ratio_hm = self.nodes["health"].base_value / (self.nodes["mind"].base_value + 1e-9)
        dist_ratio_hm = dists["health"] / (dists["mind"] + 1e-9)

        if is_golden(val_ratio_hm, tolerance_val) and is_golden(dist_ratio_hm, tolerance_dist):
            sparks.append({
                "name": "Mind-Body Unity (심신일체)",
                "type": "Absolute_Oneness",
                "ratio": float(val_ratio_hm),
                "description": "Health and Mind unite in perfect tensegrity. Damage resistance is boosted."
            })

        return sparks

    def get_topology(self) -> Dict[str, Any]:
        """Returns the current spatial layout and state of the field."""
        return {
            "nodes": {
                name: {
                    "position": node.position.tolist(),
                    "velocity": node.velocity.tolist(),
                    "mass": node.mass,
                    "base_value": node.base_value
                } for name, node in self.nodes.items()
            },
            "catastrophe": self.get_catastrophe_vector().__dict__,
            "resonance": self.evaluate_resonance(),
            "active_axis": self.active_axis
        }

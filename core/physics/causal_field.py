import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class InformationVoxel:
    id: str
    content: Any
    tensor: np.ndarray # Structural signature (N-dim)
    mass: float = 1.0
    position: np.ndarray = None
    velocity: np.ndarray = None
    potential: float = 0.0
    # Chromatic Vector: [Red (Flux), Blue (Order/Resistance), Yellow (Entropy)]
    chromatic_vector: np.ndarray = None

    def __post_init__(self):
        if self.chromatic_vector is None:
            self.chromatic_vector = np.array([0.33, 0.33, 0.34], dtype=np.float32) # Default neutral balance
        if self.position is None:
            self.position = np.zeros(3, dtype=np.float32) # Default 3D space
        if self.velocity is None:
            self.velocity = np.zeros(3, dtype=np.float32)
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float32)
        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity, dtype=np.float32)

@dataclass
class ConnectivityBeam:
    source_id: str
    target_id: str
    strength: float # Coupling strength (Relationship)
    rest_length: float # Ideal distance (Connectivity)
    current_tension: float = 0.0
    break_threshold: float = 5.0 # Max tension before 'tearing'
    is_broken: bool = False

class CausalField:
    """
    [Causal Field Engine: The Gear of Continuity]
    Implements the 4 Continuities to move beyond brute-force discrete calculation.
    1. Relationship: Boundary formation through coupled potentials.
    2. Connectivity: Topology maintenance via beam networks.
    3. Mobility: Conservation of energy/momentum as vectors.
    4. Informational Continuity: Prediction/Interpolation across discrete steps.
    """
    def __init__(self, dimensions: int = 3):
        self.dimensions = dimensions
        self.voxels: Dict[str, InformationVoxel] = {}
        self.beams: List[ConnectivityBeam] = []

        # Field-wide properties
        self.global_potential_gradient = np.zeros(dimensions)
        self.time_step_accumulator = 0.0

    def add_voxel(self, voxel: InformationVoxel):
        self.voxels[voxel.id] = voxel

    def link_voxels(self, id_a: str, id_b: str, strength: float = 1.0):
        if id_a in self.voxels and id_b in self.voxels:
            pos_a = self.voxels[id_a].position
            pos_b = self.voxels[id_b].position
            dist = np.linalg.norm(pos_a - pos_b)
            beam = ConnectivityBeam(source_id=id_a, target_id=id_b, strength=strength, rest_length=dist)
            self.beams.append(beam)

    def apply_impact(self, target_id: str, impulse: np.ndarray):
        """
        [Mobility]
        Instead of just moving a particle, we inject 'Energy' into the Relationship field.
        """
        if target_id in self.voxels:
            voxel = self.voxels[target_id]
            voxel.velocity += impulse / voxel.mass

    def step(self, dt: float = 0.1):
        """
        Advances the field using Continuous Causal Dynamics.
        """
        self._update_connectivity_and_tension(dt)
        self._flow_potential(dt)
        self._preserve_mobility(dt)
        self._enforce_informational_continuity(dt)

    def _update_connectivity_and_tension(self, dt: float):
        """
        [Relationship & Connectivity]
        Calculates tension in beams and handles structural 'tearing'.
        """
        for beam in self.beams:
            if beam.is_broken: continue

            v_a = self.voxels[beam.source_id]
            v_b = self.voxels[beam.target_id]

            diff = v_b.position - v_a.position
            dist = np.linalg.norm(diff)

            # Hooke's Law approximation for connectivity tension
            extension = dist - beam.rest_length
            beam.current_tension = beam.strength * abs(extension)

            if beam.current_tension > beam.break_threshold:
                beam.is_broken = True
                # Informational Continuity: Record the break as an event
                continue

            # Apply force to voxels (Potential alignment)
            force = beam.strength * extension * (diff / (dist + 1e-9))
            v_a.velocity += (force / v_a.mass) * dt
            v_b.velocity -= (force / v_b.mass) * dt

    def _flow_potential(self, dt: float):
        """
        [Potential Difference Dynamics]
        Information flows towards areas of 'Resonance' (Lower Potential).
        """
        voxel_ids = list(self.voxels.keys())
        if not voxel_ids: return

        # Calculate local potential based on structural tensor resonance
        # (Simplified: higher similarity to neighbors = lower potential)
        for vid in voxel_ids:
            v = self.voxels[vid]
            # Potential is influenced by neighbors in the connectivity map
            connected_potentials = []
            for beam in self.beams:
                if beam.is_broken: continue
                neighbor_id = None
                if beam.source_id == vid: neighbor_id = beam.target_id
                elif beam.target_id == vid: neighbor_id = beam.source_id

                if neighbor_id:
                    neighbor = self.voxels[neighbor_id]
                    # Resonance = Dot product of tensors
                    resonance = np.dot(v.tensor, neighbor.tensor)
                    connected_potentials.append(1.0 - resonance)

            if connected_potentials:
                v.potential = np.mean(connected_potentials)
            else:
                v.potential *= 0.9 # Decay isolated potential

    def _preserve_mobility(self, dt: float):
        """
        [Mobility & Chromatic Modulation]
        Integrates velocity into position with momentum conservation.
        The Chromatic Vector modulates the field's physical properties:
        - Red (Flux) increases velocity impact.
        - Blue (Order) increases damping (resistance).
        - Yellow (Entropy) adds Brownian-like noise.
        """
        for v in self.voxels.values():
            r, b, y = v.chromatic_vector

            # 1. Flux (Red) increases effective mobility
            flux_boost = 1.0 + r

            # 2. Order (Blue) increases damping
            damping = 0.95 * (1.0 - (b * 0.2)) # More Blue = more damping (max 20% increase)

            # 3. Entropy (Yellow) adds noise
            noise = (np.random.rand(self.dimensions).astype(np.float32) - 0.5) * y * 0.1

            v.position += v.velocity * flux_boost * dt
            v.velocity = (v.velocity + noise) * damping

    def _enforce_informational_continuity(self, dt: float):
        """
        [Informational Continuity]
        Interpolates/Predicts states to bridge discrete calculation gaps.
        If a voxel is moving fast, we 'smear' its influence.
        """
        for voxel in self.voxels.values():
            speed = np.linalg.norm(voxel.velocity)
            displacement = speed * dt
            
            if displacement < 0.01:
                continue
            
            projected_pos = voxel.position + voxel.velocity * dt
            
            for beam in self.beams:
                if beam.is_broken:
                    continue
                
                partner_id = None
                if beam.source_id == voxel.id:
                    partner_id = beam.target_id
                elif beam.target_id == voxel.id:
                    partner_id = beam.source_id
                
                if partner_id is None:
                    continue
                
                partner = self.voxels[partner_id]
                projected_dist = float(np.linalg.norm(projected_pos - partner.position))
                
                adaptation_rate = float(min(0.3, displacement * 0.1))
                beam.rest_length += (projected_dist - beam.rest_length) * adaptation_rate
            
            trail_strength = float(min(1.0, displacement * 0.5))
            voxel.potential += trail_strength
            
            flux_injection = float(min(0.1, displacement * 0.05))
            voxel.chromatic_vector[0] = min(1.0, voxel.chromatic_vector[0] + flux_injection)
            
            total = float(np.sum(voxel.chromatic_vector))
            if total > 0:
                voxel.chromatic_vector /= total

    def get_topology(self) -> Dict[str, Any]:
        return {
            "voxels": {vid: {"pos": v.position.tolist(), "potential": v.potential} for vid, v in self.voxels.items()},
            "beams": [{"s": b.source_id, "t": b.target_id, "tension": b.current_tension, "broken": b.is_broken} for b in self.beams]
        }

if __name__ == "__main__":
    cf = CausalField()
    v1 = InformationVoxel("v1", "Source", np.array([1,0,0], dtype=np.float32), position=np.array([0,0,0], dtype=np.float32))
    v2 = InformationVoxel("v2", "Target", np.array([1,0.1,0], dtype=np.float32), position=np.array([1,0,0], dtype=np.float32))
    cf.add_voxel(v1)
    cf.add_voxel(v2)
    cf.link_voxels("v1", "v2", strength=2.0)

    print("Initial Topology:", cf.get_topology())
    cf.apply_impact("v1", np.array([-5.0, 0, 0], dtype=np.float32))

    for _ in range(10):
        cf.step(0.1)

    print("Final Topology:", cf.get_topology())

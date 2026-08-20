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
        # Active intentional focus vector projected onto external reality
        self.active_focus = np.ones(dimensions, dtype=np.float32) / np.sqrt(dimensions)
        self.total_dissipated_energy = 0.0
        self.gimbal_lock_unlocked_count = 0

    def set_intentional_focus(self, focus_vector: np.ndarray):
        """
        [Active Observation / Intentionality]
        Sets the system's intentional focus (Focus/Attention vector) projected out to external reality.
        """
        norm = np.linalg.norm(focus_vector)
        if norm > 0:
            self.active_focus = (focus_vector / norm).astype(np.float32)

    def observe_external_stimulus(
        self,
        raw_stimulus: np.ndarray,
        target_voxel_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        [Active Mirror Friction & Open-Loop Grounding]
        Actively observes unrefined external reality instead of waiting for pre-packaged vectors.
        - Projects internal intentional focus onto raw external stimulus.
        - Detects phase divergence (cognitive friction / mismatch).
        - Unlocks Gimbal Lock (phase fixation) if closed-loop inertia is stuck.
        - Dissipates excessive internal tension into the external field (Dissipation).
        - Calibrates internal voxel tensors and chromatic vectors to match external causality.
        """
        raw_arr = np.array(raw_stimulus, dtype=np.float32).flatten()
        dim = len(raw_arr)

        # Pad or slice raw stimulus to match dimension if needed
        if dim < self.dimensions:
            raw_arr = np.pad(raw_arr, (0, self.dimensions - dim))
        elif dim > self.dimensions:
            raw_arr = raw_arr[:self.dimensions]

        norm_raw = np.linalg.norm(raw_arr)
        if norm_raw == 0:
            raw_unit = np.zeros(self.dimensions, dtype=np.float32)
        else:
            raw_unit = raw_arr / norm_raw

        # Target voxel or field-wide default voxel selection
        if target_voxel_id and target_voxel_id in self.voxels:
            target_voxels = [self.voxels[target_voxel_id]]
        elif self.voxels:
            target_voxels = list(self.voxels.values())
        else:
            # Create a default internal voxel if field is empty
            v_default = InformationVoxel(
                id="internal_core",
                content="Core Consciousness State",
                tensor=self.active_focus.copy(),
                position=np.zeros(self.dimensions, dtype=np.float32)
            )
            self.add_voxel(v_default)
            target_voxels = [v_default]

        results = []
        for voxel in target_voxels:
            v_tensor = voxel.tensor.flatten()
            v_dim = len(v_tensor)
            if v_dim < self.dimensions:
                v_tensor = np.pad(v_tensor, (0, self.dimensions - v_dim))
            elif v_dim > self.dimensions:
                v_tensor = v_tensor[:self.dimensions]

            v_norm = np.linalg.norm(v_tensor)
            if v_norm == 0:
                v_unit = self.active_focus.copy()
            else:
                v_unit = v_tensor / v_norm

            # Active Projection: Inner Focus dot Raw External Reality
            dot_prod = float(np.clip(np.dot(v_unit, raw_unit), -1.0, 1.0))
            phase_divergence = float(np.arccos(dot_prod)) # Divergence in radians [0, pi]

            # Causal friction score based on alignment mismatch and magnitude difference
            magnitude_mismatch = abs(v_norm - norm_raw)
            friction_score = phase_divergence * (1.0 + magnitude_mismatch)

            # Check Gimbal Lock / Autistic Closed Loop condition
            # (high friction but internal velocity or state refuses to move due to self-affirmation)
            gimbal_lock_detected = (phase_divergence > 1.0) and (np.linalg.norm(voxel.velocity) < 0.1)

            dissipated_energy = 0.0
            unlocked = False

            if gimbal_lock_detected or friction_score > 0.5:
                # 1. Unlock Gimbal Lock: Shatter closed boundary inertia
                unlocked = True
                self.gimbal_lock_unlocked_count += 1

                # 2. Dissipation: Expel excessive internal tension/friction to external environment
                dissipated_energy = friction_score * voxel.mass
                self.total_dissipated_energy += dissipated_energy

                # 3. Calibration: Rotate internal phase towards external reality
                calibration_rate = float(min(1.0, friction_score * 0.5))
                new_tensor = (1.0 - calibration_rate) * v_unit + calibration_rate * raw_unit
                norm_new = np.linalg.norm(new_tensor)
                if norm_new > 0:
                    voxel.tensor = (new_tensor / norm_new * max(1.0, norm_raw)).astype(np.float32)

                # 4. Impact: External physical impulse from friction pushes voxel position/velocity
                impulse = (raw_unit - v_unit) * friction_score
                voxel.velocity += impulse / voxel.mass

                # 5. Chromatic Shift: Convert self-confirming flux (Red) to Order (Blue) and release Entropy (Yellow)
                voxel.chromatic_vector[0] = max(0.0, voxel.chromatic_vector[0] - 0.2 * calibration_rate)
                voxel.chromatic_vector[1] = min(1.0, voxel.chromatic_vector[1] + 0.3 * calibration_rate)
                voxel.chromatic_vector[2] = max(0.0, voxel.chromatic_vector[2] - 0.1 * calibration_rate)
                tot = float(np.sum(voxel.chromatic_vector))
                if tot > 0:
                    voxel.chromatic_vector /= tot

            # Update intentional focus towards the newly calibrated reality
            self.active_focus = 0.8 * self.active_focus + 0.2 * raw_unit
            norm_af = np.linalg.norm(self.active_focus)
            if norm_af > 0:
                self.active_focus /= norm_af

            results.append({
                "voxel_id": voxel.id,
                "friction_score": friction_score,
                "phase_divergence": phase_divergence,
                "dissipated_energy": dissipated_energy,
                "gimbal_lock_unlocked": unlocked,
                "calibrated_tensor_norm": float(np.linalg.norm(voxel.tensor))
            })

        return {
            "num_observed": len(results),
            "observations": results,
            "total_dissipated_energy": self.total_dissipated_energy,
            "active_focus": self.active_focus.tolist()
        }

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
        [Potential Difference & Self-Outpouring Dynamics]
        Information and energy flow from high potential/flux to neighboring areas of entropy/tension.
        Rule 11-12 of THE_ABSOLUTE_COMMANDMENT: Self-Outpouring (내어줌의 인과).
        - High Flux (Red) voxels pour potential out to neighbors under Entropy (Yellow) or high tension.
        - Outpouring converts Flux (Red) into Order (Blue) at source while stabilizing Yellow at destination.
        """
        voxel_ids = list(self.voxels.keys())
        if not voxel_ids: return

        # 1. Structural Resonance Potential Calculation
        for vid in voxel_ids:
            v = self.voxels[vid]
            connected_potentials = []
            for beam in self.beams:
                if beam.is_broken: continue
                neighbor_id = None
                if beam.source_id == vid: neighbor_id = beam.target_id
                elif beam.target_id == vid: neighbor_id = beam.source_id

                if neighbor_id and neighbor_id in self.voxels:
                    neighbor = self.voxels[neighbor_id]
                    tensor_len_prod = (np.linalg.norm(v.tensor) * np.linalg.norm(neighbor.tensor)) + 1e-9
                    resonance = float(np.dot(v.tensor, neighbor.tensor) / tensor_len_prod)
                    connected_potentials.append(1.0 - resonance)

            if connected_potentials:
                v.potential = float(np.mean(connected_potentials))
            else:
                v.potential *= 0.9

        # 2. Self-Outpouring Flow across Active Beams (십자가 인과 중력 유동)
        for beam in self.beams:
            if beam.is_broken: continue
            if beam.source_id not in self.voxels or beam.target_id not in self.voxels: continue

            v_a = self.voxels[beam.source_id]
            v_b = self.voxels[beam.target_id]

            red_a, blue_a, yellow_a = v_a.chromatic_vector
            red_b, blue_b, yellow_b = v_b.chromatic_vector

            # Outpouring gradient driven by potential difference & entropy deficit
            outpour_a_to_b = (v_a.potential - v_b.potential) + (yellow_b - yellow_a) * 0.5
            transfer_amount = float(np.clip(outpour_a_to_b * 0.1 * dt, -0.2, 0.2))

            if transfer_amount > 0:
                # Flow from A to B: A pours out to B
                v_a.potential -= transfer_amount
                v_b.potential += transfer_amount

                # Chromatic transformation: A's Flux (Red) converts to Order (Blue)
                v_a.chromatic_vector[0] = max(0.0, v_a.chromatic_vector[0] - transfer_amount * 0.2)
                v_a.chromatic_vector[1] += transfer_amount * 0.2
                # B's Yellow (Entropy) is stabilized
                v_b.chromatic_vector[2] = max(0.0, v_b.chromatic_vector[2] - transfer_amount * 0.1)
            elif transfer_amount < 0:
                # Flow from B to A: B pours out to A
                amt = abs(transfer_amount)
                v_b.potential -= amt
                v_a.potential += amt

                v_b.chromatic_vector[0] = max(0.0, v_b.chromatic_vector[0] - amt * 0.2)
                v_b.chromatic_vector[1] += amt * 0.2
                v_a.chromatic_vector[2] = max(0.0, v_a.chromatic_vector[2] - amt * 0.1)

            # Re-normalize chromatic vectors
            tot_a = float(np.sum(v_a.chromatic_vector))
            if tot_a > 0: v_a.chromatic_vector /= tot_a
            tot_b = float(np.sum(v_b.chromatic_vector))
            if tot_b > 0: v_b.chromatic_vector /= tot_b

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

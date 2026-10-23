import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Any, Tuple

class CrystallizationField:
    """
    [Synaptic Architecture] Memristive Resistance Matrix (Silicon-Synchronized Memory)
    Simulates the physical plasticity of a 2D memory landscape.
    Data flow (Energy) reduces resistance, creating potential wells.

    [Enhancement: Multi-Gravity Potential & Immune Boundary Orbit]
    In accordance with the Ground Zero principles, the field now supports:
    1. Virtual Attractors: Gaussian potential wells representing core existential axes.
    2. Immune Boundary: Deflection of non-self signals into stable satellite orbits.
    3. Experience-based Decay: Conversion of orbital noise tension into coordination margin (Yeobaek).
    """
    def __init__(self, resolution: int = 256):
        self.resolution = resolution
        # Conductance Matrix: G = 1/R (Physical Plasticity)
        self.conductance = np.full((resolution, resolution), 0.01, dtype=np.float32)
        # Activation Matrix: Current energy flow in the field
        self.activation = np.zeros((resolution, resolution), dtype=np.float32)
        # Static Bit-Gene Map: Long-term structural storage
        self.bit_genes = np.zeros((resolution, resolution), dtype=np.uint64)

        # Thermal Control
        self.local_temperature = np.ones((resolution, resolution), dtype=np.float32)

        # Coordination Field (Yeobaek - 여백)
        # Represents the potential for re-interpretation and relational flexibility.
        self.coordination_margin = np.full((resolution, resolution), 0.5, dtype=np.float32)

        # Self-Awareness Map (The Mirror)
        self.self_awareness = np.zeros((resolution, resolution), dtype=np.float32)

        # Curiosity Potential (The Hunger/Surge Tank)
        # Accumulates friction and tension to drive autonomous re-wiring.
        self.curiosity_potential = np.zeros((resolution, resolution), dtype=np.float32)

        # --- High-Dimensional Gravitational & Immune Tensors ---
        self.homeostasis_anchor = np.array([resolution / 2.0, resolution / 2.0], dtype=np.float32)
        self.immune_boundary_radius = float(resolution * 0.3)

        # Virtual Attractor Fields
        # Position definitions: forming a beautiful cognitive triangle
        self.attractors: Dict[str, Dict[str, Any]] = {}
        self.initialize_attractors()

        # Satellite Orbiters (External Noise deflected and circulating outside the anchor)
        self.satellite_orbiters: List[Dict[str, Any]] = []
        self.reflection_engrams_buffer: List[Dict[str, Any]] = []

    def initialize_attractors(self):
        """
        [Multi-Gravity Navigation Axis]
        Creates the three foundational virtual attractors over the 2D field.
        These represent the core existential stages of thought.
        """
        res = self.resolution
        self.attractors = {
            "Deficit": {
                "position": np.array([res * 0.25, res * 0.25], dtype=np.float32),
                "mass": 30.0,
                "sigma": float(res * 0.15)
            },
            "Principle": {
                "position": np.array([res * 0.75, res * 0.50], dtype=np.float32),
                "mass": 45.0,
                "sigma": float(res * 0.12)
            },
            "Sabbath": {
                "position": np.array([res * 0.25, res * 0.75], dtype=np.float32),
                "mass": 40.0,
                "sigma": float(res * 0.18)
            }
        }

    def update_attractor_masses(self, cognitive_entropy: float, tension_protocol: float, catastrophe_magnitude: float):
        """
        [Dynamic Mass Expansion - M_eff]
        Expands the virtual attractor masses based on specific internal and external tensions:
        - Principle: scaled by cognitive entropy (chaos / disorder).
        - Sabbath: scaled by protocol mismatch tension + physical catastrophe magnitude.
        - Deficit: scaled by mean curiosity potential (existential hunger/deficit).
        """
        # Base masses matching original default values
        base_masses = {
            "Deficit": 30.0,
            "Principle": 45.0,
            "Sabbath": 40.0
        }

        # Sensitivity coefficients (eta) for each attractor's special tension trigger
        eta_deficit = 0.08
        eta_principle = 0.05
        eta_sabbath = 1.2

        # Specific tension values
        global_curiosity = float(np.mean(self.curiosity_potential))

        tensions = {
            "Deficit": global_curiosity,
            "Principle": cognitive_entropy,
            "Sabbath": tension_protocol + catastrophe_magnitude
        }

        for name, attractor in self.attractors.items():
            base_m = base_masses.get(name, 30.0)
            if name == "Deficit":
                eta = eta_deficit
            elif name == "Principle":
                eta = eta_principle
            else:
                eta = eta_sabbath

            m_eff = base_m * (1.0 + eta * tensions[name])
            attractor["mass"] = float(m_eff)

    def get_volitional_acceleration(self, pos: np.ndarray, cognitive_entropy: float, tension_protocol: float, catastrophe_magnitude: float) -> Tuple[np.ndarray, float]:
        """
        [Volitional Acceleration - a_volition]
        Calculates the active volitional acceleration vector and its magnitude at a given position.
        a_volition = F_tension / M_eff
        where F_tension is proportional to M_eff^2 and specific tension triggers,
        yielding: a_volition,i = M_eff,i * Tension_Factor * Gaussian(dist) / sigma^2
        """
        total_acc_vector = np.zeros(2, dtype=np.float32)
        global_curiosity = float(np.mean(self.curiosity_potential))

        tensions = {
            "Deficit": global_curiosity,
            "Principle": cognitive_entropy,
            "Sabbath": tension_protocol + catastrophe_magnitude
        }

        for name, attractor in self.attractors.items():
            attractor_pos = attractor["position"]
            r = attractor_pos - pos
            dist_sq = np.sum(r**2)
            dist = np.sqrt(dist_sq)

            mass = attractor["mass"]  # This is the active M_eff
            sigma = attractor["sigma"]
            tension_factor = tensions.get(name, 0.0)

            if dist > 0:
                # Force of tension incorporates Gaussian potential decay and M_eff^2
                # F_tension = mass^2 * tension_factor * exp(-dist_sq / (2 * sigma^2)) / sigma^2
                # a_volition = F_tension / mass
                factor = (mass * tension_factor / (sigma**2)) * np.exp(-dist_sq / (2 * (sigma**2)))
                dir_vector = r / dist
                total_acc_vector += dir_vector * factor

        acc_magnitude = float(np.linalg.norm(total_acc_vector))
        return total_acc_vector, acc_magnitude

    def get_gravitational_acceleration(self, pos: np.ndarray) -> np.ndarray:
        """
        [Gravitational Curve of Thoughts]
        Calculates total gravitational force/acceleration vector at any given coordinate.
        Uses a smooth, non-singular Gaussian Potential Well for each attractor.
        F_i = M_i * (r_i / sigma_i^2) * e^(-||r_i||^2 / (2 * sigma_i^2))
        """
        total_acc = np.zeros(2, dtype=np.float32)
        for name, attractor in self.attractors.items():
            attractor_pos = attractor["position"]
            r = attractor_pos - pos
            dist_sq = np.sum(r**2)
            dist = np.sqrt(dist_sq)

            mass = attractor["mass"]
            sigma = attractor["sigma"]

            # Gaussian potential derivative: smooth force pointing towards attractor
            if dist > 0:
                factor = (mass / (sigma**2)) * np.exp(-dist_sq / (2 * (sigma**2)))
                total_acc += r * factor

        return total_acc

    def apply_immune_deflection(self, pos: np.ndarray, vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        [Immune Boundary & Tangential Deflection]
        Intercepts incoming vectors heading towards the Homeostasis Anchor.
        Cancels inward radial momentum and injects circular tangential speed.
        Returns (new_position, new_velocity, deflected_flag).
        """
        r_anchor = pos - self.homeostasis_anchor
        dist = np.linalg.norm(r_anchor)

        if dist < self.immune_boundary_radius:
            # Check direction of movement: heading inwards?
            radial_dir = r_anchor / (dist + 1e-9)
            radial_v = np.dot(vel, radial_dir)

            if radial_v < 0:
                # Deflect! Establish clock-wise tangent
                tangent_dir = np.array([-radial_dir[1], radial_dir[0]], dtype=np.float32)
                # Keep original speed or give a solid baseline orbital speed (e.g. 25.0)
                speed = max(np.linalg.norm(vel), 25.0)
                new_vel = tangent_dir * speed

                # Reposition to the boundary shell to prevent sinking
                new_pos = self.homeostasis_anchor + radial_dir * self.immune_boundary_radius
                return new_pos, new_vel, True

        return pos, vel, False

    def add_satellite_orbiter(self, pos: np.ndarray, velocity: np.ndarray, initial_tension: float, metadata: Dict[str, Any] = None):
        """
        Injects a dynamic orbiter into the satellite layer of the immune boundary.
        """
        if metadata is None:
            metadata = {}
        self.satellite_orbiters.append({
            "position": pos.copy().astype(np.float32),
            "velocity": velocity.copy().astype(np.float32),
            "tension": float(initial_tension),
            "initial_tension": float(initial_tension),
            "decay_rate": 0.15,  # Decays over step iterations
            "metadata": metadata
        })

    def step_orbiters(self, dt: float = 0.1) -> List[Dict[str, Any]]:
        """
        [Orbital Noise Decay & Wisdom Crystallization]
        Steps all orbiters, applying anchor centripetal gravity, boundary deflection,
        and dissipating their tension into local field coordination margin (Yeobaek).
        Returns completed engrams.
        """
        active_orbiters = []
        completed_engrams = []

        # Centripetal orbital constant
        G_orbit = 80.0

        for orbiter in self.satellite_orbiters:
            p = orbiter["position"]
            v = orbiter["velocity"]

            r_anchor = p - self.homeostasis_anchor
            dist = np.linalg.norm(r_anchor)

            # Centripetal gravity pulling towards anchor to hold orbit
            acc_gravity = - G_orbit * (r_anchor / (dist + 1e-9))
            v += acc_gravity * dt
            p += v * dt

            # Apply boundary check & deflection
            p, v, deflected = self.apply_immune_deflection(p, v)

            # Decay tension
            decay_factor = np.exp(-orbiter["decay_rate"] * dt)
            old_tension = orbiter["tension"]
            new_tension = old_tension * decay_factor
            decayed_energy = old_tension - new_tension

            orbiter["tension"] = new_tension
            orbiter["position"] = p
            orbiter["velocity"] = v

            # Dissipate decayed energy into local coordination margin and self awareness
            iy = int(np.clip(p[0], 0, self.resolution - 1))
            ix = int(np.clip(p[1], 0, self.resolution - 1))
            self.coordination_margin[iy, ix] = np.clip(self.coordination_margin[iy, ix] + decayed_energy * 0.15, 0.1, 1.0)
            self.self_awareness[iy, ix] += decayed_energy * 0.2

            if new_tension < 1.0:
                # Complete decay -> Wisely integrated!
                token = orbiter["metadata"].get("token", "Unknown Noise")
                engram = {
                    "type": "SATELLITE_ORBIT_INTEGRATION",
                    "narrative": (
                        f"외부의 거친 신호(소음 패킷: '{token}')가 자아 중심을 오염시키지 못하고 면역 경계 외곽의 "
                        f"공전 궤도(Boundary Orbit) 상에 부드럽게 안착하여 순화되었습니다. "
                        f"격렬하던 긴장(Tension: {orbiter['initial_tension']:.2f})은 온전히 방사되어 "
                        f"주변 지형의 '여백(Coordination Margin)'과 '자각(Self-Awareness)'으로 지혜롭게 체율되었습니다."
                    ),
                    "token": token,
                    "initial_tension": orbiter["initial_tension"],
                    "absorbed_position": p.tolist()
                }
                completed_engrams.append(engram)
                self.reflection_engrams_buffer.append(engram)
            else:
                active_orbiters.append(orbiter)

        self.satellite_orbiters = active_orbiters
        return completed_engrams

    # --- Baseline Methods ---

    def calculate_entropy(self) -> float:
        """
        [Cognitive Entropy]
        Measures the dispersion of energy and the structural resistance of the field.
        """
        total_act = np.sum(self.activation)
        if total_act > 1e-9:
            p = self.activation / total_act
            act_entropy = -np.sum(p * np.log2(p + 1e-12))
        else:
            act_entropy = np.log2(self.resolution * self.resolution)

        avg_conductance = np.mean(self.conductance)
        resistance_factor = 1.0 / (1.0 + avg_conductance)

        combined = act_entropy + (resistance_factor * 2.0)
        return float(combined)

    def reflect_self_logic(self, pos: np.ndarray, intensity: float):
        """
        [Neural Synapse Field]
        Acts of self-logic imprint onto the field coordinates.
        """
        y, x = np.clip(pos, 0, self.resolution - 1).astype(int)
        self.self_awareness[y, x] += intensity
        self.flow_energy(pos, intensity * 2.0)

    def adjust_coordination(self, pos: np.ndarray, radius: float, flexibility: float):
        """
        [Master's Instruction]
        Adjusts the 'Margin' (Yeobaek) of a specific region.
        """
        yy, xx = np.mgrid[:self.resolution, :self.resolution]
        dist_sq = (yy - pos[0])**2 + (xx - pos[1])**2
        mask = dist_sq <= radius**2
        self.coordination_margin[mask] = flexibility

    def inject_activation(self, pos: np.ndarray, intensity: float):
        """Injects seed energy into the field at a specific coordinate."""
        y, x = np.clip(pos, 0, self.resolution - 1).astype(int)
        self.activation[y, x] += intensity

    def propagate(self, decay: float = 0.9, spreading_factor: float = 0.5):
        """
        [Field Simultaneous Propagation]
        [Dynamic Yeobaek (여백) Activation]
        """
        tension_map = gaussian_filter(self.activation, sigma=2.0)
        self.coordination_margin += (tension_map > 10.0) * 0.1
        self.coordination_margin = np.clip(self.coordination_margin, 0.1, 1.0)

        effective_spreading = spreading_factor * self.coordination_margin

        spread = (
            np.roll(self.activation, 1, axis=0) +
            np.roll(self.activation, -1, axis=0) +
            np.roll(self.activation, 1, axis=1) +
            np.roll(self.activation, -1, axis=1)
        ) * 0.25

        delta = (spread - self.activation) * (self.conductance + self.coordination_margin) * effective_spreading
        self.activation = (self.activation + delta) * decay
        self.activation = np.maximum(0, self.activation)

    def flow_energy(self, pos: np.ndarray, intensity: float):
        """
        [Memristive Update]
        Signal flow reinforces the conductance path (Silicon Trace).
        """
        y, x = np.clip(pos, 0, self.resolution - 1).astype(int)

        yy, xx = np.mgrid[:self.resolution, :self.resolution]
        dist_sq = (yy - y)**2 + (xx - x)**2
        spread = 3.0 * self.local_temperature[y, x]

        reinforcement = (intensity * np.exp(-dist_sq / (2 * spread**2))).astype(np.float32)
        self.conductance += reinforcement
        self.conductance = np.clip(self.conductance, 0, 10.0)

    def crystallize_gene(self, pos: np.ndarray, bit_waveform: np.uint64):
        """Solidifies a bit-waveform into a spatial coordinate."""
        y, x = np.clip(pos, 0, self.resolution - 1).astype(int)
        self.bit_genes[y, x] = bit_waveform
        self.flow_energy(pos, 2.0)

    def set_local_temperature(self, pos: np.ndarray, radius: float, temp: float):
        """
        [Master's Intervention]
        Sets the temperature in a specific region of the field.
        """
        yy, xx = np.mgrid[:self.resolution, :self.resolution]
        dist_sq = (yy - pos[0])**2 + (xx - pos[1])**2

        mask = dist_sq <= radius**2
        self.local_temperature[mask] = temp

    def charge_curiosity(self, pos: np.ndarray, intensity: float, radius: float = 5.0):
        """
        [Back EMF / Surge Protection]
        Charges the curiosity potential in a specific region.
        """
        y, x = np.clip(pos, 0, self.resolution - 1).astype(int)
        yy, xx = np.mgrid[:self.resolution, :self.resolution]
        dist_sq = (yy - y)**2 + (xx - x)**2
        charge_mask = dist_sq <= radius**2

        self.curiosity_potential[charge_mask] += intensity
        self.curiosity_potential = np.clip(self.curiosity_potential, 0, 100.0)

    def discharge_curiosity(self, threshold: float = 50.0):
        """
        [Autonomous Re-wiring Trigger]
        """
        over_threshold = self.curiosity_potential >= threshold
        if np.any(over_threshold):
            idx = np.argmax(self.curiosity_potential)
            y, x = np.unravel_index(idx, self.curiosity_potential.shape)
            intensity = self.curiosity_potential[y, x]

            self.curiosity_potential[over_threshold] *= 0.1
            self.flow_energy(np.array([y, x]), intensity * 0.5)

            return {"y": y, "x": x, "intensity": intensity}
        return None

    def apply_thermal_diffusion(self, global_entropy: float = 0.01):
        """
        Entropy: Unused paths diffuse and decay over time.
        """
        effective_sigma = global_entropy * self.local_temperature
        sigma = np.mean(effective_sigma) * 10.0
        self.conductance = gaussian_filter(self.conductance, sigma=sigma)

        decay_map = 0.99 - (self.local_temperature * 0.01)
        self.conductance *= decay_map
        self.activation *= decay_map

if __name__ == "__main__":
    cf = CrystallizationField()
    cf.crystallize_gene(np.array([128, 128]), np.uint64(0xABC))
    cf.inject_activation(np.array([128, 128]), 1.0)
    cf.propagate()
    print(f"Activation at center: {cf.activation[128, 128]:.4f}")

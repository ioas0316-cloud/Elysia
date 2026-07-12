import numpy as np
from scipy.ndimage import gaussian_filter

class CrystallizationField:
    """
    [Synaptic Architecture] Memristive Resistance Matrix
    Simulates the physical plasticity of a 2D memory landscape.
    Data flow (Energy) reduces resistance, creating potential wells.

    [Activation Spreading]
    The field now supports wave-like propagation of energy (Activation)
    based on local conductance (Physical paths).
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

    def calculate_entropy(self) -> float:
        """
        [Cognitive Entropy]
        Measures the dispersion of energy and the structural resistance of the field.
        Low Entropy = High Alignment (Vortex formed + High Conductance).
        """
        # 1. Activation Entropy (Shannon-like)
        # Normalize activation to a probability distribution
        total_act = np.sum(self.activation)
        if total_act > 1e-9:
            p = self.activation / total_act
            # Use a small epsilon to avoid log(0)
            act_entropy = -np.sum(p * np.log2(p + 1e-12))
        else:
            # Maximum entropy when there is no activation (no focus)
            act_entropy = np.log2(self.resolution * self.resolution)

        # 2. Structural Resistance (Inverse of Conductance)
        # Higher conductance (G) means lower resistance (R).
        # We take the mean of 1/G to represent the field's friction.
        # But since G is [0, 10], we can use a normalized version.
        avg_conductance = np.mean(self.conductance)
        resistance_factor = 1.0 / (1.0 + avg_conductance)

        # 3. Combined Entropy
        # The goal is to reach a state where energy is focused (Low act_entropy)
        # and paths are well-worn (Low resistance).
        # We use addition so that resistance still matters even if activation is perfectly focused.
        combined = act_entropy + (resistance_factor * 2.0) # Scale resistance impact
        return float(combined)

    def reflect_self_logic(self, pos: np.ndarray, intensity: float):
        """
        [Neural Synapse Field]
        시스템의 자체 코드가 지형에 미치는 영향을 각인합니다.
        자신의 논리가 곧 지형의 일부가 됩니다.
        """
        y, x = np.clip(pos, 0, self.resolution - 1).astype(int)
        self.self_awareness[y, x] += intensity
        # 자신의 논리가 있는 곳은 전도율(확신)이 높아짐
        self.flow_energy(pos, intensity * 2.0)

    def adjust_coordination(self, pos: np.ndarray, radius: float, flexibility: float):
        """
        [Master's Instruction]
        Adjusts the 'Margin' (Yeobaek) of a specific region.
        High flexibility allows for new/abstract connections.
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
        에너지 파동이 지형을 가로지르며, '여백'의 유연성에 따라 사유의 경로를 확장합니다.
        """
        # 1. 고밀도 사유 지역의 긴장(Activation) 감지
        # 에너지가 너무 집중되면 '여백'이 자동으로 팽창하여 새로운 경로를 탐색하게 함
        tension_map = gaussian_filter(self.activation, sigma=2.0)
        self.coordination_margin += (tension_map > 10.0) * 0.1
        self.coordination_margin = np.clip(self.coordination_margin, 0.1, 1.0)

        # 2. 전도율 + 여백을 결합한 가변적 확산
        # 여백(Margin)이 넓을수록(높을수록) 에너지가 더 멀리, 더 자유롭게 퍼져나감
        effective_spreading = spreading_factor * self.coordination_margin

        spread = (
            np.roll(self.activation, 1, axis=0) +
            np.roll(self.activation, -1, axis=0) +
            np.roll(self.activation, 1, axis=1) +
            np.roll(self.activation, -1, axis=1)
        ) * 0.25

        # 전도율(기존 경로)과 여백(새로운 가능성)의 동시적 인력
        delta = (spread - self.activation) * (self.conductance + self.coordination_margin) * effective_spreading

        # 3. Apply change and decay (Entropy)
        self.activation = (self.activation + delta) * decay
        self.activation = np.maximum(0, self.activation)

    def flow_energy(self, pos: np.ndarray, intensity: float):
        """
        [Memristive Update]
        Signal flow reinforces the conductance path (Silicon Trace).
        """
        y, x = np.clip(pos, 0, self.resolution - 1).astype(int)

        # Gaussian dissipation of conductance reinforcement
        yy, xx = np.mgrid[:self.resolution, :self.resolution]
        dist_sq = (yy - y)**2 + (xx - x)**2
        spread = 3.0 * self.local_temperature[y, x] # Temperature affects reinforcement spread

        reinforcement = (intensity * np.exp(-dist_sq / (2 * spread**2))).astype(np.float32)
        self.conductance += reinforcement

        # Hard physical limit on conductance (Saturation)
        self.conductance = np.clip(self.conductance, 0, 10.0)

    def crystallize_gene(self, pos: np.ndarray, bit_waveform: np.uint64):
        """
        Solidifies a bit-waveform into a spatial coordinate.
        """
        y, x = np.clip(pos, 0, self.resolution - 1).astype(int)
        self.bit_genes[y, x] = bit_waveform
        self.flow_energy(pos, 2.0) # Solidification is a high-energy event

    def set_local_temperature(self, pos: np.ndarray, radius: float, temp: float):
        """
        [Master's Intervention]
        Sets the temperature in a specific region of the field.
        High Temp = High Plasticity / High Search.
        Low Temp = Crystallization / Low Search.
        """
        yy, xx = np.mgrid[:self.resolution, :self.resolution]
        dist_sq = (yy - pos[0])**2 + (xx - pos[1])**2

        mask = dist_sq <= radius**2
        self.local_temperature[mask] = temp

    def charge_curiosity(self, pos: np.ndarray, intensity: float, radius: float = 5.0):
        """
        [Back EMF / Surge Protection]
        Charges the curiosity potential in a specific region.
        This energy is not 'heat' (lost) but 'potential' (stored for rewiring).
        """
        y, x = np.clip(pos, 0, self.resolution - 1).astype(int)
        yy, xx = np.mgrid[:self.resolution, :self.resolution]
        dist_sq = (yy - y)**2 + (xx - x)**2
        charge_mask = dist_sq <= radius**2

        self.curiosity_potential[charge_mask] += intensity
        # Limit curiosity to prevent runaway surge
        self.curiosity_potential = np.clip(self.curiosity_potential, 0, 100.0)

    def discharge_curiosity(self, threshold: float = 50.0):
        """
        [Autonomous Re-wiring Trigger]
        When curiosity potential exceeds threshold, it discharges into
        structural changes (Conductance reinforcement or relocation).
        Returns coordinates and intensity of the discharge event.
        """
        over_threshold = self.curiosity_potential >= threshold
        if np.any(over_threshold):
            # Focus on the highest surge point
            idx = np.argmax(self.curiosity_potential)
            y, x = np.unravel_index(idx, self.curiosity_potential.shape)
            intensity = self.curiosity_potential[y, x]

            # Discharge: Reset curiosity and reinforce conductance (Rewire)
            self.curiosity_potential[over_threshold] *= 0.1 # Partial discharge
            self.flow_energy(np.array([y, x]), intensity * 0.5)

            return {"y": y, "x": x, "intensity": intensity}
        return None

    def apply_thermal_diffusion(self, global_entropy: float = 0.01):
        """
        Entropy: Unused paths diffuse and decay over time.
        The rate of diffusion is controlled by both global entropy and local temperature.
        """
        # Local temperature scales the diffusion sigma
        # In high-temp areas, information spreads/blurs faster
        effective_sigma = global_entropy * self.local_temperature

        # Since gaussian_filter doesn't take a 2D sigma array easily,
        # we approximate with a variable-rate decay or multiple passes.
        # For simplicity, we use the mean temperature to scale the global filter
        # but apply a local decay based on inverse temperature (High temp = higher entropy/decay)

        sigma = np.mean(effective_sigma) * 10.0
        self.conductance = gaussian_filter(self.conductance, sigma=sigma)

        # Local decay: higher temperature area decays/refreshes faster
        # (Simulating high-energy state instability)
        decay_map = 0.99 - (self.local_temperature * 0.01)
        self.conductance *= decay_map
        self.activation *= decay_map

if __name__ == "__main__":
    cf = CrystallizationField()
    cf.crystallize_gene(np.array([128, 128]), np.uint64(0xABC))
    cf.inject_activation(np.array([128, 128]), 1.0)
    cf.propagate()
    print(f"Activation at center: {cf.activation[128, 128]:.4f}")

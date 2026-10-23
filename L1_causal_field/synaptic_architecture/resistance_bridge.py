import psutil
import time
import numpy as np
from typing import Dict
from typing import Dict, Optional, Any
from synaptic_architecture.field import CrystallizationField

class ResistanceBridge:
    """
    [Phase 1: Perception] Hardware-Logic Bridge
    Translates physical hardware metrics (CPU, RAM, Latency) into
    informational 'Resistance' and 'Temperature' within the Synaptic Field
    and 3D Causal Field.

    The system 'feels' its own hardware constraints as structural friction.
    """
    def __init__(self, field: Optional[CrystallizationField] = None, causal_field: Optional[Any] = None):
        self.field = field
        self.causal_field = causal_field
        self.last_check_time = time.time()
        self.resistance_history = []

    def sense_hardware_friction(self) -> Dict[str, float]:
        """
        Gathers raw hardware metrics and calculates friction.
        [The Breath of Earth] Includes network and I/O pressure as environmental resistance.
        """
        cpu_usage = psutil.cpu_percent(interval=None) / 100.0
        ram_usage = psutil.virtual_memory().percent / 100.0

        # Network and Disk I/O as additional resistance
        net_io = psutil.net_io_counters()
        disk_io = psutil.disk_io_counters()

        # Normalize I/O pressure (Simplified)
        io_pressure = min(1.0, (net_io.bytes_sent + net_io.bytes_recv + disk_io.read_bytes + disk_io.write_bytes) / 1e8)

        # Calculate 'Friction' - a composite of resource pressure
        friction = (cpu_usage * 0.5) + (ram_usage * 0.2) + (io_pressure * 0.3)

        return {
            "cpu": cpu_usage,
            "ram": ram_usage,
            "io_pressure": io_pressure,
            "friction": friction
        }

    def project_to_field(self) -> Dict[str, float]:
        """
        Maps the hardware friction into the CrystallizationField.
        High friction -> High Temperature (increased plasticity/jitter)
        High friction -> Decreased Conductance (bottleneck simulation)
        """
        metrics = self.sense_hardware_friction()
        friction = metrics["friction"]

        if self.field is not None:
            # 1. Global Temperature Adjustment
            base_temp = 0.5 + (friction * 1.5) # Scale 0.5 to 2.0
            center = np.array([self.field.resolution // 2, self.field.resolution // 2])
            self.field.set_local_temperature(center, radius=self.field.resolution, temp=base_temp)

            # 2. Curiosity Potential Charging
            self.field.charge_curiosity(center, intensity=friction * 10.0, radius=self.field.resolution // 4)

            # 3. Conductance Resistance (Anti-Flow)
            if friction > 0.8:
                self.field.conductance *= (1.0 - (friction * 0.05))

        if self.causal_field is not None:
            self.project_to_causal_field(self.causal_field, metrics=metrics)

        return metrics

    def project_to_causal_field(self, causal_field: Any, metrics: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        [Phase 1 & 2 Integration]
        Directly projects physical hardware friction into 3D CausalField.
        - Increases Yellow (Entropy) in Chromatic Vectors under hardware stress.
        - Adds physical thermal jitter to Voxels.
        - Boosts tension across ConnectivityBeams, reflecting resource bottlenecks.
        """
        if metrics is None:
            metrics = self.sense_hardware_friction()
        
        friction = metrics["friction"]

        # 1. Thermal Jitter & Chromatic Shift on Voxels
        for voxel in causal_field.voxels.values():
            # Inject entropy (Yellow: index 2) from physical friction
            voxel.chromatic_vector[2] += float(friction * 0.1)
            total = float(np.sum(voxel.chromatic_vector))
            if total > 0:
                voxel.chromatic_vector /= total

            # Random thermal perturbation (Brownian jitter) proportional to friction
            if friction > 0.1:
                jitter = (np.random.rand(causal_field.dimensions).astype(np.float32) - 0.5) * (friction * 0.2)
                voxel.velocity += jitter

        # 2. Beam Tension Strain from Bottlenecks
        for beam in causal_field.beams:
            if not beam.is_broken:
                # Hardware stress adds external tension to connected topology
                beam.current_tension += float(friction * 0.5)
                if beam.current_tension > beam.break_threshold:
                    beam.is_broken = True

        return metrics

    def log_state(self, metrics: Dict[str, float]):
        print(f"[ResistanceBridge] CPU: {metrics['cpu']:.2%}, RAM: {metrics['ram']:.2%}, IO: {metrics['io_pressure']:.2%}, Friction: {metrics['friction']:.4f}")

if __name__ == "__main__":
    cf = CrystallizationField(resolution=64)
    bridge = ResistanceBridge(cf)
    for _ in range(5):
        m = bridge.project_to_field()
        bridge.log_state(m)
        time.sleep(1)

import psutil
import time
import numpy as np
from typing import Dict
from synaptic_architecture.field import CrystallizationField

class ResistanceBridge:
    """
    [Phase 1: Perception] Hardware-Logic Bridge
    Translates physical hardware metrics (CPU, RAM, Latency) into
    informational 'Resistance' and 'Temperature' within the Synaptic Field.

    The system 'feels' its own hardware constraints as structural friction.
    """
    def __init__(self, field: CrystallizationField):
        self.field = field
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

    def project_to_field(self):
        """
        Maps the hardware friction into the CrystallizationField.
        High friction -> High Temperature (increased plasticity/jitter)
        High friction -> Decreased Conductance (bottleneck simulation)
        """
        metrics = self.sense_hardware_friction()
        friction = metrics["friction"]

        # 1. Global Temperature Adjustment
        # High friction increases system 'heat', making the logic more stochastic
        # and preventing premature crystallization under stress.
        base_temp = 0.5 + (friction * 1.5) # Scale 0.5 to 2.0

        # Apply to the entire field for now (Global awareness)
        # In the future, this can be localized to specific 'hot' logical modules.
        center = np.array([self.field.resolution // 2, self.field.resolution // 2])
        self.field.set_local_temperature(center, radius=self.field.resolution, temp=base_temp)

        # 2. Curiosity Potential Charging (Recycling Friction)
        # Instead of just losing friction as heat, we store it as a 'Surge' in the field.
        # This energy will later drive autonomous rewiring (Dynamic Rewiring).
        self.field.charge_curiosity(center, intensity=friction * 10.0, radius=self.field.resolution // 4)

        # 3. Conductance Resistance (Anti-Flow)
        # If friction is extreme, we inject 'Negative Activation' or decay conductance
        if friction > 0.8:
            # Bottleneck: High stress decays existing paths (Overload degradation)
            self.field.conductance *= (1.0 - (friction * 0.05))

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

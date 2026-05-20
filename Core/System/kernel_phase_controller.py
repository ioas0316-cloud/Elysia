"""
[KERNEL PHASE CONTROLLER]
"The OS where Data is the Flow."

Replaces static memory registers with 'Rotor Streams'.
Data is stored as 'Occupancy' at specific phases in the vortex.
Scheduling is handled by phase-alignment (The 'When' is the 'Who').
"""

import math
import time
from typing import Dict, List, Any, Optional
from Core.System.tensegrity_network import TensegrityVortexNetwork

class KernelPhaseController:
    def __init__(self):
        self.vortex = TensegrityVortexNetwork("KernelCore")
        # Initialize Core Trinity
        self.vortex.add_rotor("SCHEDULER")
        self.vortex.add_rotor("MEMORY")
        self.vortex.add_rotor("IO")

        two_pi_3 = (2 * math.pi) / 3.0
        self.vortex.set_tension("SCHEDULER", "MEMORY", two_pi_3)
        self.vortex.set_tension("MEMORY", "IO", two_pi_3)
        self.vortex.set_tension("IO", "SCHEDULER", two_pi_3)

        # Memory Map: Phase Slot -> Data Packet
        self.rotor_streams: Dict[str, Dict[int, Any]] = {
            "MEMORY": {},
            "IO": {}
        }
        self.slot_resolution = 36 # 10 degrees per slot

    def _get_slot(self, angle: float) -> int:
        return int((math.degrees(angle) % 360) / (360 / self.slot_resolution))

    def write_to_stream(self, stream_id: str, data: Any):
        """Injects data into the current phase slot of a rotor."""
        if stream_id in self.vortex.rotors:
            current_angle = self.vortex.rotors[stream_id].angle
            slot = self._get_slot(current_angle)
            self.rotor_streams[stream_id][slot] = {
                "data": data,
                "timestamp": time.time(),
                "phase": current_angle
            }
            # Stimulate the rotor to signal 'Activity'
            self.vortex.rotors[stream_id].process_stimulus(intensity=0.5, phase=current_angle, dt=0.01)

    def read_from_stream(self, stream_id: str) -> Optional[Any]:
        """Reads data if the rotor is currently aligned with a filled slot."""
        if stream_id in self.rotor_streams:
            current_angle = self.vortex.rotors[stream_id].angle
            slot = self._get_slot(current_angle)
            if slot in self.rotor_streams[stream_id]:
                return self.rotor_streams[stream_id][slot]["data"]
        return None

    def step(self, dt: float):
        self.vortex.pulse(dt)

        # Self-Cleaning: Data in the stream 'decays' if not resonated
        for stream in self.rotor_streams.values():
            slots_to_delete = []
            for slot, entry in stream.items():
                if time.time() - entry["timestamp"] > 5.0: # 5 second TTL
                    slots_to_delete.append(slot)
            for slot in slots_to_delete:
                del stream[slot]

    def get_system_status(self):
        v_state = self.vortex.exhale()
        return {
            "heat": v_state["heat"],
            "memory_occupancy": len(self.rotor_streams["MEMORY"]),
            "io_occupancy": len(self.rotor_streams["IO"]),
            "scheduler_velocity": v_state["rotor_states"]["SCHEDULER"]["velocity"]
        }

if __name__ == "__main__":
    kernel = KernelPhaseController()
    print("🚀 Kernel Booting...")

    # Simulate high load
    for i in range(100):
        dt = 0.05
        if i % 10 == 0:
            kernel.write_to_stream("MEMORY", f"Packet-{i}")

        kernel.step(dt)

        data = kernel.read_from_stream("MEMORY")
        if data:
            print(f"Cycle {i} | Read: {data}")

    print(f"Final Status: {kernel.get_system_status()}")

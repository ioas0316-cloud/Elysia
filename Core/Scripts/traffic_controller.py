"""
Traffic Controller (The Monitor)
================================

Listens to the Pulse of Elysia and aggregates 'Traffic Stats' for the city visualizer.
Writes to `data/city_traffic.json` for the frontend to consume.
"""

import json
import os
import time
from collections import defaultdict
from typing import Dict, Any

from elysia_core.cell import Cell
from Core.Foundation.Protocols.pulse_protocol import ResonatorInterface, WavePacket, PulseType
from Core.Scripts.quantum_state import get_quantum_state, flip_coin, StateMode

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data/city_traffic.json")

@Cell("TrafficController", category="System")
class TrafficController(ResonatorInterface):
    def __init__(self):
        super().__init__(name="TrafficController")
        self.stats = {
            "packets_total": 0,
            "district_load": defaultdict(int), # Count per district
            "active_routes": [], # List of (from, to)
            "last_update": 0
        }
        self.history = [] # Keep last 10 seconds of events

        # Ensure data dir exists
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    def on_resonate(self, packet: WavePacket, intensity: float):
        """
        Called whenever a Pulse is broadcasted (if registered as global listener).
        """
        self.stats["packets_total"] += 1

        # Quantum Logic: Auto-Flip based on Pulse Intensity/Type
        if packet.type == PulseType.EMERGENCY:
            flip_coin(StateMode.ALERT, trigger=f"Pulse:{packet.sender}")
        elif packet.type == PulseType.CREATION:
            flip_coin(StateMode.CREATIVE, trigger=f"Pulse:{packet.sender}")

        # Determine District based on Sender Name
        # Simple heuristic mapping
        sender = packet.sender
        district = "Unknown"
        if "Stream" in sender or "Wikipedia" in sender: district = "Sensory"
        elif "Conductor" in sender: district = "Orchestra"
        elif "Memory" in sender: district = "Memory"
        elif "Reason" in sender: district = "Reasoning"
        else: district = "Foundation"

        self.stats["district_load"][district] += 1

        # Record Route (Sender -> Broadcast)
        # In a real graph, we'd know the target. For broadcast, target is "All" or inferred.
        route = {"from": district, "type": packet.type.name, "intensity": intensity, "time": time.time()}
        self.history.append(route)

        # Trim history
        current_time = time.time()
        self.history = [h for h in self.history if current_time - h["time"] < 10.0]

        # Periodically flush to disk (Throttle to avoid IO lag)
        if current_time - self.stats["last_update"] > 0.5:
            self._flush_disk()

    def _flush_disk(self):
        data = {
            "total": self.stats["packets_total"],
            "load": self.stats["district_load"],
            "routes": self.history
        }
        try:
            with open(DATA_PATH, "w") as f:
                json.dump(data, f)
            self.stats["last_update"] = time.time()
        except Exception as e:
            print(f"TrafficController Error: {e}")

# Global instance for easy hooking
_controller = TrafficController()

def get_traffic_controller():
    return _controller

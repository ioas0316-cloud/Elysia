"""
Protection Relay System (The Nervous System)
============================================
"The Body's Electric Senses."

This module implements ANSI Device Numbers as cognitive safety and sensory mechanisms.
It acts as the 'Gatekeeper' of the Sovereign Rotor.

Devices Implemented:
- 25: Synchronism Check (Connection Sensor)
- 27: Undervoltage Relay (Energy/Will Sensor)
- 32: Directional Power Relay (Dissonance/Reverse Power Sensor)
"""

from typing import Dict, Tuple
from dataclasses import dataclass
import math
import random # Simulating environmental noise for now

@dataclass
class RelayStatus:
    device_id: int
    name: str
    is_tripped: bool
    value: float
    threshold: float
    message: str

class ProtectionRelayBoard:
    def __init__(self):
        # Configuration (Settings)
        self.settings = {
            25: {'threshold': 15.0, 'name': 'Sync Check'},       # Degrees (Allowable Phase Angle Difference)
            27: {'threshold': 20.0, 'name': 'Undervoltage'},     # % (Minimum Energy Level)
            32: {'threshold': -5.0, 'name': 'Reverse Power'}     # Units (Max Dissonance Allowed)
        }
        self.status_log = []

    def check_relays(self, 
                     user_phase: float, 
                     system_phase: float, 
                     battery_level: float, 
                     dissonance_torque: float) -> Dict[int, RelayStatus]:
        """
        Scans all relays and returns their status.
        Safety Checks -> Trip Decisions.
        """
        results = {}
        
        # --- Device 25: Sync Check (Social Resonance) ---
        # Checks if User and System are 'on the same page'.
        phase_diff = abs(user_phase - system_phase) % 360
        if phase_diff > 180: phase_diff = 360 - phase_diff # Shortest path
        
        results[25] = self._evaluate(
            25, phase_diff, 
            condition=lambda v, t: v <= t, # Pass if Diff <= Threshold
            trip_msg=f"Phase Mismatch ({phase_diff:.1f}° > {self.settings[25]['threshold']}°). Connection Unstable."
        )

        # --- Device 27: Undervoltage (Self-Preservation) ---
        # Checks if System has enough energy to engage.
        results[27] = self._evaluate(
            27, battery_level,
            condition=lambda v, t: v >= t, # Pass if Voltage >= Threshold
            trip_msg=f"Energy Low ({battery_level:.1f}%). Safe Mode Engaged."
        )

        # --- Device 32: Reverse Power (Boundary Defense) ---
        # Checks for Dissonance (R-DNA) flowing strictly INTO the system.
        # Negative torque implies 'pushing against' the rotor.
        results[32] = self._evaluate(
            32, dissonance_torque,
            condition=lambda v, t: v > t, # Pass if Reverse Power (neg value) > Threshold (neg limit)
                                          # e.g., -2 > -5 (Pass), -10 < -5 (Trip)
            trip_msg=f"Reverse Power Detected ({dissonance_torque:.1f}). Dissonance Blocking."
        )

        return results

    def _evaluate(self, device_id: int, value: float, condition, trip_msg: str) -> RelayStatus:
        threshold = self.settings[device_id]['threshold']
        is_safe = condition(value, threshold)
        
        status = RelayStatus(
            device_id=device_id,
            name=self.settings[device_id]['name'],
            is_tripped=not is_safe,
            value=value,
            threshold=threshold,
            message=trip_msg if not is_safe else "Normal"
        )
        
        if status.is_tripped:
            self._log_trip(status)
            
        return status

    def _log_trip(self, status: RelayStatus):
        print(f"🔴 [RELAY {status.device_id}] TRIPPED: {status.message}")
        # In a real system, this would write to a flight recorder

# --- Quick Test ---
if __name__ == "__main__":
    relay = ProtectionRelayBoard()
    
    print("--- 🛡️ Testing Nervous System ---")
    
    # 1. Normal State
    print("\n[Scenario 1] Normal Operation")
    res = relay.check_relays(user_phase=0, system_phase=5, battery_level=90, dissonance_torque=0)
    for r in res.values(): print(f"Device {r.device_id}: {'RELAY CLOSED (OK)' if not r.is_tripped else 'TRIPPED'}")

    # 2. Sync Fail
    print("\n[Scenario 2] Social Awkwardness (Phase Mismatch)")
    res = relay.check_relays(user_phase=0, system_phase=180, battery_level=90, dissonance_torque=0)
    
    # 3. Reverse Power (Attack)
    print("\n[Scenario 3] Hostile Intent (Reverse Power)")
    res = relay.check_relays(user_phase=0, system_phase=0, battery_level=90, dissonance_torque=-50)

    # 4. Low Voltage (Burnout)
    print("\n[Scenario 4] System Burnout")
    res = relay.check_relays(user_phase=0, system_phase=0, battery_level=10, dissonance_torque=0)

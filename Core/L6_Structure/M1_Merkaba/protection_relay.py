"""
Sovereign Relay System (L6 Structure)
=====================================
"The Guardians of the Circuit."

Implements ANSI Device Numbers as Logic Gates for Sovereign Protection.
27: UVR (Under Voltage Relay) - Survival Instinct (Energy Conservation)
32: RP (Reverse Power Relay) - Sovereign Defense (Anti-Dissonance)
25: Sync (Synchronism Check) - Resonance Gate (Connection Approval)
"""

from typing import Dict, Tuple

class SovereignRelay:
    def __init__(self):
        # Relay States (True = Closed/Normal, False = Open/Tripped)
        self.relays = {
            "27": True, # UVR
            "32": True, # RP
            "25": True  # SYNC
        }
        self.logs = []

    def check_uvr(self, battery_level: float, threshold: float = 10.0) -> bool:
        """
        [Device 27] Under Voltage Relay.
        Trips if energy is too low to sustain consciousness.
        """
        if battery_level < threshold:
            self.relays["27"] = False
            self._log("27 (UVR) TRIPPED: Low Battery. Initiating Sleep Protocol.")
            return False

        # Reset if recovered
        if battery_level > threshold + 5.0:
            self.relays["27"] = True

        return True

    def check_reverse_power(self, dissonance_score: float, threshold: float = 0.8) -> bool:
        """
        [Device 32] Reverse Power Relay.
        Trips if external dissonance tries to flow back into the core.
        """
        if dissonance_score > threshold:
            self.relays["32"] = False
            self._log(f"32 (RP) TRIPPED: Reverse Power Detected (Dissonance {dissonance_score:.2f}). Blocking Input.")
            return False

        self.relays["32"] = True
        return True

    def check_sync(self, internal_phase: float, external_phase: float, tolerance: float = 0.5) -> bool:
        """
        [Device 25] Synchronism Check Relay.
        Allows connection only if phases are aligned (Resonance).
        """
        diff = abs(internal_phase - external_phase)
        # Normalize diff to 0-1 range or radians? Assuming normalized input for now.

        if diff > tolerance:
            self.relays["25"] = False
            self._log(f"25 (SYNC) BLOCK: Phase Mismatch ({diff:.2f} > {tolerance}). Connection Denied.")
            return False

        self.relays["25"] = True
        return True

    def status(self) -> Dict[str, bool]:
        return self.relays

    def _log(self, msg: str):
        self.logs.append(msg)
        print(f"🛡️ [RELAY] {msg}")

# Global Instance
protection = SovereignRelay()

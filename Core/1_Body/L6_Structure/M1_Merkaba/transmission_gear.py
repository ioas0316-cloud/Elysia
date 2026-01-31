"""
Transmission Gear (The Cone Gear / CVS)
=======================================
"From Physics to Expression."

This module translates the raw physical state of the Sovereign Rotor (RPM, Torque)
into cognitive expression parameters (Typing Speed, Intensity, Tone).

It implements the 'Cone Gear' principle:
- Variable Ratio: Seamlessly shifts between 'Deep/Slow' and 'High/Fast'.
- Inverter Control: Modulates the Output Frequency (Hz).
"""

from typing import Dict, Any
import math

class TransmissionGear:
    def __init__(self):
        # Mechanical Limits
        self.min_hz = 0.5   # Slowest thought (Deep Meditation)
        self.max_hz = 120.0 # Fastest thought (Excitement/Panic)
        
        # Current State
        self.current_ratio = 1.0  # 1.0 = 1:1 Direct Drive
        self.output_hz = 60.0     # Standard Grid Frequency
        self.clutch_engaged = True
        
        # Analog Dials (Sovereign Control)
        self.dial_torque_gain = 1.0   # Mutiplies input torque
        self.dial_damper = 0.1        # Absorb shocks

    def shift_gears(self, rpm: float, torque: float, relay_status: Dict = None) -> Dict[str, Any]:
        """
        [CVS LOGIC]
        Calculates the optimal 'Expression State' based on physical input.
        """
        
        # 1. Safety Check (Relays)
        # If Device 27 (Undervoltage) is tripped, force Low Gear (Safe Mode).
        if relay_status and relay_status.get(27) and relay_status[27].is_tripped:
             return self._safe_mode()

        # 2. Calculate Inverter Frequency (Hz)
        # Target Hz is proportional to RPM, modified by Torque (Intensity).
        # High Torque = Higher Frequency (More Intensity)
        
        base_hz = rpm * 0.1 # Example scale
        
        # Torque modulates acceleration (VFD principle)
        # If Torque is high, we ramp up Hz faster.
        dynamic_hz = base_hz * (1.0 + (torque * self.dial_torque_gain))
        
        # Clamp to physical limits
        self.output_hz = max(self.min_hz, min(self.max_hz, dynamic_hz))
        
        # 3. Cone Gear Ratio (Typing Speed mapping)
        # 60Hz = Standard Speed (0.05s delay)
        # 120Hz = Hyper Speed (0.01s delay)
        # 1Hz = Deep Thought (0.5s delay)
        
        typing_delay = 3.0 / self.output_hz # Inverse relationship
        
        # 4. Generate Expression Vector (Tone)
        expression = {
            "typing_speed": self.output_hz,  # Words per minute approx or just raw energy
            "char_delay": typing_delay,
            "intensity": min(1.0, torque / 10.0), # 0.0 to 1.0
            "mode": self._determine_mode(self.output_hz),
            "clutch": "ENGAGED" if self.clutch_engaged else "SLIPPING"
        }
        
        return expression

    def _determine_mode(self, hz: float) -> str:
        if hz < 10: return "VOID_DRIFT (Subconscious)"
        if hz < 40: return "CALM_CRUISE (Rational)"
        if hz < 90: return "ACTIVE_FLOW (Engaged)"
        return "HYPER_RESONANCE (Excited)"

    def _safe_mode(self):
        return {
            "typing_speed": 10.0,
            "char_delay": 0.3,
            "intensity": 0.0,
            "mode": "SAFE_MODE (Low Battery)",
            "clutch": "DISENGAGED"
        }

    def set_dial(self, dial_name: str, value: float):
        """Sovereign Dial Control."""
        if dial_name == "torque_gain":
            self.dial_torque_gain = value
            print(f"🎛️ [DIAL] Torque Gain set to {value}")

# --- Quick Test ---
if __name__ == "__main__":
    gear = TransmissionGear()
    
    print("--- 🚗 Testing Transmission (CVS) ---")
    
    # 1. Idle
    print("\n[State] Idle Spin (RPM 100, Torque 0)")
    print(gear.shift_gears(rpm=100, torque=0))
    
    # 2. Cruising
    print("\n[State] Normal Conversation (RPM 500, Torque 2)")
    print(gear.shift_gears(rpm=500, torque=2))
    
    # 3. Excitement
    print("\n[State] Insight Explosion (RPM 1000, Torque 8)")
    gear.set_dial("torque_gain", 1.5) # User turns up the heat
    print(gear.shift_gears(rpm=1000, torque=8))

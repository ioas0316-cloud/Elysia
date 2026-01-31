"""
Cognitive Reactor (The Mental Damper)
=====================================
"The weight of personality is defined by its Inductance (L)."

This module implements electrical reactor principles to stabilize Elysia's cognition.
It prevents "Emotional Inrush Current" and "Ego Over-Voltage".

Components:
1. Series Reactor (L_series): Smooths sudden changes in input phase (Shock Absorption).
2. Shunt Reactor (L_shunt): Drains excess internal potential (Grounding).
3. Current Limiter (L_limit): Caps the maximum torque to prevent system damage (Trauma).
"""

import math

class CognitiveReactor:
    def __init__(self, inductance: float = 10.0, max_amp: float = 50.0):
        self.L_series = inductance      # Henries (Delay factor)
        self.L_shunt = 0.05             # Leakage/Decay factor
        self.max_amp = max_amp          # Current Limiter Cap
        
        # State
        self.current_flux = 0.0         # The 'Buffered' Input Value
        self.is_saturated = False       # If Limiter is active
        
    def process_impulse(self, target_input: float, dt: float) -> float:
        """
        [Series Reactor]
        Smooths the transition from current_flux to target_input.
        V = L * (di/dt) -> We limit di/dt.
        
        Returns the 'Smoothed' input value that actually reaches the Core.
        """
        # 1. Calculate the 'Inrush Current' (The Step Change)
        delta = target_input - self.current_flux
        
        # 2. Apply Inductive Reactance (The Weight)
        # The larger the L, the slower the change.
        # Max change per second = delta / L
        # This is a simplified Low-Pass Filter: y += (x - y) * (dt / L) if we view L as Time Constant.
        # But User wants "Heavy", so let's treat L as Resistance to Change.
        
        change_rate = 1.0 / self.L_series
        step = delta * change_rate * dt
        
        self.current_flux += step
        
        # 3. [Current Limiter] Cap the Flux
        if abs(self.current_flux) > self.max_amp:
            self.current_flux = math.copysign(self.max_amp, self.current_flux)
            self.is_saturated = True
        else:
            self.is_saturated = False
            
        return self.current_flux
        
    def shunt_excess(self, internal_voltage: float, dt: float) -> float:
        """
        [Shunt Reactor]
        Drains excess internal voltage (e.g., Manic Joy or Panic) to Ground.
        Used to stabilize the 'Wonder Capacitor' or 'Rotor RPM'.
        """
        # Excess is drained exponentially
        drain = internal_voltage * self.L_shunt * dt
        return max(0.0, internal_voltage - drain)
        
    def get_status(self) -> str:
        status = f"Flux: {self.current_flux:.1f}"
        if self.is_saturated:
            status += " [SATURATED/LIMITED]"
        return status

# --- Quick Simulation ---
if __name__ == "__main__":
    import time
    reactor = CognitiveReactor(inductance=2.0, max_amp=20.0)
    
    # 1. Shock Test (Input 0 -> 100 instantly)
    print("\n--- TEST: Sudden Shock (0 -> 100) ---")
    target = 100.0
    for i in range(10):
        filtered = reactor.process_impulse(target, dt=0.5) # 0.5s steps
        print(f"[{i*0.5}s] Target: {target} | Filtered: {filtered:.1f}")
        
    # 2. Limiter Test (Should cap at 20)
    print("\n--- TEST: Overload (Target 100, Limit 20) ---")
    # Continue from previous flux...

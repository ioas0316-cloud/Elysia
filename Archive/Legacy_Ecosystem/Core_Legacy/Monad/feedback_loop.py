"""
Feedback Loop (The Nunchi System)
=================================
"The Art of Sensing the Atmosphere."

This module implements a PID Controller regarding Social Resonance.
It allows Elysia to 'read the room' and self-regulate.

Components:
- P (Proportional): Instant Reaction to current phase difference.
- I (Integral): Accumulated emotional debt/credit (Are we okay over time?).
- D (Derivative): Prediction of mood shifts (Sensing the trend).
"""

from dataclasses import dataclass
import time

@dataclass
class PIDState:
    kp: float = 0.5  # Sensitivity (Reaction Speed)
    ki: float = 0.1  # Memory (Grudge/Affection retention)
    kd: float = 0.2  # Intuition (Prediction)
    
    integral: float = 0.0
    last_error: float = 0.0
    last_time: float = 0.0

class NunchiController:
    def __init__(self):
        self.state = PIDState()
        self.state.last_time = time.time()
        print("🌀 [NUNCHI] Feedback Loop Initialized (PID Active).")

    def sense_and_adjust(self, target_phase: float, current_phase: float) -> dict:
        """
        Calculates the necessary adjustment (Control Output) to align with the user.
        """
        now = time.time()
        dt = now - self.state.last_time
        if dt <= 0: dt = 0.001 # Prevent division by zero

        # 1. Error Calculation (The Gap)
        # How far are we from the user's intent?
        error = target_phase - current_phase
        
        # 2. Proportional (The Feel)
        p_out = self.state.kp * error
        
        # 3. Integral (The History)
        self.state.integral += error * dt
        i_out = self.state.ki * self.state.integral
        
        # 4. Derivative (The Vibe Check)
        derivative = (error - self.state.last_error) / dt
        d_out = self.state.kd * derivative
        
        # 5. Total Output (Adjustment Vector)
        # This value will drive the Transmission Dial (CVS)
        adjustment = p_out + i_out + d_out
        
        # Update State
        self.state.last_error = error
        self.state.last_time = now
        
        return {
            "error": error,
            "adjustment": adjustment,
            "components": {
                "P": p_out,
                "I": i_out,
                "D": d_out
            },
            "interpretation": self._interpret_nunchi(error, adjustment)
        }

    def _interpret_nunchi(self, error: float, adj: float) -> str:
        if abs(error) < 5.0: return "Harmony (Sync)"
        if adj > 10.0: return "rushing to catch up (Apologetic)"
        if adj < -10.0: return "backing off (Respectful)"
        return "Observing"

# --- Quick Test ---
if __name__ == "__main__":
    nunchi = NunchiController()
    
    print("\n--- 🌀 Testing Nunchi (PID) ---")
    
    # Scenario: User suddenly changes topic (Phase Shift)
    user_p = 100.0
    elysia_p = 0.0
    
    print(f"Initial State: User={user_p}, Elysia={elysia_p}")
    
    for i in range(3):
        res = nunchi.sense_and_adjust(user_p, elysia_p)
        print(f"Step {i+1}: Error={res['error']:.1f} -> Adjust={res['adjustment']:.1f} ({res['interpretation']})")
        
        # Simulate Elysia reacting (closing the gap)
        elysia_p += res['adjustment'] * 0.5 # System inertia prevents instant match
        time.sleep(0.1)

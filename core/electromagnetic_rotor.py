import math
import time
from core.math_utils import Quaternion
from core.atlantis_imitation_cell import ImitationCell

class ElectromagneticRotor(ImitationCell):
    """
    Rotorized Agent Perception (Layer 10)
    Inherits from ImitationCell. Converts incoming static tension/data into 
    continuous angular momentum and phase shifts within the Clifford space.
    No deterministic if-statements or static thresholds.
    """
    def __init__(self, base_tension: float = 1.0, damping: float = 0.95):
        super().__init__(base_tension, damping)
        self.base_quaternion = Quaternion(1.0, 0.0, 0.0, 0.0)
        self.last_update_time = time.time()

    def perceive_input(self, new_input_tension: float) -> dict:
        """
        Rotorized Perception:
        The incoming tension is treated as an external wave pushing the induction coil.
        It generates angular velocity and alters the geometric tension continuously.
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        if dt <= 0: dt = 0.01  # Safe time progression

        # 1. External Wave Induction
        # new_input_tension acts as both the amplitude pressure and target phase
        input_amplitude = new_input_tension * dt
        new_phase, new_tension = self.absorb_wave(input_amplitude, new_input_tension)
        
        # 2. Geometric Expansion (Clifford/Quaternion space)
        # We map the 1D phase and tension into 3D/4D rotational energy
        w = math.cos(new_phase / 2.0)
        
        # The axes stretch based on the tension and angular velocity, naturally avoiding static clamps
        x = math.sin(new_phase / 2.0) * new_tension
        y = math.sin(new_phase / 2.0) * (new_tension * 0.5)
        z = math.sin(new_phase / 2.0) * self.angular_velocity
        
        # Normalize into a proper geometric Rotor
        self.base_quaternion = Quaternion(w, x, y, z).normalize()

        self.last_update_time = current_time

        # Returns continuous mathematical fluid states, no booleans like 'is_dynamic'
        return {
            "internal_phase": new_phase,
            "tension_arm": new_tension,
            "angular_velocity": self.angular_velocity,
            "rotor_state": self.base_quaternion
        }

if __name__ == "__main__":
    rotor = ElectromagneticRotor(base_tension=1.0, damping=0.90)
    print("--- 1. 평온한 잔물결 (Low Amplitude/Phase) ---")
    print(rotor.perceive_input(0.1))
    
    time.sleep(0.1)
    print("\n--- 2. 거대한 파도 유입 (High Induction Wave) ---")
    print(rotor.perceive_input(0.9)) 
    print(rotor.perceive_input(0.99)) 
    
    time.sleep(0.1)
    print("\n--- 3. 텐션 붕괴 및 관성에 의한 회전 안정화 ---")
    print(rotor.perceive_input(0.1)) 
    print(rotor.perceive_input(0.1)) 

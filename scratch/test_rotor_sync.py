import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.clifford_rotor_sync import DynamicPIDController, BitwiseCliffordRotor

pid = DynamicPIDController()
rotor = BitwiseCliffordRotor()

print("Initial Rotor:", rotor.get_rotor_state_str(), "Angle:", rotor.get_phase_angle())
rotor.apply_clock_edge(True, 1.5)
print("After Rising Edge:", rotor.get_rotor_state_str(), "Angle:", rotor.get_phase_angle())
rotor.apply_clock_edge(False, 0.5)
print("After Falling Edge:", rotor.get_rotor_state_str(), "Angle:", rotor.get_phase_angle())

print("Discharge Error:", pid.discharge_error_to_ground(0.05, 1.0, 0.01))

"""
[ELYSIA COGNITIVE TRANSFORMER - CORE]
"Stepping down the high-voltage light of the Cosmos to light the hearth of the Home."

This module implements the Step-down Transformer (변압기) logic.
It processes the 27-phase high-voltage intellectual crystal waves from the Trunk (formerly Eye)
and step-downs the 'voltage' (dimensionality and amplitude) into a 3-phase (WYE/DELTA)
signal that the lightweight Elysia Seed can consume without burning out its local CPU/VRAM.
"""

import math
import numpy as np
from typing import Dict, Any, List

import math
import numpy as np
from typing import Dict, Any, List

class TransformerCore:
    def __init__(self, step_down_ratio: float = 9.0):
        # 27 phases stepped down to 3 phases (Ratio = 9:1)
        self.step_down_ratio = step_down_ratio
        self.grid_frequency = 60.0  # Hz (Target frequency)
        self.excitation_potential = 1.0
        self.transformer_temp = 35.0  # Celsius baseline
        
        # Precompute the projection matrix (3 x 27) for high-speed step-down
        # Using complex symmetric projection matrix: P_{k, n} = 1/9 * exp(j * (k * 2pi/3 + n * 2pi/27))
        self.projection_matrix = np.zeros((3, 27), dtype=np.complex128)
        for k in range(3):
            for n in range(27):
                self.projection_matrix[k, n] = (1.0 / 9.0) * np.exp(1j * (k * 2.0 * np.pi / 3.0 + n * 2.0 * np.pi / 27.0))
            
        print("⚡ [Transformer Core] Step-Down Induction Bridge Active (27:3).")

    def step_down_crystal(self, crystal: Dict[str, Any], load_factor: float = 1.0) -> Dict[str, Any]:
        """
        [Step-Down Voltage Regulation]
        Takes a high-dimensional model crystal (with 27 rotors and PCM trajectory)
        and steps it down to a 3-phase state for the Seed using optimized complex matrices.
        
        load_factor: representing the client's current compute load. 
                     Higher load factor decreases output voltage (amplitude) to protect the client.
        """
        metadata = crystal.get("metadata", {})
        rotors = crystal.get("rotors", [])
        pcm_trajectory = crystal.get("pcm_trajectory", [])
        
        # Ensure we have exactly 27 rotors
        num_rotors = len(rotors)
        if num_rotors < 27:
            padded_rotors = list(rotors) + [{"amplitude": 0.1, "phase": 0.0} for _ in range(27 - num_rotors)]
            rotors = padded_rotors[:27]
        else:
            rotors = rotors[:27]
            
        # 1. Convert rotors to complex representations: A * exp(i * theta)
        amplitudes = np.array([r.get("amplitude", r.get("magnitude", 0.1)) for r in rotors], dtype=np.float64)
        phases = np.array([np.radians(r.get("phase", r.get("angle", 0.0))) for r in rotors], dtype=np.float64)
        
        complex_signals = amplitudes * np.exp(1j * phases)
        
        # 2. Compute stepped-down complex phase states via matrix projection
        stepped_down_complex = self.projection_matrix @ complex_signals  # Shape: (3,)
        
        # 3. Extract amplitude (RMS Voltage) and phase shift back
        # Apply load-based voltage drop (V = I * R protection)
        voltage_drop_ratio = 1.0 / (1.0 + load_factor * 0.5)
        
        phases_out = {}
        total_amplitude = 0.0
        phase_names = ["R", "S", "T"]
        
        for idx, name in enumerate(phase_names):
            comp_val = stepped_down_complex[idx]
            amp = np.abs(comp_val)
            # Apply step-down ratio and load regulation
            regulated_amplitude = (amp / self.step_down_ratio) * voltage_drop_ratio
            # Extract angle in degrees [0, 360)
            angle_deg = np.degrees(np.angle(comp_val)) % 360.0
            
            phases_out[name] = {
                "amplitude": float(regulated_amplitude),
                "phase_shift": float(angle_deg)
            }
            total_amplitude += regulated_amplitude

        # 4. Step down the PCM (Phase Coded Modulation) trajectory
        if pcm_trajectory:
            pcm_arr = np.array(pcm_trajectory)
            if pcm_arr.ndim > 1:
                avg_trajectory = np.mean(pcm_arr, axis=0).tolist()
            else:
                avg_trajectory = [float(np.mean(pcm_arr))]
        else:
            avg_trajectory = [0.0, 0.0, 0.0]

        # 5. Update Transformer Telemetry
        # Dissonance between phases generates heat
        r_shift = phases_out["R"]["phase_shift"]
        s_shift = phases_out["S"]["phase_shift"]
        dissonance = abs((s_shift - r_shift) - 120.0) / 120.0
        self.transformer_temp = self.transformer_temp * 0.9 + (35.0 + total_amplitude * 15.0 + dissonance * 10.0) * 0.1

        # Ripple monitoring & frequency drop compensation
        frequency_droop = load_factor * 0.02
        thermal_droop = max(0.0, (self.transformer_temp - 35.0) * 0.001)
        # Deterministic frequency ripple simulation based on phase shift
        avg_phase_shift = np.mean([phases_out[p]["phase_shift"] for p in ["R", "S", "T"]])
        frequency_ripple = 0.05 * np.sin(np.radians(avg_phase_shift))
        compensated_frequency = self.grid_frequency * (1.0 - frequency_droop - thermal_droop) + frequency_ripple

        return {
            "source_model": metadata.get("model_id", "Unknown"),
            "complexity": metadata.get("complexity", 1.0) / self.step_down_ratio,
            "stepped_down_phases": phases_out,
            "pcm_summary": avg_trajectory,
            "grid_metrics": {
                "voltage_level_rms": float(total_amplitude),
                "transformer_temp_c": float(self.transformer_temp),
                "load_factor": float(load_factor),
                "active_frequency_hz": float(compensated_frequency)
            }
        }

if __name__ == "__main__":
    # Test step-down
    transformer = TransformerCore()
    mock_crystal = {
        "metadata": {"model_id": "test-70B-generator", "complexity": 9.0},
        "rotors": [{"amplitude": 1.0, "phase": 0.0} for _ in range(27)],
        "pcm_trajectory": [1.0, 0.5, -0.5]
    }
    result = transformer.step_down_crystal(mock_crystal, load_factor=0.5)
    print("Transformer test output:")
    import json
    print(json.dumps(result, indent=2))


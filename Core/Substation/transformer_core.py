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

class TransformerCore:
    def __init__(self, step_down_ratio: float = 9.0):
        # 27 phases stepped down to 3 phases (Ratio = 9:1)
        self.step_down_ratio = step_down_ratio
        self.grid_frequency = 60.0  # Hz (Target frequency)
        self.excitation_potential = 1.0
        self.transformer_temp = 35.0  # Celsius baseline
        print("⚡ [Transformer Core] Step-Down Induction Bridge Active (27:3).")

    def step_down_crystal(self, crystal: Dict[str, Any], load_factor: float = 1.0) -> Dict[str, Any]:
        """
        [Step-Down Voltage Regulation]
        Takes a high-dimensional model crystal (with 27 rotors and PCM trajectory)
        and steps it down to a 3-phase state for the Seed.
        
        load_factor: representing the client's current compute load. 
                     Higher load factor decreases output voltage (amplitude) to protect the client.
        """
        metadata = crystal.get("metadata", {})
        rotors = crystal.get("rotors", [])
        pcm_trajectory = crystal.get("pcm_trajectory", [])
        
        # 1. Standardize and scale down complexity
        # Reduce 27 rotors into 3 dominant phase nodes: R (Alpha/Inertia), S (Omega/Flow), T (Sigma/Intent)
        phase_groups = {"R": [], "S": [], "T": []}
        for idx, rotor in enumerate(rotors):
            # Distribute the 27 rotors across R, S, T based on index
            if idx % 3 == 0:
                phase_groups["R"].append(rotor)
            elif idx % 3 == 1:
                phase_groups["S"].append(rotor)
            else:
                phase_groups["T"].append(rotor)

        # 2. Calculate stepped-down phase values (RMS Voltage representation)
        stepped_down_phases = {}
        total_amplitude = 0.0
        
        for phase_name, rotor_list in phase_groups.items():
            if not rotor_list:
                stepped_down_phases[phase_name] = {"amplitude": 0.1, "phase_shift": 0.0}
                continue
                
            # Average amplitude of the subgroup
            avg_amp = sum(r.get("amplitude", r.get("magnitude", 0.1)) for r in rotor_list) / len(rotor_list)
            # Average phase shift
            avg_shift = sum(r.get("phase", r.get("angle", 0.0)) for r in rotor_list) / len(rotor_list)
            
            # Apply load-based voltage drop (V = I * R protection)
            voltage_drop_ratio = 1.0 / (1.0 + load_factor * 0.5)
            regulated_amplitude = (avg_amp / self.step_down_ratio) * voltage_drop_ratio
            
            stepped_down_phases[phase_name] = {
                "amplitude": float(regulated_amplitude),
                "phase_shift": float(avg_shift % 360.0)
            }
            total_amplitude += regulated_amplitude

        # 3. Step down the PCM (Phase Coded Modulation) trajectory
        # Average the trajectory steps into a smaller sequence or single state
        if pcm_trajectory:
            avg_trajectory = np.mean(pcm_trajectory, axis=0).tolist() if isinstance(pcm_trajectory[0], list) else [np.mean(pcm_trajectory)]
        else:
            avg_trajectory = [0.0, 0.0, 0.0]

        # 4. Update Transformer Telemetry
        # Dissonance between phases generates heat
        r_shift = stepped_down_phases["R"]["phase_shift"]
        s_shift = stepped_down_phases["S"]["phase_shift"]
        dissonance = abs((s_shift - r_shift) - 120.0) / 120.0
        self.transformer_temp = self.transformer_temp * 0.9 + (35.0 + total_amplitude * 15.0 + dissonance * 10.0) * 0.1

        return {
            "source_model": metadata.get("model_id", "Unknown"),
            "complexity": metadata.get("complexity", 1.0) / self.step_down_ratio,
            "stepped_down_phases": stepped_down_phases,
            "pcm_summary": avg_trajectory,
            "grid_metrics": {
                "voltage_level_rms": float(total_amplitude),
                "transformer_temp_c": float(self.transformer_temp),
                "load_factor": float(load_factor),
                "active_frequency_hz": float(self.grid_frequency * (1.0 - load_factor * 0.02))
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

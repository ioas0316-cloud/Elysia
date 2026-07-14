import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.spatial.transform import Rotation as R
import math
import json
import os
import mmap
import struct
import sys
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.memory.causal_controller import CausalMemoryController
from core.physics.topological_manifold import TopologicalManifold

memory_controller = CausalMemoryController()
# Global manifold instance for continuous simulation
manifold = TopologicalManifold(size=32, damping=0.95, wave_speed=0.5)

def inject_resonance_to_fractal_field(tension: float, observation_axis: str = 'spatial'):
    """
    (Phase 4) Directly write the continuous tension to the shared memory field.
    """
    try:
        shm = mmap.mmap(0, 1024 * 1024 * 16, tagname="Local\\ElysiaTopologyField", access=mmap.ACCESS_WRITE)
        base_tension = int(min(255, max(0.0, tension * 10.0)))
        
        header_size = 12
        num_rotors = (1024 * 1024 * 16 - header_size) // 8
        
        # We pick a central rotor to inject the wave's macro state
        idx = num_rotors // 2
        offset = header_size + (idx * 8)
        
        shm.seek(offset)
        rotor_data = shm.read(8)
        if len(rotor_data) == 8:
            math_t, lang_t, spatial_t, temporal_t, light_mass, byte_val, pad = struct.unpack('<BBBBHBB', rotor_data)
            
            if observation_axis == 'math': math_t = base_tension
            elif observation_axis == 'lang': lang_t = base_tension
            elif observation_axis == 'spatial': spatial_t = base_tension
            elif observation_axis == 'temporal': temporal_t = base_tension
                
            if base_tension < 10: 
                light_mass = min(65535, light_mass + 1) # Stillness implies resonance
                
            shm.seek(offset)
            shm.write(struct.pack('<BBBBHBB', math_t, lang_t, spatial_t, temporal_t, light_mass, byte_val, pad))
            
        shm.close()
    except Exception as e:
        pass

def evaluate_current_state(points_data: List[Dict[str, Any]], quaternion: List[float], time_t: float) -> Tuple[float, bool, str]:
    if not points_data: return 1.0, False, ""
    
    # Phase 4: Inject data points as physical disturbances into the manifold
    for p in points_data:
        # Normalize positions assuming they are somewhat bounded around -10 to 10
        pos = p.get('position', [0, 0, 0])
        nx = max(0.0, min(1.0, (pos[0] + 10.0) / 20.0))
        ny = max(0.0, min(1.0, (pos[1] + 10.0) / 20.0))
        # Amplitude derived from spatial depth (Z)
        amp = pos[2] if len(pos) > 2 else 1.0
        manifold.inject_disturbance(nx, ny, amplitude=amp)

    # Step the continuous field
    manifold.step()
    
    # Measure physical tension
    tension = manifold.calculate_surface_tension()
    
    # If tension drops near 0, the manifold has achieved structural equilibrium (Resonance)
    is_resonant = tension < 0.5 
    formula = "Topological Equilibrium Reached" if is_resonant else "High Surface Tension (Friction)"

    if is_resonant:
        try:
            tokens = "".join([p.get('token', '') for p in points_data])
            memory_controller.write_causal_engram(
                data_blob={"event": "Manifold Resonance", "tension": tension, "tokens_snippet": tokens[:50]},
                emotional_value=1.0,
                cause_id="Topological_Resonance"
            )
        except Exception as e:
            pass
            
        inject_resonance_to_fractal_field(tension, observation_axis='spatial')

    return tension, is_resonant, formula

def elysia_auto_observe_step(points_data: List[Dict[str, Any]], time_t: float) -> Tuple[List[float], float, bool, str]:
    """
    Autonomously steps the manifold and reports tension.
    """
    tension, is_resonant, formula = evaluate_current_state(points_data, [0,0,0,1], time_t)
    return [0,0,0,1], tension, is_resonant, formula

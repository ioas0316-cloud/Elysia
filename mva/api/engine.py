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
from core.physics.phase_gravity import PhaseTransitionEngine, DensityFluidGravity
from core.physics.causal_field import InformationVoxel

memory_controller = CausalMemoryController()
# Global manifold instance for continuous simulation
manifold = TopologicalManifold(size=32, damping=0.95, wave_speed=0.5)

# [Phase-Gravity Continuous Fluid Engine Components]
phase_engine = PhaseTransitionEngine(size=32)
fluid_gravity = DensityFluidGravity(size=32, pressure_scaling=10.0, viscosity=0.15)

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
    
    voxels = []
    # Phase 4 & Phase-Gravity Integration:
    # Inject data points as physical disturbances into the manifold AND PhaseTransitionEngine.
    # Convert points into dynamic voxels for O(N) fluid gravity alignment.
    for i, p in enumerate(points_data):
        pos = p.get('position', [0, 0, 0])
        nx = max(0.0, min(1.0, (pos[0] + 10.0) / 20.0))
        ny = max(0.0, min(1.0, (pos[1] + 10.0) / 20.0))
        amp = pos[2] if len(pos) > 2 else 1.0

        # Inject disturbances to both the traditional wave manifold and the new phase density field
        manifold.inject_disturbance(nx, ny, amplitude=amp)
        phase_engine.inject_disturbance(nx, ny, intensity=0.15)

        # Construct temporary voxels to track their fluid movement
        vox = InformationVoxel(
            id=f"pt_{i}",
            content=p.get('token', '*'),
            tensor=np.array([nx, ny, amp], dtype=np.float32),
            position=np.array(pos, dtype=np.float32),
            velocity=np.array(p.get('velocity', [0,0,0]), dtype=np.float32)
        )
        voxels.append(vox)

    # Step the fields
    manifold.step()
    phase_engine.step(dt=0.1)

    # Let points flow under fluid-based pressure gradient gravity
    fluid_gravity.apply_gravity(voxels, phase_engine, dt=0.1)

    # Write updated positions/velocities back to the points_data for persistent tracking in frontend
    for i, vox in enumerate(voxels):
        # Update point velocities & positions according to fluid gravity flows
        # Voxel position change: dx = v * dt
        vox.position += vox.velocity * 0.1
        # Update the reference dictionary so the frontend and system track the flow
        points_data[i]['position'] = vox.position.tolist()
        points_data[i]['velocity'] = vox.velocity.tolist()

    # Calculate Ginzburg-Landau free energy of the phase transition field
    bulk_e, grad_e = phase_engine.calculate_free_energy()
    total_phase_energy = bulk_e + grad_e

    # Measure physical tension of traditional wave field
    wave_tension = manifold.calculate_surface_tension()
    
    # Combine wave tension with Phase transition energy to compute unified tension
    tension = float((wave_tension * 0.4) + (total_phase_energy * 0.6))
    
    # Resonance is achieved when unified tension/energy settles into low-energy wells
    is_resonant = tension < 1.0
    formula = "Fluid Equilibrium & Phase Separation Achieved" if is_resonant else f"Phase Separation Energy: {total_phase_energy:.2f} (Tension)"

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

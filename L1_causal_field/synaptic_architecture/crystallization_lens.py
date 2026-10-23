import numpy as np
from typing import Dict, List, Any
from core.physics.voxel_protocol import FieldLens, FieldDelta
from core.physics.scale_projection import project_3d_to_2d, compute_projection_jacobian
from synaptic_architecture.field import CrystallizationField

class CrystallizationLens:
    def __init__(self, resolution: int = 256):
        self._lens_id = "lens_crystallization_2d"
        self.field = CrystallizationField(resolution=resolution)
        self.resolution = resolution

    @property
    def lens_id(self) -> str:
        return self._lens_id

    def observe(self, voxels: Dict[str, Any], beams: List[Any]) -> Dict[str, Any]:
        # Reset activation map
        self.field.activation.fill(0.0)
        
        observations = {}
        for vid, voxel in voxels.items():
            # 1. Project 3D position to 2D
            pos_2d = project_3d_to_2d(voxel.position, self.resolution, projection_plane="xy")
            
            row, col = int(np.clip(pos_2d[0], 0, self.resolution - 1)), int(np.clip(pos_2d[1], 0, self.resolution - 1))
            
            # 2. Map voxel energy (potential + kinetic) to activation
            energy = voxel.potential + 0.5 * voxel.mass * float(np.sum(voxel.velocity**2))
            self.field.activation[row, col] += float(energy)
            
            # 3. Read local 2D properties at this projected coordinate
            obs = {
                "pos_2d": pos_2d,
                "local_conductance": float(self.field.conductance[row, col]),
                "local_temperature": float(self.field.local_temperature[row, col]),
                "local_yeobaek": float(self.field.coordination_margin[row, col]),
            }
            observations[vid] = obs
            
        return observations

    def modulate(self, voxels: Dict[str, Any], beams: List[Any], dt: float) -> List[FieldDelta]:
        # 1. Observe the current state
        obs = self.observe(voxels, beams)
        
        # 2. Advance the 2D field dynamics (diffusion, yeobaek calculation)
        # Note: CrystallizationField doesn't have a single step(dt) currently, so we'll simulate the update
        # For now, we calculate gravitational pull from attractors and update voxel properties
        
        # We assume external cognitive entropy/tension is derived from the field state or passed in
        # Defaulting to moderate values for now
        self.field.update_attractor_masses(cognitive_entropy=0.5, tension_protocol=0.5, catastrophe_magnitude=0.1)
        
        deltas = []
        for vid, voxel in voxels.items():
            v_obs = obs.get(vid)
            if not v_obs: continue
            
            pos_2d = v_obs["pos_2d"]
            
            # 3. Calculate 2D Cognitive Gravity (Volitional Acceleration)
            acc_2d, _ = self.field.get_volitional_acceleration(pos_2d, 0.5, 0.5, 0.1)
            
            # 4. Lift 2D acceleration back to 3D using Jacobian transpose
            J = compute_projection_jacobian(voxel.position, self.resolution, projection_plane="xy")
            # a_3D = J^T * a_2D
            acc_3d = np.dot(J.T, acc_2d)
            delta_vel = acc_3d * dt
            
            # 5. Modulate voxel scalar properties based on the 2D field's landscape
            row, col = int(np.clip(pos_2d[0], 0, self.resolution - 1)), int(np.clip(pos_2d[1], 0, self.resolution - 1))
            new_conductance = float(self.field.conductance[row, col])
            new_yeobaek = float(self.field.coordination_margin[row, col])
            
            d_conductance = (new_conductance - voxel.conductance) * 0.1 * dt
            d_yeobaek = (new_yeobaek - voxel.coordination_margin) * 0.1 * dt
            
            deltas.append(FieldDelta(
                target_voxel_ids=[vid],
                delta_velocity=delta_vel,
                delta_conductance=d_conductance,
                delta_coordination_margin=d_yeobaek,
                source_lens=self.lens_id
            ))
            
        return deltas

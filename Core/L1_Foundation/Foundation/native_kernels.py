"""
Native Kernels (Sovereign JIT Transmutation)
=============================================
Core.L1_Foundation.Foundation.native_kernels

"Bypassing the Ghost in the Shell."
"                            ."

This module contains high-performance JIT-compiled kernels
for physics and rendering pre-calculations.
"""

import numpy as np
import numba

@numba.njit(fastmath=True, cache=True)
def calculate_force_kernel(positions, field_positions, field_strengths, field_decays):
    """
    [JIT] Native Force Calculation Kernel.
    Vectorized and compiled to machine code for O(N*M) speedup.
    """
    n = positions.shape[0]
    m = field_positions.shape[0]
    forces = np.zeros_like(positions)
    
    for i in range(n):
        pos = positions[i]
        total_force = np.zeros_like(pos)
        
        # 1. External Fields Resonance
        for j in range(m):
            f_pos = field_positions[j]
            strength = field_strengths[j]
            decay = field_decays[j]
            
            diff = f_pos - pos
            dist = np.sqrt(np.sum(diff**2))
            
            if dist > 0.01:
                # Potential V = -S / (d + 1) * exp(-decay * d)
                # Force F = -grad(V)
                # Simplified force magnitude for efficiency
                force_mag = strength / (dist + 1.0)**2 * np.exp(-decay * dist)
                total_force += (diff / dist) * force_mag
        
        forces[i] = total_force
        
    return forces

@numba.njit(fastmath=True)
def evolve_positions_kernel(positions, velocities, forces, mass, dt, damping=0.05):
    """
    [JIT] Symplectic Integration Kernel.
    Updates positions and velocities in the native layer.
    """
    n = positions.shape[0]
    new_positions = np.zeros_like(positions)
    new_velocities = np.zeros_like(velocities)
    
    for i in range(n):
        m = mass[i]
        if m > 0:
            accel = forces[i] / m
            # Apply Damping (Entropic Resistance)
            accel -= (damping * velocities[i]) / m
            
            new_v = velocities[i] + accel * dt
            new_p = positions[i] + new_v * dt
            
            new_velocities[i] = new_v
            new_positions[i] = new_p
        else:
            # Massless (Photon-like)
            new_positions[i] = positions[i] + velocities[i] * dt
            new_velocities[i] = velocities[i]
            
    return new_positions, new_velocities

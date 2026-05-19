import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple

@jit
def compute_inter_monad_proximity(vectors: jnp.ndarray) -> jnp.ndarray:
    """
    [LIGHTNING PATH 2.0]
    Calculates the N x N proximity matrix using a JIT-compiled RBF kernel.
    vectors: [N, 12] Tensor
    returns: [N, N] Proximity Matrix
    """
    # Squared Euclidean Distance
    dist_sq = jnp.sum((vectors[:, None, :] - vectors[None, :, :])**2, axis=-1)
    # RBF Kernel (Synaptic focus)
    return jnp.exp(-dist_matrix_scale(dist_sq))

@jit
def dist_matrix_scale(dist_sq: jnp.ndarray) -> jnp.ndarray:
    return dist_sq * 2.0

@jit
def calculate_synaptic_flow(proximity: jnp.ndarray, masses: jnp.ndarray, conductivity_base: float) -> jnp.ndarray:
    """
    Computes the collective energy buzz jumping between monads.
    [N, N] @ [N, 1] -> [N, 1]
    """
    return jnp.matmul(proximity, masses) * conductivity_base

@jit
def aggregate_unified_field(vectors: jnp.ndarray, masses: jnp.ndarray) -> jnp.ndarray:
    """
    Vectorized field aggregation across all 12 dimensions.
    """
    return jnp.sum(vectors * masses, axis=0)

@jit
def update_monad_physics(masses: jnp.ndarray, similarities: jnp.ndarray, synaptic_flow: jnp.ndarray, decay_rate: float) -> jnp.ndarray:
    """
    [LIGHTNING PATH 2.0]
    Fuses mass growth, circuit flow, and natural decay into a single kernel operation.
    Reduces host-device transfer overhead.
    """
    resonant_mask = (similarities > 0.9).astype(jnp.float32)
    gravity_mask = (similarities > 0.8).astype(jnp.float32)
    
    # Growth factors
    growth = (resonant_mask * 0.01) + (gravity_mask * 0.005)
    # New Mass = (Current + Growth + Flow) * Decay
    new_masses = (masses + growth + (synaptic_flow * 0.1)) * (1.0 - decay_rate)
    return new_masses

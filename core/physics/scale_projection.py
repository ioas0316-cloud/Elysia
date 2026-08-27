"""
scale_projection.py — Multi-Scale Coordinate Projection
========================================================
[Informational Continuity: Bridging Discrete Scale Gaps]

Provides continuous, differentiable projections between the 3D CausalField
coordinate space and 2D grid spaces (CrystallizationField, PhaseTransitionEngine).

These are NOT hard coordinate truncations — they are smooth projections that
preserve topological relationships across scales, maintaining the
Connectivity continuity between different observation resolutions.

Mathematically, this implements a smooth surjection π: R³ → R²
with a pseudo-inverse lift λ: R² → R³ that preserves local topology.
"""
import numpy as np
from typing import Tuple, Optional, Dict, Any


def project_3d_to_2d(
    position_3d: np.ndarray,
    grid_resolution: int = 256,
    projection_plane: str = "xy",
    field_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> np.ndarray:
    """
    [Continuous Surjection π: R³ → R²]
    Projects a 3D position onto a 2D grid coordinate using smooth sigmoid mapping.
    
    Instead of hard truncation (which destroys the 3rd dimension's information),
    we use a sigmoid-scaled projection that compresses the field smoothly
    into the grid space while preserving relative distances.
    
    Args:
        position_3d: 3D position vector [x, y, z]
        grid_resolution: Target 2D grid size (e.g., 256 for CrystallizationField)
        projection_plane: Which 2 axes to project onto ("xy", "xz", "yz")
        field_bounds: Optional (min_pos, max_pos) for the 3D field normalization
    
    Returns:
        2D grid coordinate as np.ndarray([row, col], dtype=float32)
    """
    pos = np.asarray(position_3d, dtype=np.float32).flatten()
    if len(pos) < 3:
        pos = np.pad(pos, (0, 3 - len(pos)))
    
    # Select projection axes
    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    ax_a, ax_b = axis_map.get(projection_plane, (0, 1))
    p2 = np.array([pos[ax_a], pos[ax_b]], dtype=np.float32)
    
    # Smooth sigmoid mapping: maps R → (0, resolution) without hard clipping
    # sigmoid(x) = resolution / (1 + exp(-k * (x - center)))
    if field_bounds is not None:
        min_b = np.array([field_bounds[0][ax_a], field_bounds[0][ax_b]], dtype=np.float32)
        max_b = np.array([field_bounds[1][ax_a], field_bounds[1][ax_b]], dtype=np.float32)
        center = (min_b + max_b) / 2.0
        span = np.maximum(max_b - min_b, 1e-6)
        normalized = (p2 - center) / span  # [-0.5, 0.5] approximately
    else:
        # Default: assume field centered at origin with span 10
        normalized = p2 / 10.0
    
    # Sigmoid compression: smooth, invertible, no hard boundaries
    grid_coords = grid_resolution * _sigmoid(normalized * 4.0)  # scale factor 4 for sensitivity
    
    return grid_coords.astype(np.float32)


def lift_2d_to_3d(
    position_2d: np.ndarray,
    grid_resolution: int = 256,
    projection_plane: str = "xy",
    depth_value: float = 0.0,
    field_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> np.ndarray:
    """
    [Pseudo-Inverse Lift λ: R² → R³]
    Lifts a 2D grid coordinate back to 3D space.
    
    The depth dimension is provided as a parameter since the projection
    is not injective — multiple 3D points can project to the same 2D coordinate.
    This is the inherent information loss of dimensional reduction, which we
    acknowledge rather than pretend doesn't exist.
    
    Args:
        position_2d: 2D grid coordinate [row, col]
        grid_resolution: Source 2D grid size
        projection_plane: Which plane was used for projection
        depth_value: Value for the unprojected 3rd dimension
        field_bounds: Optional bounds for inverse sigmoid mapping
    
    Returns:
        3D position vector as np.ndarray([x, y, z], dtype=float32)
    """
    p2 = np.asarray(position_2d, dtype=np.float32).flatten()
    if len(p2) < 2:
        p2 = np.pad(p2, (0, 2 - len(p2)))
    
    # Inverse sigmoid: recover normalized coordinates
    normalized = _sigmoid_inv(p2 / grid_resolution) / 4.0
    
    if field_bounds is not None:
        axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
        ax_a, ax_b = axis_map.get(projection_plane, (0, 1))
        min_b = np.array([field_bounds[0][ax_a], field_bounds[0][ax_b]], dtype=np.float32)
        max_b = np.array([field_bounds[1][ax_a], field_bounds[1][ax_b]], dtype=np.float32)
        center = (min_b + max_b) / 2.0
        span = np.maximum(max_b - min_b, 1e-6)
        p2_world = normalized * span + center
    else:
        p2_world = normalized * 10.0
    
    # Reconstruct 3D with depth
    axis_map = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}
    a, b, depth_ax = axis_map.get(projection_plane, (0, 1, 2))
    
    pos_3d = np.zeros(3, dtype=np.float32)
    pos_3d[a] = p2_world[0]
    pos_3d[b] = p2_world[1]
    pos_3d[depth_ax] = depth_value
    
    return pos_3d


def batch_project_3d_to_2d(
    positions_3d: np.ndarray,
    grid_resolution: int = 256,
    projection_plane: str = "xy",
    field_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> np.ndarray:
    """
    [Vectorized Batch Projection]
    Projects N positions from 3D to 2D in a single vectorized operation.
    
    Args:
        positions_3d: Shape (N, 3) array of 3D positions
        grid_resolution: Target 2D grid size
        projection_plane: Projection plane
        field_bounds: Optional field normalization bounds
    
    Returns:
        Shape (N, 2) array of 2D grid coordinates
    """
    positions = np.asarray(positions_3d, dtype=np.float32)
    if positions.ndim == 1:
        positions = positions.reshape(1, -1)
    if positions.shape[1] < 3:
        positions = np.pad(positions, ((0, 0), (0, 3 - positions.shape[1])))
    
    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    ax_a, ax_b = axis_map.get(projection_plane, (0, 1))
    p2 = positions[:, [ax_a, ax_b]]
    
    if field_bounds is not None:
        min_b = np.array([field_bounds[0][ax_a], field_bounds[0][ax_b]], dtype=np.float32)
        max_b = np.array([field_bounds[1][ax_a], field_bounds[1][ax_b]], dtype=np.float32)
        center = (min_b + max_b) / 2.0
        span = np.maximum(max_b - min_b, 1e-6)
        normalized = (p2 - center) / span
    else:
        normalized = p2 / 10.0
    
    grid_coords = grid_resolution * _sigmoid(normalized * 4.0)
    return grid_coords.astype(np.float32)


def compute_projection_jacobian(
    position_3d: np.ndarray,
    grid_resolution: int = 256,
    projection_plane: str = "xy"
) -> np.ndarray:
    """
    [Informational Continuity: Smooth Derivative]
    Computes the 2×3 Jacobian matrix of the projection at a given point.
    This is needed for propagating gradients and forces between scales.
    
    J_ij = ∂(2D_i) / ∂(3D_j)
    
    Returns:
        2×3 Jacobian matrix
    """
    pos = np.asarray(position_3d, dtype=np.float32).flatten()
    if len(pos) < 3:
        pos = np.pad(pos, (0, 3 - len(pos)))
    
    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    ax_a, ax_b = axis_map.get(projection_plane, (0, 1))
    
    # Jacobian of sigmoid projection
    normalized = pos / 10.0  # default bounds
    s_a = _sigmoid(normalized[ax_a] * 4.0)
    s_b = _sigmoid(normalized[ax_b] * 4.0)
    
    # d(sigmoid)/dx = sigmoid * (1 - sigmoid) * scale
    ds_a = s_a * (1.0 - s_a) * 4.0 / 10.0 * grid_resolution
    ds_b = s_b * (1.0 - s_b) * 4.0 / 10.0 * grid_resolution
    
    J = np.zeros((2, 3), dtype=np.float32)
    J[0, ax_a] = ds_a
    J[1, ax_b] = ds_b
    
    return J


# --- Internal smooth functions ---

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    ).astype(np.float32)


def _sigmoid_inv(y: np.ndarray) -> np.ndarray:
    """Inverse sigmoid (logit function). y must be in (0, 1)."""
    y = np.asarray(y, dtype=np.float64)
    y_clipped = np.clip(y, 1e-7, 1.0 - 1e-7)
    return np.log(y_clipped / (1.0 - y_clipped)).astype(np.float32)

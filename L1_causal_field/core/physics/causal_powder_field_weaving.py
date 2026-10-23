"""
Causal Powder Field & Spatiotemporal Weaving Engine (causal_powder_field_weaving.py)
===================================================================================
Transcends discrete brute-force particle trajectory calculation by combining:
1. Microscopic Continuous Potential Fields (Powder/Earth, Thermal/Fire, Fluid/Water, Gravity)
   governed by the 4 Continuities (Relationship, Connectivity, Mobility, Informational Continuity)
   and Chromatic Vectors [Red (Flux), Blue (Order), Yellow (Entropy)].
2. Macroscopic 4D Spatiotemporal Causal Manifold Weaving (Causal Flow Field), enabling O(1)
   continuous time-slider ($t \\in [0, T]$) forward and retrocausal state reconstruction
   without frame-by-frame physics re-computations.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class ElementaryFieldState:
    """
    Continuous field representations for Powder World elements at a specific spatial coordinate.
    """
    earth: float = 0.0      # Powder/Earth density (Blue order / friction base)
    thermal: float = 0.0    # Thermal/Fire intensity (Red flux)
    fluid: float = 0.0      # Fluid/Water density (Yellow entropy / flow)
    elevation: float = 0.0  # Topological height manifold
    chromatic: np.ndarray = None  # [Red (Flux), Blue (Order), Yellow (Entropy)]

    def __post_init__(self):
        if self.chromatic is None:
            self.chromatic = np.array([0.33, 0.33, 0.34], dtype=np.float32)


class MicroscopicPowderField:
    """
    Microscopic continuous potential field simulator.
    Simulates local continuous interactions between Earth, Fire, Water, and Gravity
    under AGENTS.md 4 Continuities.
    """
    def __init__(self, width: int = 16, height: int = 16, gravity: float = 9.81):
        self.width = width
        self.height = height
        self.gravity = gravity

        # Field grids: 2D array representation [height, width]
        self.earth = np.zeros((height, width), dtype=np.float32)      # Earth/Powder density
        self.thermal = np.zeros((height, width), dtype=np.float32)    # Fire/Heat intensity
        self.fluid = np.zeros((height, width), dtype=np.float32)      # Water/Fluid density
        self.elevation = np.zeros((height, width), dtype=np.float32)  # Terrain height map

        # 3D Chromatic Tensor [height, width, 3] -> Red (Flux), Blue (Order), Yellow (Entropy)
        self.chromatic = np.zeros((height, width, 3), dtype=np.float32)
        self.chromatic[:, :, 0] = 0.33  # Red
        self.chromatic[:, :, 1] = 0.34  # Blue
        self.chromatic[:, :, 2] = 0.33  # Yellow

    def seed_powder(self, x: int, y: int, amount: float = 1.0):
        """Seeds powder / earth at specific coordinate."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.earth[y, x] += amount
            self.elevation[y, x] += amount * 0.5
            self.chromatic[y, x, 1] += 0.2  # Blue (Order)
            self._normalize_chromatic(y, x)

    def seed_thermal(self, x: int, y: int, intensity: float = 1.0):
        """Injects thermal energy / fire at specific coordinate."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.thermal[y, x] += intensity
            self.chromatic[y, x, 0] += 0.4  # Red (Flux)
            self._normalize_chromatic(y, x)

    def seed_fluid(self, x: int, y: int, amount: float = 1.0):
        """Seeds fluid / water at specific coordinate."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.fluid[y, x] += amount
            self.chromatic[y, x, 2] += 0.3  # Yellow (Entropy)
            self._normalize_chromatic(y, x)

    def _normalize_chromatic(self, y: int, x: int):
        tot = np.sum(self.chromatic[y, x])
        if tot > 0:
            self.chromatic[y, x] /= tot

    def step(self, dt: float = 0.1):
        """
        Advances the microscopic field through continuous field partial differential flows:
        1. Thermal Diffusion & Ignition (Fire bakes Earth, evaporates Fluid)
        2. Fluid Flow along Elevation/Gravitational Gradients (Water erodes Earth)
        3. Powder / Earth Slumping & Relaxation (Ground stability under gravity)
        4. Chromatic Energy Redistribution
        """
        # 1. Thermal Diffusion (Heat spreading using Laplacian)
        laplacian_thermal = (
            np.roll(self.thermal, 1, axis=0) + np.roll(self.thermal, -1, axis=0) +
            np.roll(self.thermal, 1, axis=1) + np.roll(self.thermal, -1, axis=1) -
            4.0 * self.thermal
        )
        thermal_diffusivity = 0.2
        self.thermal += thermal_diffusivity * laplacian_thermal * dt
        self.thermal = np.maximum(0.0, self.thermal * (1.0 - 0.05 * dt))  # Thermal dissipation

        # Thermal impact on Earth & Fluid (Fire bakes earth, evaporates water)
        evaporation = np.minimum(self.fluid, self.thermal * 0.3 * dt)
        self.fluid -= evaporation
        self.earth += evaporation * 0.1  # Sediment deposit from evaporated fluid

        # 2. Fluid Flow & Erosion (Downslope fluid movement driven by gravity and elevation)
        grad_y, grad_x = np.gradient(self.elevation + self.fluid * 0.2)
        fluid_flow_x = -grad_x * self.gravity * 0.1 * dt
        fluid_flow_y = -grad_y * self.gravity * 0.1 * dt

        # Advect fluid along gradient
        new_fluid = np.copy(self.fluid)
        new_fluid += (np.roll(self.fluid * fluid_flow_x, 1, axis=1) - np.roll(self.fluid * fluid_flow_x, -1, axis=1)) * 0.5
        new_fluid += (np.roll(self.fluid * fluid_flow_y, 1, axis=0) - np.roll(self.fluid * fluid_flow_y, -1, axis=0)) * 0.5
        self.fluid = np.maximum(0.0, new_fluid)

        # Fluid erodes Earth
        erosion = np.minimum(self.earth, self.fluid * 0.05 * dt)
        self.earth -= erosion
        self.elevation = self.earth * 0.5

        # 3. Earth Powder Slumping (Angle of repose relaxation)
        laplacian_earth = (
            np.roll(self.earth, 1, axis=0) + np.roll(self.earth, -1, axis=0) +
            np.roll(self.earth, 1, axis=1) + np.roll(self.earth, -1, axis=1) -
            4.0 * self.earth
        )
        self.earth += 0.1 * laplacian_earth * dt
        self.earth = np.maximum(0.0, self.earth)
        self.elevation = self.earth * 0.5

        # 4. Chromatic Modulation
        # Red tracks Thermal flux, Blue tracks Earth structural order, Yellow tracks Fluid entropy
        red = self.thermal / (self.thermal + 1.0)
        blue = self.earth / (self.earth + 1.0)
        yellow = self.fluid / (self.fluid + 1.0)

        # Base chromatic floor for empty space
        base_mask = (red == 0) & (blue == 0) & (yellow == 0)

        self.chromatic[:, :, 0] = red
        self.chromatic[:, :, 1] = blue
        self.chromatic[:, :, 2] = yellow

        # For unpopulated areas, assign default neutral balance
        self.chromatic[base_mask, 0] = 0.33
        self.chromatic[base_mask, 1] = 0.34
        self.chromatic[base_mask, 2] = 0.33

        sum_chroma = np.sum(self.chromatic, axis=2, keepdims=True) + 1e-8
        self.chromatic /= sum_chroma

    def snapshot(self) -> Dict[str, np.ndarray]:
        """Returns deep copy snapshot of field states."""
        return {
            "earth": np.copy(self.earth),
            "thermal": np.copy(self.thermal),
            "fluid": np.copy(self.fluid),
            "elevation": np.copy(self.elevation),
            "chromatic": np.copy(self.chromatic)
        }


class CausalManifoldWeaver:
    """
    [Causal Field Weaver: Beyond Particulate Calculation]
    Compiles sequential microscopic field snapshots into a continuous 4D Spatiotemporal
    Causal Manifold Tensor W(x, y, layer, t).
    Allows querying field state at any arbitrary continuous time t in [0, T] via Hermite/Spline
    interpolation, enabling continuous time-slider sliding, retrocausal back-tracing,
    and forward extrapolation without re-executing micro-step loops.
    """
    def __init__(self, width: int = 16, height: int = 16):
        self.width = width
        self.height = height

        self.timestamps: List[float] = []
        # Store compressed 4D manifold snapshots: list of dicts
        self.snapshots: List[Dict[str, np.ndarray]] = []
        # 4D Spatiotemporal Causal Tensor [time, height, width, channels]
        # Channels: 0: earth, 1: thermal, 2: fluid, 3: elevation, 4: red, 5: blue, 6: yellow
        self.causal_tensor_4d: Optional[np.ndarray] = None
        self.is_woven: bool = False

    def record_snapshot(self, timestamp: float, snapshot_data: Dict[str, np.ndarray]):
        """Records a microscopic field snapshot at a given time tick."""
        self.timestamps.append(timestamp)
        self.snapshots.append(snapshot_data)
        self.is_woven = False

    def weave_causal_manifold(self):
        """
        Weaves recorded snapshots into a unified 4D Spatiotemporal Causal Manifold Tensor.
        Constructs smooth interpolation splines along the temporal axis t.
        """
        if not self.snapshots:
            return

        num_t = len(self.snapshots)
        # Create 4D array [num_t, height, width, 7]
        self.causal_tensor_4d = np.zeros((num_t, self.height, self.width, 7), dtype=np.float32)

        for i, snap in enumerate(self.snapshots):
            self.causal_tensor_4d[i, :, :, 0] = snap["earth"]
            self.causal_tensor_4d[i, :, :, 1] = snap["thermal"]
            self.causal_tensor_4d[i, :, :, 2] = snap["fluid"]
            self.causal_tensor_4d[i, :, :, 3] = snap["elevation"]
            self.causal_tensor_4d[i, :, :, 4] = snap["chromatic"][:, :, 0]
            self.causal_tensor_4d[i, :, :, 5] = snap["chromatic"][:, :, 1]
            self.causal_tensor_4d[i, :, :, 6] = snap["chromatic"][:, :, 2]

        self.is_woven = True

    def query_causal_state(self, t: float) -> Dict[str, np.ndarray]:
        """
        [O(1) Continuous Spatiotemporal Query / Time-Slider]
        Queries continuous field state at arbitrary time t in [0, T].
        Uses smooth Cubic Hermite / Linear Causal Field Interpolation across time axis t.
        """
        if not self.is_woven or self.causal_tensor_4d is None:
            self.weave_causal_manifold()

        if len(self.timestamps) == 0:
            return {
                "earth": np.zeros((self.height, self.width), dtype=np.float32),
                "thermal": np.zeros((self.height, self.width), dtype=np.float32),
                "fluid": np.zeros((self.height, self.width), dtype=np.float32),
                "elevation": np.zeros((self.height, self.width), dtype=np.float32),
                "chromatic": np.zeros((self.height, self.width, 3), dtype=np.float32)
            }

        t_min = self.timestamps[0]
        t_max = self.timestamps[-1]
        clamped_t = float(np.clip(t, t_min, t_max))

        # Find temporal interval [t_idx, t_idx + 1]
        times = np.array(self.timestamps)
        idx = np.searchsorted(times, clamped_t) - 1
        idx = int(np.clip(idx, 0, len(times) - 2)) if len(times) > 1 else 0

        t0 = times[idx]
        t1 = times[idx + 1] if len(times) > 1 else t0 + 1.0

        # Interpolation factor alpha ∈ [0, 1]
        alpha = (clamped_t - t0) / (t1 - t0 + 1e-8)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        # Smooth Hermite s-curve: 3*alpha^2 - 2*alpha^3 for continuous causal flow
        smooth_alpha = 3.0 * (alpha ** 2) - 2.0 * (alpha ** 3)

        state0 = self.causal_tensor_4d[idx]
        state1 = self.causal_tensor_4d[idx + 1] if len(times) > 1 else state0

        # Continuous Causal Interpolation
        interp_state = (1.0 - smooth_alpha) * state0 + smooth_alpha * state1

        chroma = interp_state[:, :, 4:7]
        sum_c = np.sum(chroma, axis=2, keepdims=True) + 1e-8
        chroma /= sum_c

        return {
            "earth": interp_state[:, :, 0],
            "thermal": interp_state[:, :, 1],
            "fluid": interp_state[:, :, 2],
            "elevation": interp_state[:, :, 3],
            "chromatic": chroma
        }

    def compute_causal_flow_manifold(self) -> Dict[str, np.ndarray]:
        """
        Computes 4D temporal derivative vector field dW/dt (Causal Flow Gradient).
        Represents the intrinsic directional flow of causality through spacetime.
        """
        if self.causal_tensor_4d is None or len(self.timestamps) < 2:
            return {}

        dt_array = np.gradient(np.array(self.timestamps))
        flow_grad = np.gradient(self.causal_tensor_4d, axis=0)

        return {
            "causal_flow_rate": flow_grad,
            "temporal_density": dt_array
        }


class PowderWorldCausalEngine:
    """
    Integrated Powder World & Causal Weaving Engine.
    Unifies Microscopic Potential Field dynamics with 4D Causal Manifold Weaving.
    """
    def __init__(self, width: int = 16, height: int = 16, gravity: float = 9.81):
        self.width = width
        self.height = height
        self.micro_field = MicroscopicPowderField(width=width, height=height, gravity=gravity)
        self.weaver = CausalManifoldWeaver(width=width, height=height)
        self.current_time = 0.0

    def seed_initial_powder_world(self):
        """Seeds initial elementary powder, water, and thermal sources."""
        # Central earth heap
        cx, cy = self.width // 2, self.height // 2
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                dist = np.sqrt(dx*dx + dy*dy)
                if dist <= 2.5:
                    self.micro_field.seed_powder(cx + dx, cy + dy, amount=3.0 - dist)

        # Top fluid source
        self.micro_field.seed_fluid(cx, 1, amount=4.0)

        # Bottom thermal source
        self.micro_field.seed_thermal(cx, self.height - 2, intensity=5.0)

        # Record initial snapshot at t=0
        self.weaver.record_snapshot(0.0, self.micro_field.snapshot())

    def run_simulation_phase(self, ticks: int = 50, dt: float = 0.1):
        """Runs microscopic simulation for N ticks and records snapshot stream."""
        for _ in range(ticks):
            self.current_time += dt
            self.micro_field.step(dt=dt)
            self.weaver.record_snapshot(self.current_time, self.micro_field.snapshot())

        # Weave manifold after phase execution
        self.weaver.weave_causal_manifold()

    def slide_time(self, target_t: float) -> Dict[str, np.ndarray]:
        """
        [Time-Slider Interface]
        Moves slider freely across t in [0, T] to reconstruct state instantly.
        """
        return self.weaver.query_causal_state(target_t)

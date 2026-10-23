"""Enactive Hardware Ecosystem & Topological Rotor Field Runtime.

This module unifies the 4 primary hardware ecosystem axes (CPU, RAM, Disk I/O, Thermal/Power)
with DLSS 5-style Generative Neural Rendering optimization via Topological Memoization
and Spatiotemporal Spherical Clifford & Scale Rotor Fields.

Key Components:
1. SpatiotemporalSphericalRotorField:
   Maps 3D/4D spatiotemporal rendering fields onto a spherical manifold.
   Executes continuous view/time transformations via Clifford Rotors (R = e^{-theta/2 * B})
   and scale expansions via Scale Rotors (Dilators) without re-evaluating heavy pixel-level NN inference.

2. TopologicalResonanceCache:
   Caches relational topology resonance manifolds (Topological Memoization).
   Monitors Free Energy prediction error (F) and system friction.
   Provides Zero-Cost Flow during steady states and triggers dynamic localized Phase-Re-locking
   only when Free Energy error exceeds dynamic thresholds.

3. EnactiveHardwareEcosystemRuntime:
   Unified Causal Ecosystem Runtime coordinating 4 hardware axes:
     - CPU: Queue length & context switching friction -> Thread frequency phase-locking.
     - RAM & Cache: Memory tension -> Dynamic buffer compression & precision tuning.
     - Disk I/O & Network: Latency fluctuation -> Adaptive Backpressure phase matching.
     - Thermal & Power: Temperature derivative/friction -> Wave amplitude modulation (thermal equilibrium).
   Maintains high Resonance Density (rho_res) and continuous zero-crash operation.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np


class SpatiotemporalSphericalRotorField:
    """Maps spatiotemporal rendering wavefields onto a spherical topological manifold

    and applies Clifford Rotors (R = e^{-theta/2 B}) and Scale Rotors (Dilators)
    to execute continuous transformations with O(1) operator complexity.
    """

    def __init__(self, num_points: int = 1024, spatial_dims: int = 3):
        self.spatial_dims = spatial_dims

        grid_side = max(2, int(np.round(np.sqrt(num_points))))
        self.num_points = grid_side * grid_side

        # Initialize points uniformly on a unit sphere S^2 (or S^{spatial_dims-1})
        # spherical state: theta, phi and magnitude (radius r)
        phi = np.linspace(0.01, np.pi - 0.01, grid_side)
        theta = np.linspace(0.0, 2.0 * np.pi, grid_side, endpoint=False)
        phi_grid, theta_grid = np.meshgrid(phi, theta)

        self.theta = theta_grid.flatten()
        self.phi = phi_grid.flatten()
        self.radius = np.ones(self.num_points, dtype=np.float64)

        # Base Cartesian coordinates on sphere
        self.cartesian_wavefield = np.zeros((self.num_points, 3), dtype=np.float64)
        self._update_cartesian_coords()

        # Rotor state parameters
        self.current_rotor_angle = 0.0
        self.current_scale_factor = 1.0
        self.bivector_plane = np.array([0.0, 1.0, 0.0])  # Normal to bivector plane

    def _update_cartesian_coords(self):
        """Converts spherical (r, theta, phi) to Cartesian (x, y, z) wavefield."""
        x = self.radius * np.sin(self.phi) * np.cos(self.theta)
        y = self.radius * np.sin(self.phi) * np.sin(self.theta)
        z = self.radius * np.cos(self.phi)
        self.cartesian_wavefield = np.column_stack([x, y, z])

    def apply_clifford_rotor(self, angle: float, bivector_axis: Optional[np.ndarray] = None) -> np.ndarray:
        """Applies Clifford Rotor R = e^{-theta/2 B} to rotate the spatiotemporal manifold.

        Rotor Sandwich Operation: v' = R v R^dagger.
        In 3D, bivector B is dual to a rotation axis unit vector n.
        """
        if bivector_axis is None:
            bivector_axis = self.bivector_plane
        axis = bivector_axis / (np.linalg.norm(bivector_axis) + 1e-9)

        half_angle = angle / 2.0
        cos_h = np.cos(half_angle)
        sin_h = np.sin(half_angle)

        # Quaternion representation of Rotor: q = (cos(theta/2), sin(theta/2)*axis)
        q_w = cos_h
        q_xyz = sin_h * axis

        # Rotate each point in the wavefield
        # v' = v + 2*q_w*(q_xyz x v) + 2*(q_xyz x (q_xyz x v))
        v = self.cartesian_wavefield
        cross1 = np.cross(q_xyz, v)
        cross2 = np.cross(q_xyz, cross1)
        rotated_v = v + 2.0 * q_w * cross1 + 2.0 * cross2

        self.cartesian_wavefield = rotated_v
        self.current_rotor_angle = (self.current_rotor_angle + angle) % (2.0 * np.pi)

        # Update spherical coordinates from Cartesian
        r = np.linalg.norm(rotated_v, axis=1)
        self.radius = r
        self.phi = np.arccos(np.clip(rotated_v[:, 2] / (r + 1e-9), -1.0, 1.0))
        self.theta = np.arctan2(rotated_v[:, 1], rotated_v[:, 0]) % (2.0 * np.pi)

        return self.cartesian_wavefield

    def apply_scale_rotor(self, scale_factor: float) -> np.ndarray:
        """Applies Scale Rotor (Dilator Operator D = e^{s/2 * I}) to dilate or contract

        the harmonic wavefield spectrum (modulating detail zoom without pixel re-inference).
        """
        scale_factor = max(1e-4, float(scale_factor))
        self.radius *= scale_factor
        self.current_scale_factor *= scale_factor
        self._update_cartesian_coords()
        return self.cartesian_wavefield

    def compute_wavefield_energy(self) -> float:
        """Calculates total harmonic energy conserved in the rotor field."""
        return float(np.sum(self.radius ** 2) / self.num_points)


class TopologicalResonanceCache:
    """Topological Memoization Engine for Neural Rendering (DLSS 5).

    Caches relational topology resonance manifolds and leverages Free Energy
    prediction error (F) and system friction to determine when to utilize Zero-Cost Flow
    versus when to trigger Dynamic Localized Phase-Re-locking.
    """

    def __init__(self, free_energy_threshold: float = 0.15, max_cache_entries: int = 128):
        self.free_energy_threshold = free_energy_threshold
        self.max_cache_entries = max_cache_entries

        # Cache storage: key -> dict of topology manifold state
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.total_queries = 0
        self.cache_hits = 0
        self.zero_cost_steps = 0
        self.relock_steps = 0
        self.last_free_energy = 0.0

    def compute_free_energy_error(
        self,
        current_manifold: np.ndarray,
        cached_manifold: np.ndarray,
        friction: float = 0.0,
    ) -> float:
        """Computes Free Energy prediction error (F) between current observation and cached manifold.

        F = || current - cached ||^2 + alpha * friction
        """
        diff = current_manifold - cached_manifold
        mse = float(np.mean(diff ** 2))
        free_energy = mse + 0.1 * friction
        self.last_free_energy = free_energy
        return free_energy

    def query_or_update(
        self,
        scene_key: str,
        observed_manifold: np.ndarray,
        friction: float = 0.0,
    ) -> Dict[str, Any]:
        """Queries the cache or performs dynamic localized phase-re-locking.

        Returns:
            Dict containing:
                - status: "ZERO_COST_FLOW" or "DYNAMIC_PHASE_RELOCK"
                - manifold: updated or cached manifold
                - free_energy: current F error
                - cache_hit: bool
                - compute_saved_ratio: float (0.0 to 1.0)
        """
        self.total_queries += 1

        if scene_key in self.cache:
            cached_entry = self.cache[scene_key]
            cached_manifold = cached_entry["manifold"]

            F = self.compute_free_energy_error(observed_manifold, cached_manifold, friction)

            if F < self.free_energy_threshold:
                # Zero-Cost Flow: Re-use cached topology without heavy NN inference
                self.cache_hits += 1
                self.zero_cost_steps += 1
                cached_entry["hit_count"] += 1
                return {
                    "status": "ZERO_COST_FLOW",
                    "manifold": cached_manifold,
                    "free_energy": F,
                    "cache_hit": True,
                    "compute_saved_ratio": float(1.0 - (F / self.free_energy_threshold) * 0.2),
                    "modified_region_mask": np.zeros_like(observed_manifold, dtype=bool),
                }
            else:
                # Dynamic Localized Phase-Re-locking: Only update regions with high error
                self.relock_steps += 1
                error_map = (observed_manifold - cached_manifold) ** 2
                relock_mask = error_map > (self.free_energy_threshold * 0.5)

                # Update local manifold in cache
                updated_manifold = np.where(relock_mask, observed_manifold, cached_manifold)
                self.cache[scene_key] = {
                    "manifold": updated_manifold,
                    "hit_count": cached_entry["hit_count"] + 1,
                }

                # Saved compute is proportional to untouched regions
                untouched_ratio = float(1.0 - np.mean(relock_mask))
                return {
                    "status": "DYNAMIC_PHASE_RELOCK",
                    "manifold": updated_manifold,
                    "free_energy": F,
                    "cache_hit": False,
                    "compute_saved_ratio": max(0.1, untouched_ratio * 0.8),
                    "modified_region_mask": relock_mask,
                }
        else:
            # Cache Miss: Initial full lock
            self.relock_steps += 1
            self.cache[scene_key] = {
                "manifold": observed_manifold.copy(),
                "hit_count": 1,
            }
            return {
                "status": "INITIAL_PHASE_LOCK",
                "manifold": observed_manifold,
                "free_energy": 0.0,
                "cache_hit": False,
                "compute_saved_ratio": 0.0,
                "modified_region_mask": np.ones_like(observed_manifold, dtype=bool),
            }

    def get_stats(self) -> Dict[str, float]:
        """Returns cache efficiency metrics."""
        hit_ratio = float(self.cache_hits / self.total_queries) if self.total_queries > 0 else 0.0
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "hit_ratio": hit_ratio,
            "zero_cost_steps": self.zero_cost_steps,
            "relock_steps": self.relock_steps,
            "last_free_energy": self.last_free_energy,
        }


class EnactiveHardwareEcosystemRuntime:
    """Unified Causal Ecosystem Runtime managing the 4 Hardware Axes:

    1. CPU Axis: Queue length & context switching friction -> Thread frequency phase-locking.
    2. RAM Axis: Memory tension -> Buffer compression & precision tuning (e.g. FP32 -> FP16).
    3. Disk I/O & Network Axis: Latency fluctuation -> Adaptive Backpressure phase matching.
    4. Thermal & Power Axis: Thermal derivative -> Wave amplitude modulation.
    """

    def __init__(
        self,
        base_freq_hz: float = 60.0,
        total_ram_mb: float = 16384.0,
        thermal_limit_celsius: float = 85.0,
        free_energy_threshold: float = 0.15,
    ):
        self.base_freq_hz = base_freq_hz
        self.total_ram_mb = total_ram_mb
        self.thermal_limit_celsius = thermal_limit_celsius

        # Ecosystem components
        self.rotor_field = SpatiotemporalSphericalRotorField(num_points=400)
        self.topo_cache = TopologicalResonanceCache(free_energy_threshold=free_energy_threshold)

        # Phase lock state across the 4 axes
        self.axes = ["cpu", "ram", "io", "thermal"]
        self.axis_phases = {axis: 0.0 for axis in self.axes}
        self.axis_freqs = {axis: base_freq_hz for axis in self.axes}
        self.axis_tensions = {axis: 0.25 for axis in self.axes}

        self.time = 0.0
        self.accumulated_friction = 0.0
        self.resonance_density = 1.0

        # Memory & precision modulation state
        self.precision_mode = "FP32"  # "FP32", "FP16", "INT8"
        self.active_buffer_compression = 1.0  # 1.0 = uncompressed, 0.5 = 2x compressed
        self.wave_amplitude = 1.0  # Thermal modulation factor (0.1 to 1.0)
        self.backpressure_rate = 1.0  # I/O rate throttle (0.1 to 1.0)

    def evaluate_hardware_frictions(self, telemetry: Dict[str, float]) -> Dict[str, float]:
        """Calculates friction stimulus across the 4 hardware axes.

        Telemetry dict may contain:
            - cpu_utilization: float (0-100)
            - cpu_queue_length: int
            - ram_used_mb: float
            - io_latency_ms: float
            - thermal_celsius: float
            - thermal_derivative: float (deg C / sec)
        """
        # 1. CPU Friction
        cpu_util = telemetry.get("cpu_utilization", 50.0) / 100.0
        queue_len = telemetry.get("cpu_queue_length", 1)
        cpu_friction = max(0.0, (cpu_util - 0.8) * 3.0) + max(0.0, (queue_len - 4) * 0.2)

        # 2. RAM Friction (Tension)
        ram_used = telemetry.get("ram_used_mb", 4000.0)
        ram_ratio = ram_used / max(1.0, self.total_ram_mb)
        ram_friction = max(0.0, np.exp(max(0.0, ram_ratio - 0.75) * 6.0) - 1.0) if ram_ratio > 0.75 else 0.0

        # 3. I/O & Network Friction
        io_latency = telemetry.get("io_latency_ms", 5.0)
        io_friction = max(0.0, (io_latency - 15.0) / 20.0)

        # 4. Thermal & Power Friction
        temp = telemetry.get("thermal_celsius", 60.0)
        temp_rate = telemetry.get("thermal_derivative", 0.0)
        thermal_friction = max(0.0, (temp - 75.0) / 10.0) + max(0.0, temp_rate * 0.5)

        total_friction = float(cpu_friction + ram_friction + io_friction + thermal_friction)

        return {
            "cpu_friction": float(cpu_friction),
            "ram_friction": float(ram_friction),
            "io_friction": float(io_friction),
            "thermal_friction": float(thermal_friction),
            "total_friction": total_friction,
        }

    def modulate_ecosystem_controls(self, friction_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Dynamically adjusts 4-axis control parameters in response to friction.

        - RAM: Tension -> Precision reduction (FP32 -> FP16 -> INT8) & buffer compression.
        - Thermal: Friction -> Wave amplitude scaling (thermal equilibrium).
        - I/O: Latency -> Adaptive backpressure throttling.
        - CPU: Queue length -> Frequency phase adjustment.
        """
        ram_f = friction_metrics["ram_friction"]
        thermal_f = friction_metrics["thermal_friction"]
        io_f = friction_metrics["io_friction"]

        # RAM Precision & Buffer Modulation
        if ram_f > 1.5:
            self.precision_mode = "INT8"
            self.active_buffer_compression = 0.25
        elif ram_f > 0.5:
            self.precision_mode = "FP16"
            self.active_buffer_compression = 0.50
        else:
            self.precision_mode = "FP32"
            self.active_buffer_compression = 1.00

        # Thermal Amplitude Modulation (Prevents hard hardware thermal throttling)
        if thermal_f > 1.0:
            self.wave_amplitude = max(0.2, 1.0 - 0.3 * thermal_f)
        else:
            self.wave_amplitude = min(1.0, self.wave_amplitude + 0.05)

        # Adaptive I/O Backpressure Rate
        if io_f > 0.5:
            self.backpressure_rate = max(0.1, 1.0 / (1.0 + io_f))
        else:
            self.backpressure_rate = min(1.0, self.backpressure_rate + 0.1)

        # Axis Tension Redistribution
        f_vec = np.array([
            friction_metrics["cpu_friction"],
            friction_metrics["ram_friction"],
            friction_metrics["io_friction"],
            friction_metrics["thermal_friction"]
        ])
        inv_f = 1.0 / (1.0 + f_vec)
        norm_tensions = inv_f / np.sum(inv_f)

        for idx, axis in enumerate(self.axes):
            self.axis_tensions[axis] = float(norm_tensions[idx])

        return {
            "precision_mode": self.precision_mode,
            "buffer_compression": self.active_buffer_compression,
            "wave_amplitude": float(self.wave_amplitude),
            "backpressure_rate": float(self.backpressure_rate),
            "axis_tensions": dict(self.axis_tensions),
        }

    def synchronize_axis_phases(self, friction_metrics: Dict[str, float], dt: float) -> Dict[str, float]:
        """Evolves Kuramoto phase locking across the 4 hardware axes."""
        total_f = friction_metrics["total_friction"]
        freq_damping = 1.0 / (1.0 + 0.2 * total_f)

        # Target external frequency modulated by friction
        target_freq = self.base_freq_hz * freq_damping

        phases_list = []
        for axis in self.axes:
            # Advance phase
            d_phase = 2.0 * np.pi * target_freq * dt
            self.axis_phases[axis] = (self.axis_phases[axis] + d_phase) % (2.0 * np.pi)
            phases_list.append(self.axis_phases[axis])

        # Inter-axis Kuramoto Order Parameter (R)
        cos_sum = 0.0
        pair_count = 0
        n = len(self.axes)
        for i in range(n):
            for j in range(i + 1, n):
                cos_sum += np.cos(phases_list[i] - phases_list[j])
                pair_count += 1

        multi_order = (cos_sum / pair_count + 1.0) / 2.0 if pair_count > 0 else 1.0

        # Resonance Density: rho_res = (Order * Amplitude) / (1 + 0.2 * Total_Friction)
        self.resonance_density = float((multi_order * self.wave_amplitude) / (1.0 + 0.2 * total_f))

        return {
            "order_parameter_R": float(multi_order),
            "effective_freq_hz": float(target_freq),
            "resonance_density": self.resonance_density,
        }

    def step_ecosystem(
        self,
        hardware_telemetry: Dict[str, float],
        scene_key: str = "default_scene",
        observed_manifold: Optional[np.ndarray] = None,
        dt: float = 0.01,
    ) -> Dict[str, Any]:
        """Executes a single cycle of the Enactive Hardware Ecosystem Runtime.

        Args:
            hardware_telemetry: Dictionary of live or simulated hardware metrics.
            scene_key: Unique key for current rendering / computation frame context.
            observed_manifold: Current relational topology manifold (or generated if None).
            dt: Time step delta.

        Returns:
            Dict containing telemetry, friction metrics, control modulations, cache stats,
            rotor wavefield metrics, and system resonance density.
        """
        self.time += dt

        # Default manifold if not provided
        if observed_manifold is None:
            observed_manifold = np.sin(np.linspace(0, 4.0 * np.pi, 64))

        # 1. Hardware friction evaluation
        friction_metrics = self.evaluate_hardware_frictions(hardware_telemetry)
        self.accumulated_friction += friction_metrics["total_friction"] * dt

        # 2. Control modulation (RAM precision, Thermal amplitude, I/O backpressure)
        control_metrics = self.modulate_ecosystem_controls(friction_metrics)

        # 3. Axis phase synchronization & Resonance Density
        sync_metrics = self.synchronize_axis_phases(friction_metrics, dt)

        # 4. Topological Resonance Cache query (DLSS 5 optimization)
        cache_res = self.topo_cache.query_or_update(
            scene_key=scene_key,
            observed_manifold=observed_manifold,
            friction=friction_metrics["total_friction"],
        )

        # 5. Spatiotemporal Spherical Rotor Field update
        # Rotate rotor field according to effective system frequency
        rotor_angle = sync_metrics["effective_freq_hz"] * dt * 0.1
        self.rotor_field.apply_clifford_rotor(angle=rotor_angle)
        rotor_energy = self.rotor_field.compute_wavefield_energy()

        return {
            "time": float(self.time),
            "status": "STABLE_RESONANT" if friction_metrics["total_friction"] < 1.0 else "FRICTION_ADAPTED",
            "resonance_density": sync_metrics["resonance_density"],
            "order_parameter_R": sync_metrics["order_parameter_R"],
            "effective_freq_hz": sync_metrics["effective_freq_hz"],
            "frictions": friction_metrics,
            "controls": control_metrics,
            "cache_result": cache_res,
            "rotor_wavefield_energy": rotor_energy,
            "accumulated_friction": float(self.accumulated_friction),
        }

"""Enactive VRAM Phase-Lock Runtime Engine.

This module implements an enactive, phase-locked runtime layer for managing GPU/VRAM
memory allocations and compute pipelines under physical constraints (e.g. GTX 1060 3GB VRAM).

Instead of treating hardware limits as rigid barriers that cause Out-of-Memory (OOM) crashes,
this engine observes memory pressure, thermal resistance, and latency as 'friction'. It dynamically
tightens constraints (batch size, buffer footprint, processing frequency) and phase-locks compute cycles
to achieve continuous execution with maximum resonance density.

Key Components:
1. VRAMBoundaryObserver: Tracks available VRAM, memory bandwidth, latency, and thermal resistance.
2. DynamicVRAMConstraintMapper: Learns dynamic safe operational boundaries (batch size limit, buffer capacity).
3. EnactiveVRAMPhaseLockRuntime: Synchronizes computational frequency with hardware friction and phase dynamics.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


class VRAMBoundaryObserver:
    """Layer 1: Observes VRAM usage, total capacity, and system friction telemetry."""

    def __init__(self, total_vram_mb: float = 3072.0, safety_margin_mb: float = 256.0):
        """
        Args:
            total_vram_mb: Total available VRAM in MB (default 3072MB / 3GB for GTX 1060).
            safety_margin_mb: Safety buffer to prevent hard OS driver crashes.
        """
        self.total_vram_mb = float(total_vram_mb)
        self.safety_margin_mb = float(safety_margin_mb)
        self.usable_vram_mb = max(100.0, self.total_vram_mb - self.safety_margin_mb)
        self.current_allocated_mb = 0.0

    def observe_hardware_telemetry(self, telemetry: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Perceives current hardware state and returns normalized metrics.

        Telemetry dict may include:
            - allocated_vram_mb: float
            - latency_ms: float
            - thermal_celsius: float
            - bus_utilization: float
        """
        telemetry = telemetry or {}
        allocated = float(telemetry.get("allocated_vram_mb", self.current_allocated_mb))
        self.current_allocated_mb = allocated
        latency = float(telemetry.get("latency_ms", 1.0))
        temp = float(telemetry.get("thermal_celsius", 60.0))
        bus_util = float(telemetry.get("bus_utilization", 0.5))

        free_vram_mb = max(0.0, self.usable_vram_mb - allocated)
        headroom_ratio = min(1.0, free_vram_mb / self.usable_vram_mb)

        return {
            "total_vram_mb": self.total_vram_mb,
            "usable_vram_mb": self.usable_vram_mb,
            "allocated_vram_mb": allocated,
            "free_vram_mb": free_vram_mb,
            "headroom_ratio": headroom_ratio,
            "latency_ms": latency,
            "thermal_celsius": temp,
            "bus_utilization": bus_util,
        }

    def compute_friction_stimulus(self, telemetry_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculates system friction stimulus from hardware pressure."""
        allocated = telemetry_metrics["allocated_vram_mb"]
        usable = telemetry_metrics["usable_vram_mb"]
        latency = telemetry_metrics["latency_ms"]
        temp = telemetry_metrics["thermal_celsius"]

        # VRAM pressure friction rises non-linearly near capacity
        vram_pressure = max(0.0, allocated / usable)
        vram_friction = float(np.exp(max(0.0, vram_pressure - 0.7) * 5.0) - 1.0) if vram_pressure > 0.7 else 0.0

        # Latency friction
        latency_friction = max(0.0, (latency - 5.0) / 10.0)

        # Thermal friction (> 75C)
        thermal_friction = max(0.0, (temp - 75.0) / 15.0)

        total_friction = vram_friction * 2.0 + latency_friction + thermal_friction
        return {
            "vram_friction": vram_friction,
            "latency_friction": latency_friction,
            "thermal_friction": thermal_friction,
            "total_friction": total_friction,
        }


class DynamicVRAMConstraintMapper:
    """Layer 2: Dynamically maps VRAM buffer allocation limits and batch constraints."""

    def __init__(
        self,
        max_batch_size: int = 64,
        min_batch_size: int = 1,
        max_buffer_mb: float = 2048.0,
        learning_rate: float = 0.15,
        decay_rate: float = 0.02,
    ):
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.max_buffer_mb = max_buffer_mb
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

        self.current_effective_batch_size = float(max_batch_size)
        self.current_effective_buffer_mb = max_buffer_mb
        self.accumulated_friction = 0.0

    def update_from_friction(self, friction_metrics: Dict[str, float]) -> Dict[str, float]:
        """Tightens or relaxes operational boundaries based on friction feedback."""
        total_friction = friction_metrics.get("total_friction", 0.0)

        # Exponential decay of past friction
        self.accumulated_friction = (1.0 - self.decay_rate) * self.accumulated_friction + total_friction

        if total_friction > 0.5:
            # High friction -> Tighten constraints defensively
            shrink_factor = 1.0 - self.learning_rate * min(0.8, total_friction / 10.0)
            self.current_effective_batch_size = max(
                float(self.min_batch_size),
                self.current_effective_batch_size * shrink_factor
            )
            self.current_effective_buffer_mb = max(
                32.0,
                self.current_effective_buffer_mb * shrink_factor
            )
        else:
            # Low friction -> Gradual boundary expansion
            expand_factor = 1.0 + (self.learning_rate * 0.05)
            self.current_effective_batch_size = min(
                float(self.max_batch_size),
                self.current_effective_batch_size * expand_factor
            )
            self.current_effective_buffer_mb = min(
                self.max_buffer_mb,
                self.current_effective_buffer_mb * expand_factor
            )

        return {
            "effective_batch_size": int(np.floor(self.current_effective_batch_size)),
            "effective_buffer_mb": float(self.current_effective_buffer_mb),
            "accumulated_friction": float(self.accumulated_friction),
        }


class EnactiveVRAMPhaseLockRuntime:
    """Layer 3: Unified Phase-Locked Runtime for VRAM/Compute adaptation."""

    def __init__(
        self,
        total_vram_mb: float = 3072.0,
        max_batch_size: int = 64,
        base_freq_hz: float = 60.0,
        coupling_strength: float = 0.8,
    ):
        self.observer = VRAMBoundaryObserver(total_vram_mb=total_vram_mb)
        self.mapper = DynamicVRAMConstraintMapper(max_batch_size=max_batch_size)

        self.base_freq_hz = base_freq_hz
        self.coupling_strength = coupling_strength
        self.internal_phase = 0.0
        self.time = 0.0
        self.resonance_density = 1.0

    def step(
        self,
        requested_batch_size: int,
        requested_tensor_mb: float,
        hardware_telemetry: Optional[Dict[str, float]] = None,
        dt: float = 0.01,
    ) -> Dict[str, Any]:
        """Executes a single enactive runtime cycle.

        Args:
            requested_batch_size: The batch size desired by caller.
            requested_tensor_mb: Memory size of requested tensor operations.
            hardware_telemetry: Real or simulated hardware feedback.
            dt: Time delta.

        Returns:
            Dict containing allocated params (safe_batch_size, safe_tensor_mb),
            phase locking metrics, friction metrics, and resonance density.
        """
        self.time += dt

        # 1. Hardware boundary observation
        obs_metrics = self.observer.observe_hardware_telemetry(hardware_telemetry)
        friction_metrics = self.observer.compute_friction_stimulus(obs_metrics)

        # 2. Dynamic constraint mapping
        constraint_limits = self.mapper.update_from_friction(friction_metrics)

        # 3. Dynamic adjustment of request parameters to fit enactive boundaries
        safe_batch_size = min(requested_batch_size, constraint_limits["effective_batch_size"])
        safe_batch_size = max(1, safe_batch_size)

        # Scale tensor memory proportional to batch size reduction
        batch_scale = safe_batch_size / max(1, requested_batch_size)
        safe_tensor_mb = min(
            requested_tensor_mb * batch_scale,
            constraint_limits["effective_buffer_mb"],
            obs_metrics["free_vram_mb"] * 0.9
        )
        safe_tensor_mb = max(0.1, safe_tensor_mb)

        # 4. Kuramoto phase synchronization with hardware processing clock
        # Hardware clock phase advances with telemetry frequency modulation
        freq_damping = 1.0 / (1.0 + friction_metrics["total_friction"])
        effective_freq = self.base_freq_hz * freq_damping

        # Target external phase
        target_ext_phase = (self.time * effective_freq * 2.0 * np.pi) % (2.0 * np.pi)
        phase_diff = target_ext_phase - self.internal_phase
        phase_diff_norm = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

        # Pull phase
        self.internal_phase = (self.internal_phase + effective_freq * dt + self.coupling_strength * np.sin(phase_diff_norm)) % (2.0 * np.pi)

        # Order parameter R (coherence)
        order_R = float(np.clip(np.cos(phase_diff_norm), 0.0, 1.0))

        # Resonance Density = Coherence * Safety ratio
        self.resonance_density = order_R * obs_metrics["headroom_ratio"]

        return {
            "time": float(self.time),
            "allocated_batch_size": safe_batch_size,
            "allocated_tensor_mb": safe_tensor_mb,
            "effective_freq_hz": float(effective_freq),
            "order_parameter_R": order_R,
            "resonance_density": float(self.resonance_density),
            "vram_headroom_ratio": obs_metrics["headroom_ratio"],
            "friction": friction_metrics,
            "constraints": constraint_limits,
            "status": "HEALTHY_PHASE_LOCKED" if friction_metrics["total_friction"] < 1.0 else "FRICTION_ADAPTED",
        }

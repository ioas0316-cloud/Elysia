"""Unit tests for Enactive VRAM Phase-Lock Runtime Engine."""

import pytest
import numpy as np
from core.physics.enactive_vram_runtime import (
    VRAMBoundaryObserver,
    DynamicVRAMConstraintMapper,
    EnactiveVRAMPhaseLockRuntime,
)


def test_vram_observer_telemetry():
    observer = VRAMBoundaryObserver(total_vram_mb=3072.0, safety_margin_mb=256.0)
    assert observer.usable_vram_mb == 2816.0

    telemetry = {
        "allocated_vram_mb": 2000.0,
        "latency_ms": 10.0,
        "thermal_celsius": 80.0,
    }
    metrics = observer.observe_hardware_telemetry(telemetry)
    assert metrics["free_vram_mb"] == 816.0
    assert metrics["headroom_ratio"] == pytest.approx(816.0 / 2816.0)

    friction = observer.compute_friction_stimulus(metrics)
    assert friction["vram_friction"] > 0.0
    assert friction["latency_friction"] == pytest.approx(0.5)
    assert friction["thermal_friction"] == pytest.approx(5.0 / 15.0)


def test_dynamic_vram_constraint_mapper_friction_tightening():
    mapper = DynamicVRAMConstraintMapper(max_batch_size=64, min_batch_size=1, max_buffer_mb=2048.0)
    initial_batch = mapper.current_effective_batch_size

    # Simulate high friction
    friction_high = {"total_friction": 5.0}
    res = mapper.update_from_friction(friction_high)

    assert res["effective_batch_size"] < initial_batch
    assert res["effective_buffer_mb"] < 2048.0
    assert res["accumulated_friction"] > 0.0


def test_enactive_vram_runtime_adaptation():
    runtime = EnactiveVRAMPhaseLockRuntime(
        total_vram_mb=3072.0,
        max_batch_size=64,
        base_freq_hz=60.0
    )

    # Step under normal conditions
    step1 = runtime.step(requested_batch_size=64, requested_tensor_mb=1000.0)
    assert step1["allocated_batch_size"] == 64
    assert step1["status"] == "HEALTHY_PHASE_LOCKED"

    # Step under high VRAM friction
    high_friction_telemetry = {
        "allocated_vram_mb": 2700.0,
        "latency_ms": 25.0,
        "thermal_celsius": 85.0
    }
    step2 = runtime.step(
        requested_batch_size=64,
        requested_tensor_mb=1000.0,
        hardware_telemetry=high_friction_telemetry
    )

    assert step2["allocated_batch_size"] < 64
    assert step2["allocated_tensor_mb"] < 1000.0
    assert step2["status"] == "FRICTION_ADAPTED"

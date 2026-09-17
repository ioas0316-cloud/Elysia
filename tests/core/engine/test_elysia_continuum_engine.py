"""Tests for Elysia Wave-Frequency Continuum Architecture."""

import pytest
import numpy as np

from core.engine.continuum_stream_buffer import ContinuumStreamBuffer
from core.engine.baseline_delta_discriminator import BaselineDeltaDiscriminator
from core.engine.triadic_observation_loop import TriadicObservationLoop, DeterministicCausalReplayer


def test_continuum_stream_buffer_write_read_and_roll():
    """Tests mmap stream buffer writes, reads, and rolling ring wrapping."""
    buffer = ContinuumStreamBuffer(capacity_bytes=1024, auto_cleanup=True)

    data1 = np.ones(256, dtype=np.uint8) * 42
    seq_start1, len1 = buffer.write_stream(data1)
    assert seq_start1 == 0
    assert len1 == 256

    read_back1 = buffer.read_recent(256)
    assert len(read_back1) == 256
    assert np.all(np.frombuffer(read_back1, dtype=np.uint8) == 42)

    # Overflow buffer to trigger rolling overwrite
    data2 = np.ones(1024, dtype=np.uint8) * 99
    seq_start2, len2 = buffer.write_stream(data2)
    assert seq_start2 == 256
    assert len2 == 1024

    read_back2 = buffer.read_recent(1024)
    assert len(read_back2) == 1024
    assert np.all(np.frombuffer(read_back2, dtype=np.uint8) == 99)

    buffer.close()


def test_baseline_delta_discriminator_and_standing_wave_boundary():
    """Tests baseline update, delta extraction, and Proprioceptive Impedance Mismatch (wall) detection."""
    discriminator = BaselineDeltaDiscriminator(window_size=128, alpha=0.1, reflection_threshold=0.7)

    # 1. Normal sine wave input
    t = np.linspace(0, 2 * np.pi, 128)
    wave1 = np.sin(t)
    result1 = discriminator.process_incoming_signal(incoming_wave=wave1)

    assert "delta_residual" in result1
    assert "baseline_trajectory" in result1
    assert "groove_depth" in result1
    assert result1["standing_wave_reflection"] is False

    # 2. Simulate strong intervention wave met with unyielding opposing reflection (Standing Wave / Wall)
    intervention = np.sin(t)
    # Opposing/reflected wave with high amplitude residual
    opposing_reflection = np.sin(t) * 2.0
    result2 = discriminator.process_incoming_signal(
        incoming_wave=opposing_reflection,
        intervention_wave=intervention
    )

    assert result2["reflection_coefficient"] > 0.7
    assert result2["standing_wave_reflection"] is True
    assert result2["is_world_boundary"] is True


def test_triadic_observation_loop_and_deterministic_replayer():
    """Tests complete triadic observation loop cycle and deterministic reincarnation replay."""
    capacity = 128
    buffer = ContinuumStreamBuffer(capacity_bytes=8192, auto_cleanup=True)
    discriminator = BaselineDeltaDiscriminator(window_size=capacity, alpha=0.1)
    loop = TriadicObservationLoop(stream_buffer=buffer, discriminator=discriminator, cycle_capacity=capacity)

    # Run 5 observation cycles
    t = np.linspace(0, 4 * np.pi, capacity)
    seq_starts = []
    bytes_list = []

    for i in range(5):
        sensory_wave = np.sin(t + i * 0.5) + np.random.normal(0, 0.05, capacity)
        step_res = loop.step(sensory_wave)
        seq_starts.append(step_res["seq_start"])
        bytes_list.append(step_res["bytes_written"])

    assert loop.cycle_count == 5

    # Deterministic Causal Replay of Cycle 1
    replayer = DeterministicCausalReplayer(stream_buffer=buffer, discriminator=discriminator)
    replay_res = replayer.replay_sequence(seq_start=seq_starts[0], length_bytes=bytes_list[0], dtype=np.float32)

    assert replay_res["replayed_seq_start"] == seq_starts[0]
    assert len(replay_res["replayed_wave"]) == capacity
    assert "delta_residual" in replay_res["replayed_analysis"]

    buffer.close()

"""
Tests for UART-Inspired Asynchronous Causal Observer & Phase Encapsulation.
"""

import pytest
import numpy as np
from core.sensory.uart_async_causal_observer import (
    UARTAsyncCausalObserver,
    MicroCausalNode,
    CompositeNode,
    FramePacketNode,
    StreamFluidMetrics
)


def generate_uart_waveform(
    char: str = 'A',
    bit_duration_us: float = 104.0,
    noise_level: float = 0.05,
    transient_noise: bool = True
) -> tuple:
    """Helper to generate a noisy UART waveform for a given ASCII character."""
    byte_val = ord(char)
    # UART Frame: Start Bit (0) + 8 Data Bits (LSB first) + Stop Bit (1)
    bits = [0] + [(byte_val >> i) & 1 for i in range(8)] + [1]

    dt = 2.0  # 2 us per sample point
    samples_per_bit = int(bit_duration_us / dt)

    idle_prefix = [1.0] * 50  # Idle High state
    signal_voltages = []

    for idx, bit in enumerate(bits):
        v = 1.0 if bit == 1 else 0.0
        for s in range(samples_per_bit):
            # Add transient noise at boundary edges if requested
            if transient_noise and (s < 5 or s > samples_per_bit - 6):
                noisy_v = v + np.random.normal(0, 0.35)
            else:
                noisy_v = v + np.random.normal(0, noise_level)
            signal_voltages.append(float(noisy_v))

    full_voltages = np.array(idle_prefix + signal_voltages)
    time_series = np.arange(len(full_voltages)) * dt
    return time_series, full_voltages


def test_edge_detection_and_cadence():
    observer = UARTAsyncCausalObserver(bit_duration_us=104.0)
    time_series, voltage_series = generate_uart_waveform('A', bit_duration_us=104.0)

    start_idx = observer.detect_causal_edge(time_series, voltage_series)
    assert start_idx is not None
    assert start_idx >= 50  # Edge after idle prefix

    # Test cadence estimation with multiple edges
    edge_times = [100.0, 204.0, 308.0, 412.0]
    cadence = observer.estimate_cadence(edge_times)
    assert pytest.approx(cadence, abs=1.0) == 104.0


def test_middle_sampling_vs_boundary_sampling():
    observer = UARTAsyncCausalObserver(bit_duration_us=104.0)
    time_series, voltage_series = generate_uart_waveform('A', bit_duration_us=104.0, transient_noise=True)

    start_idx = observer.detect_causal_edge(time_series, voltage_series)
    assert start_idx is not None

    mid_samples, bound_samples = observer.sample_middle_vs_boundary(
        time_series, voltage_series, start_idx, num_bits=8
    )

    assert len(mid_samples) == 8
    assert len(bound_samples) == 8

    # Middle sampling should yield correct bit pattern for 'A' (0x41 = 01000001 -> LSB 1,0,0,0,0,0,1,0)
    # LSB first: bit 0 = 1, bit 1-5 = 0, bit 6 = 1, bit 7 = 0
    mid_bits = [m.bit_value for m in mid_samples]
    assert mid_bits == [1, 0, 0, 0, 0, 0, 1, 0]

    # Phase stability in middle sampling should be higher than boundary sampling
    avg_mid_stability = np.mean([m.phase_stability for m in mid_samples])
    avg_bound_stability = np.mean([b.phase_stability for b in bound_samples])
    assert avg_mid_stability > avg_bound_stability


def test_phase_encapsulation_composite_node():
    observer = UARTAsyncCausalObserver(bit_duration_us=104.0)
    time_series, voltage_series = generate_uart_waveform('A', bit_duration_us=104.0)

    start_idx = observer.detect_causal_edge(time_series, voltage_series)
    mid_samples, _ = observer.sample_middle_vs_boundary(time_series, voltage_series, start_idx, num_bits=8)

    composite = observer.encapsulate_phase_graph(mid_samples)

    assert isinstance(composite, CompositeNode)
    assert composite.byte_value == 0x41  # 65 = 'A'
    assert composite.symbol == 'A'
    assert composite.hex_repr == '0x41'
    assert composite.binary_repr == '01000001'
    assert composite.emergent_properties['character'] == 'A'
    assert composite.emergent_properties['is_printable'] is True


def test_frame_packet_and_fluid_backpressure():
    observer = UARTAsyncCausalObserver(bit_duration_us=104.0)
    composites = []

    for char in ['H', 'E', 'L', 'L', 'O']:
        t_series, v_series = generate_uart_waveform(char)
        start_idx = observer.detect_causal_edge(t_series, v_series)
        mid_samples, _ = observer.sample_middle_vs_boundary(t_series, v_series, start_idx)
        comp = observer.encapsulate_phase_graph(mid_samples)
        composites.append(comp)

    # Frame Packet Assembly
    frame = observer.assemble_frame_packet(composites, header_len=2)
    assert isinstance(frame, FramePacketNode)
    assert len(frame.header_nodes) == 2
    assert len(frame.payload_nodes) == 3
    assert frame.total_bytes == 5

    # Fluid Backpressure under normal capacity (10)
    metrics_normal = observer.evaluate_fluid_backpressure(composites, buffer_capacity=10)
    assert metrics_normal.buffer_density == 0.5
    assert metrics_normal.backpressure == 0.0

    # Fluid Backpressure under overload capacity (4)
    metrics_overload = observer.evaluate_fluid_backpressure(composites, buffer_capacity=4)
    assert metrics_overload.buffer_density == 1.0
    assert metrics_overload.backpressure > 0.0
    assert metrics_overload.latency_impedance > 1.0

"""
UART-Inspired Asynchronous Causal Observer & Phase Encapsulation Engine.

This module embodies the physical and causal principles of UART hardware communication:
1. Causal Edge Detection (Start Bit): Triggering observation without a global clock via falling edge transition.
2. Cadence Synchronization (Baud Rate): Aligning internal observer pacing to external phenomenon cadence.
3. Middle Sampling (Phase Stability Point): Avoiding boundary transient noise by sampling at phase equilibrium.
4. Phase Encapsulation (Graph Contraction): Fusing micro bit causal graphs into high-level Composite Nodes on Stop Bit completion.
5. Fractal Scale Invariance: Extending micro bit interactions to macro fluid/stream dynamic backpressure.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


@dataclass
class MicroCausalNode:
    """Microscopic causal node representing a single bit interval or voltage transition."""
    bit_index: int
    sample_time: float
    voltage: float
    bit_value: int
    is_middle_sample: bool
    phase_stability: float  # Measure of signal stability at sampling moment


@dataclass
class CompositeNode:
    """Macroscopic composite node created via Phase Encapsulation (Graph Contraction).

    Hides internal microscopic timing manifold while exposing emergent properties
    and structural interface pins.
    """
    node_id: str
    symbol: str
    byte_value: int
    hex_repr: str
    binary_repr: str
    micro_graph: List[MicroCausalNode]
    emergent_properties: Dict[str, Any] = field(default_factory=dict)
    interface_pins: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FramePacketNode:
    """Higher-scale crystal node composed of multiple Composite Nodes (e.g., Header + Payload)."""
    frame_id: str
    header_nodes: List[CompositeNode]
    payload_nodes: List[CompositeNode]
    total_bytes: int
    structural_integrity: bool


@dataclass
class StreamFluidMetrics:
    """Macroscopic fluid dynamic properties of encapsulated data streaming."""
    buffer_density: float
    bandwidth_rate: float
    backpressure: float
    latency_impedance: float


class UARTAsyncCausalObserver:
    """Asynchronous Causal Observer using UART physical interactions to parse,
    encapsulate, and scale microscopic physical signals into macroscopic symbols.
    """

    def __init__(self, idle_voltage: float = 1.0, bit_duration_us: float = 104.0):
        self.idle_voltage = idle_voltage
        self.bit_duration_us = bit_duration_us  # e.g., 104.17 us for 9600 bps
        self.micro_nodes: List[MicroCausalNode] = []
        self.composite_nodes: List[CompositeNode] = []

    def detect_causal_edge(self, time_series: np.ndarray, voltage_series: np.ndarray, threshold: float = 0.5) -> Optional[int]:
        """Feature 1: Detects the falling edge (Start Bit) without a global clock.

        Returns the index where High -> Low state transition occurs.
        """
        for i in range(1, len(voltage_series)):
            if voltage_series[i - 1] > threshold and voltage_series[i] <= threshold:
                return i
        return None

    def estimate_cadence(self, edge_times: List[float]) -> float:
        """Feature 2: Dynamic Baud Rate / Cadence estimation from edge intervals."""
        if len(edge_times) < 2:
            return self.bit_duration_us
        diffs = np.diff(edge_times)
        # Find minimum non-zero difference as base bit duration
        valid_diffs = diffs[diffs > 10.0]  # Filter out micro noise
        if len(valid_diffs) > 0:
            estimated = float(np.min(valid_diffs))
            return estimated
        return self.bit_duration_us

    def sample_middle_vs_boundary(
        self,
        time_series: np.ndarray,
        voltage_series: np.ndarray,
        start_idx: int,
        num_bits: int = 8
    ) -> Tuple[List[MicroCausalNode], List[MicroCausalNode]]:
        """Feature 3: Middle Sampling (Phase Stability Point) vs Boundary Sampling.

        Demonstrates how middle sampling avoids boundary transient noise.
        """
        start_time = time_series[start_idx]
        middle_samples = []
        boundary_samples = []

        for k in range(1, num_bits + 1):
            # Middle sampling target: start_time + (k + 0.5) * bit_duration
            t_mid = start_time + (k + 0.5) * self.bit_duration_us
            # Boundary sampling target: start_time + k * bit_duration
            t_bound = start_time + k * self.bit_duration_us

            idx_mid = int(np.argmin(np.abs(time_series - t_mid)))
            idx_bound = int(np.argmin(np.abs(time_series - t_bound)))

            v_mid = float(voltage_series[idx_mid])
            v_bound = float(voltage_series[idx_bound])

            bit_val_mid = 1 if v_mid > 0.5 else 0
            bit_val_bound = 1 if v_bound > 0.5 else 0

            # Measure stability as local variance around sampling point
            window = 3
            mid_window = voltage_series[max(0, idx_mid - window):min(len(voltage_series), idx_mid + window + 1)]
            bound_window = voltage_series[max(0, idx_bound - window):min(len(voltage_series), idx_bound + window + 1)]

            mid_stability = 1.0 - float(np.std(mid_window)) if len(mid_window) > 0 else 1.0
            bound_stability = 1.0 - float(np.std(bound_window)) if len(bound_window) > 0 else 0.0

            middle_samples.append(MicroCausalNode(
                bit_index=k - 1,
                sample_time=float(time_series[idx_mid]),
                voltage=v_mid,
                bit_value=bit_val_mid,
                is_middle_sample=True,
                phase_stability=max(0.0, mid_stability)
            ))

            boundary_samples.append(MicroCausalNode(
                bit_index=k - 1,
                sample_time=float(time_series[idx_bound]),
                voltage=v_bound,
                bit_value=bit_val_bound,
                is_middle_sample=False,
                phase_stability=max(0.0, bound_stability)
            ))

        return middle_samples, boundary_samples

    def encapsulate_phase_graph(self, micro_nodes: List[MicroCausalNode]) -> CompositeNode:
        """Feature 4: Graph Contraction / Phase Encapsulation.

        Collapses 8 micro bit nodes into a single Composite Node (Atom -> Molecule).
        LSB first assembly.
        """
        byte_val = 0
        for node in micro_nodes:
            byte_val |= (node.bit_value << node.bit_index)

        try:
            symbol = chr(byte_val) if 32 <= byte_val <= 126 else f"0x{byte_val:02X}"
        except Exception:
            symbol = f"0x{byte_val:02X}"

        node_id = f"composite_0x{byte_val:02X}_{int(micro_nodes[0].sample_time)}"

        composite = CompositeNode(
            node_id=node_id,
            symbol=symbol,
            byte_value=byte_val,
            hex_repr=f"0x{byte_val:02X}",
            binary_repr=f"{byte_val:08b}",
            micro_graph=micro_nodes,
            emergent_properties={
                "character": symbol,
                "ascii_code": byte_val,
                "semantic_mass": float(np.mean([n.phase_stability for n in micro_nodes])),
                "is_printable": 32 <= byte_val <= 126
            },
            interface_pins={
                "input_anchor": micro_nodes[0].sample_time,
                "output_anchor": micro_nodes[-1].sample_time,
                "bit_count": len(micro_nodes)
            }
        )
        self.composite_nodes.append(composite)
        return composite

    def assemble_frame_packet(self, composites: List[CompositeNode], header_len: int = 2) -> FramePacketNode:
        """Feature 5a: Molecule -> Crystal Assembly (Header + Payload Frame)."""
        if len(composites) <= header_len:
            header = composites
            payload = []
        else:
            header = composites[:header_len]
            payload = composites[header_len:]

        frame = FramePacketNode(
            frame_id=f"frame_{composites[0].node_id}",
            header_nodes=header,
            payload_nodes=payload,
            total_bytes=len(composites),
            structural_integrity=len(composites) >= header_len
        )
        return frame

    def evaluate_fluid_backpressure(
        self,
        composite_stream: List[CompositeNode],
        buffer_capacity: int = 10,
        flow_rate_per_ms: float = 2.0
    ) -> StreamFluidMetrics:
        """Feature 5b: Macro Fluid Dynamic Stream Backpressure & Impedance.

        Models how byte Composite Nodes act as particles in a fluid stream,
        creating backpressure and latency friction when buffer limits are reached.
        """
        stream_len = len(composite_stream)
        density = stream_len / float(buffer_capacity)

        # Backpressure arises non-linearly when density exceeds equilibrium threshold (0.7)
        if density > 0.7:
            backpressure = float(np.exp(2.5 * (density - 0.7)) - 1.0)
        else:
            backpressure = 0.0

        latency_impedance = 1.0 + backpressure * 2.5
        effective_bandwidth = flow_rate_per_ms / latency_impedance

        return StreamFluidMetrics(
            buffer_density=min(1.0, density),
            bandwidth_rate=max(0.1, effective_bandwidth),
            backpressure=backpressure,
            latency_impedance=latency_impedance
        )

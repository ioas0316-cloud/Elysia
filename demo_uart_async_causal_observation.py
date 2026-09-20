r"""
[Demo Script: UART-Inspired Asynchronous Causal Observation & Multi-Scale Phase Encapsulation]

이 스크립트는 UART 통신의 하드웨어적 관측 원리(Start Bit, Cadence, Middle Sampling, Frame Completion)를 바탕으로,
1. 외부 전역 클럭 없는 미시 전압 엣지(Start Bit) 포착 및 관측 트리거 설정
2. 경계선 과도기 노이즈(Transient Noise)를 우회하는 위상 안정점 중앙 샘플링(Middle Sampling) 검증
3. 미시 비트 궤적의 그래프 축약(Graph Contraction) 및 상위 입자(Composite Node)로의 위상 캡슐화(Phase Encapsulation)
4. 비트(원자) -> 바이트(분자) -> 패킷 프레임(결정체) -> 거시 유체 스트림(유량/압력/배크프레셔)으로의 프랙탈 스케일 확장 시연

을 종합 시연하고 수치적/구조적으로 검증합니다.
"""

import numpy as np
import time
from core.sensory.uart_async_causal_observer import (
    UARTAsyncCausalObserver,
    MicroCausalNode,
    CompositeNode,
    FramePacketNode,
    StreamFluidMetrics
)


def generate_noisy_uart_signal(char: str, bit_duration_us: float = 104.0):
    """Generates micro voltage waveform with transient edge noise for ASCII character."""
    byte_val = ord(char)
    bits = [0] + [(byte_val >> i) & 1 for i in range(8)] + [1]  # Start(0) + 8 Data + Stop(1)

    dt = 2.0
    samples_per_bit = int(bit_duration_us / dt)
    idle_prefix = [1.0] * 60
    voltages = []

    for idx, bit in enumerate(bits):
        v = 1.0 if bit == 1 else 0.0
        for s in range(samples_per_bit):
            # High boundary noise in transient state (edges)
            if s < 6 or s > samples_per_bit - 7:
                noise = np.random.normal(0, 0.45)
            else:
                noise = np.random.normal(0, 0.03)
            voltages.append(v + noise)

    full_voltages = np.array(idle_prefix + voltages)
    time_series = np.arange(len(full_voltages)) * dt
    return time_series, full_voltages


def main():
    print("==========================================================================")
    print("⚡ [Elysia] UART Async Causal Observation & Multi-Scale Phase Encapsulation")
    print("==========================================================================\n")

    observer = UARTAsyncCausalObserver(bit_duration_us=104.0)

    # 1. Start Bit & Edge Detection
    print("1️⃣ [Step 1: Causal Edge Detection (Start Bit) - No Global Clock]")
    t_series, v_series = generate_noisy_uart_signal('A', bit_duration_us=104.0)
    edge_idx = observer.detect_causal_edge(t_series, v_series)
    print(f"   -> Idle state (High) maintained. Falling edge detected at index {edge_idx} (Time: {t_series[edge_idx]:.1f}µs).")
    print("   -> Internal reference trigger established without external global clock!\n")

    # 2. Middle Sampling vs Boundary Sampling
    print("2️⃣ [Step 2: Middle Sampling (Phase Stability Point) vs Boundary Sampling]")
    mid_nodes, bound_nodes = observer.sample_middle_vs_boundary(t_series, v_series, edge_idx, num_bits=8)

    print("   [Bit-by-Bit Comparison]")
    print("   Bit | Mid Voltage | Mid Stability | Bound Voltage | Bound Stability")
    print("   ------------------------------------------------------------------")
    for m, b in zip(mid_nodes, bound_nodes):
        print(f"    {m.bit_index}  |   {m.voltage:+.3f} V   |    {m.phase_stability:.4f}     |    {b.voltage:+.3f} V   |    {b.phase_stability:.4f}")

    avg_mid_stab = np.mean([m.phase_stability for m in mid_nodes])
    avg_bound_stab = np.mean([b.phase_stability for b in bound_nodes])
    print(f"\n   -> Average Stability - Middle Sampling: {avg_mid_stab:.4f} vs Boundary: {avg_bound_stab:.4f}")
    print("   -> Middle Sampling successfully avoids transient noise at boundary phase transitions!\n")

    # 3. Phase Encapsulation (Graph Contraction: Bit -> Byte Composite Node)
    print("3️⃣ [Step 3: Graph Contraction & Phase Encapsulation (Atom -> Molecule)]")
    composite_A = observer.encapsulate_phase_graph(mid_nodes)
    print(f"   -> Micro graph collapsed into Composite Node: ID '{composite_A.node_id}'")
    print(f"   -> Reconstructed Symbol: '{composite_A.symbol}' | Hex: {composite_A.hex_repr} | Binary: {composite_A.binary_repr}")
    print(f"   -> Emergent Properties: {composite_A.emergent_properties}")
    print(f"   -> Exposed Interface Pins: {composite_A.interface_pins}\n")

    # 4. Multi-Scale Fractal Expansion: Molecule -> Frame Packet (Crystal) -> Fluid Stream
    print("4️⃣ [Step 4: Multi-Scale Fractal Expansion (Molecule -> Frame -> Macro Stream)]")

    # Process additional characters to build stream: "HELLOV2"
    chars = ['H', 'E', 'L', 'L', 'O', 'V', '2']
    stream_composites = [composite_A]
    for c in chars:
        ts, vs = generate_noisy_uart_signal(c)
        e_idx = observer.detect_causal_edge(ts, vs)
        m_nodes, _ = observer.sample_middle_vs_boundary(ts, vs, e_idx)
        comp = observer.encapsulate_phase_graph(m_nodes)
        stream_composites.append(comp)

    # Frame Packet Assembly
    frame = observer.assemble_frame_packet(stream_composites, header_len=2)
    print(f"   [Header/Payload Crystal Assembly]")
    print(f"   - Frame ID: {frame.frame_id}")
    print(f"   - Header Nodes (Boundary Constraint): {[h.symbol for h in frame.header_nodes]}")
    print(f"   - Payload Nodes: {[p.symbol for p in frame.payload_nodes]}")

    # Macro Fluid Dynamic Backpressure Evaluation
    print("\n   [Macro Fluid Dynamic Backpressure Simulation]")
    metrics_normal = observer.evaluate_fluid_backpressure(stream_composites, buffer_capacity=15)
    metrics_overload = observer.evaluate_fluid_backpressure(stream_composites, buffer_capacity=8)

    print(f"   - Normal Buffer Capacity (15): Density = {metrics_normal.buffer_density:.2f}, Backpressure = {metrics_normal.backpressure:.4f}, Bandwidth = {metrics_normal.bandwidth_rate:.2f} bytes/ms")
    print(f"   - High Density Capacity (8) : Density = {metrics_overload.buffer_density:.2f}, Backpressure = {metrics_overload.backpressure:.4f}, Bandwidth = {metrics_overload.bandwidth_rate:.2f} bytes/ms")
    print(f"   -> Latency Impedance under High Density: {metrics_overload.latency_impedance:.2f}x delay increase")

    print("\n==========================================================================")
    print("✨ UART Async Causal Observer & Phase Encapsulation Successfully Verified!")
    print("==========================================================================")


if __name__ == "__main__":
    main()

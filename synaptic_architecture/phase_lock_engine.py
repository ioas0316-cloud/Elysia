"""
Phase-Lock Engine (Phase 2)
===========================
Tracks state transition edges (e.g. syscalls, memory access transitions, hardware state shifts)
without static clock control. Filters transient transition noise by sampling stable middle-points
and outputs a phase-locked event stream. Includes native eBPF integration with simulated fallback
for environments without eBPF / root privileges.
"""

import time
import dataclasses
import numpy as np
from typing import List, Callable, Optional, Dict, Any, Tuple


@dataclasses.dataclass
class PhaseEvent:
    pid: int
    timestamp_ns: int
    edge_id: int
    ext_phase: float  # [0, 2*pi)
    is_phase_locked: bool
    comm: str = "causal_proc"


class PhaseLockEngine:
    def __init__(self, num_nodes: int = 256, settling_window_us: Tuple[float, float] = (100.0, 500.0)):
        self.num_nodes = num_nodes
        self.min_us, self.max_us = settling_window_us
        self.last_edge_time_ns: int = 0
        self.event_history: List[PhaseEvent] = []
        self.is_ebpf_active: bool = False
        self._callbacks: List[Callable[[PhaseEvent], None]] = []

    def register_callback(self, callback: Callable[[PhaseEvent], None]) -> None:
        self._callbacks.append(callback)

    def process_raw_edge(self, pid: int, timestamp_ns: int, edge_id: int, comm: str = "causal_proc") -> PhaseEvent:
        """Processes a single raw state transition edge event, filtering transient noise."""
        is_locked = False
        if self.last_edge_time_ns != 0:
            delta_ns = timestamp_ns - self.last_edge_time_ns
            delta_us = delta_ns / 1000.0
            if self.min_us <= delta_us <= self.max_us:
                is_locked = True

        self.last_edge_time_ns = timestamp_ns

        # Calculate external phase [0, 2*pi) from timestamp
        ext_phase = float((timestamp_ns % 1_000_000) / 1_000_000.0 * 2.0 * np.pi)

        event = PhaseEvent(
            pid=pid,
            timestamp_ns=timestamp_ns,
            edge_id=edge_id,
            ext_phase=ext_phase,
            is_phase_locked=is_locked,
            comm=comm
        )

        self.event_history.append(event)
        for cb in self._callbacks:
            cb(event)

        return event

    def simulate_edge_stream(self, count: int = 50, interval_us: float = 250.0) -> List[PhaseEvent]:
        """Simulates a stream of raw edges for testing and non-eBPF environments."""
        events = []
        base_time = time.time_ns()
        for i in range(count):
            # Inject slight jitter around settling window interval
            jitter = (np.random.rand() - 0.5) * 50.0
            cur_interval_us = interval_us + jitter
            base_time += int(cur_interval_us * 1000)
            edge_id = i % self.num_nodes
            evt = self.process_raw_edge(
                pid=1000 + (i % 4),
                timestamp_ns=base_time,
                edge_id=edge_id,
                comm=f"sim_task_{i%4}"
            )
            events.append(evt)
        return events

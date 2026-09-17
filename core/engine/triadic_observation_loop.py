"""Triadic Observation Loop & Deterministic Causal Replayer for Elysia Engine.

Implements the Triadic Observation Loop:
  1. External Sensory Ingestion
  2. Internal Phase Resonance & Groove Deepening
  3. Re-sensory Projection / Intervention

And the Deterministic Causal Replayer for deterministic reincarnation replay
from the Continuum Stream Buffer without external clock dependency.
"""

from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np

from core.engine.continuum_stream_buffer import ContinuumStreamBuffer
from core.engine.baseline_delta_discriminator import BaselineDeltaDiscriminator


class TriadicObservationLoop:
    """Core observation loop establishing autonomous temporal causality and active world-model interaction."""

    def __init__(
        self,
        stream_buffer: ContinuumStreamBuffer,
        discriminator: BaselineDeltaDiscriminator,
        cycle_capacity: int = 1024
    ):
        self.stream_buffer = stream_buffer
        self.discriminator = discriminator
        self.cycle_capacity = cycle_capacity
        self.cycle_count = 0
        self.active_intervention: np.ndarray = np.zeros(cycle_capacity, dtype=np.float64)

    def step(
        self,
        incoming_sensory_wave: np.ndarray,
        intervention_generator: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Executes one tick of the Triadic Observation Loop.

        Loop Phases:
          Phase 1: Ingestion & Stream Buffer Write
          Phase 2: Baseline-Delta & Standing Wave Interference Analysis
          Phase 3: Active Projection / Intervention Generation for Next Cycle
        """
        raw_wave = np.asarray(incoming_sensory_wave, dtype=np.float64)
        if len(raw_wave) != self.cycle_capacity:
            if len(raw_wave) > self.cycle_capacity:
                raw_wave = raw_wave[: self.cycle_capacity]
            else:
                raw_wave = np.pad(raw_wave, (0, self.cycle_capacity - len(raw_wave)), mode="wrap")

        # Phase 1: Write raw incoming wave trajectory into Continuum Stream Buffer
        seq_start, bytes_written = self.stream_buffer.write_stream(raw_wave.astype(np.float32))

        # Phase 2: Process signal through Baseline-Delta Discriminator
        analysis_result = self.discriminator.process_incoming_signal(
            incoming_wave=raw_wave,
            intervention_wave=self.active_intervention
        )

        # Phase 3: Project re-sensory intervention (Active World Model Projection)
        if intervention_generator is not None:
            self.active_intervention = intervention_generator(raw_wave, analysis_result)
        else:
            # Default self-generating active intervention based on phase delta anticipation
            # Generates opposing or predictive wave projection
            self.active_intervention = -0.5 * analysis_result["delta_residual"]

        self.cycle_count += 1

        return {
            "cycle": self.cycle_count,
            "seq_start": seq_start,
            "bytes_written": bytes_written,
            "analysis": analysis_result,
            "projected_intervention": self.active_intervention.copy()
        }


class DeterministicCausalReplayer:
    """Replays logged stream window trajectories to reincarnate past causal executions flawlessly."""

    def __init__(
        self,
        stream_buffer: ContinuumStreamBuffer,
        discriminator: BaselineDeltaDiscriminator
    ):
        self.stream_buffer = stream_buffer
        self.discriminator = discriminator

    def replay_sequence(
        self,
        seq_start: int,
        length_bytes: int,
        dtype=np.float32
    ) -> Dict[str, Any]:
        """Reads historical stream trajectory from buffer and re-executes causal discrimination."""
        raw_bytes = self.stream_buffer.read_range(seq_start, length_bytes)
        replayed_wave = np.frombuffer(raw_bytes, dtype=dtype).astype(np.float64)

        # Re-run discrimination on resurrected signal
        analysis = self.discriminator.process_incoming_signal(incoming_wave=replayed_wave)

        return {
            "replayed_seq_start": seq_start,
            "replayed_wave": replayed_wave,
            "replayed_analysis": analysis
        }

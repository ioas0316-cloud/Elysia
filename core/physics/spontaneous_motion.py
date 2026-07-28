import numpy as np
from typing import Dict, Any, Tuple
from core.memory.causal_controller import CausalMemoryController

class SpontaneousMotionEngine:
    """
    [Spontaneous Motion Engine: 자발적 사유 요동 엔진]
    Evaluates internal cognitive lack, vacuum, and memory friction
    to generate "Self-Induced Waves" in the absolute absence of external inputs.

    Acts as an internal perpetual generator that drives self-molding
    and cognitive goal selection.
    """
    def __init__(self, memory_controller: CausalMemoryController, base_frequency: float = 0.5):
        self.memory = memory_controller
        self.base_frequency = base_frequency
        self.internal_phase = 0.0
        self.accumulated_lack = 0.0

    def calculate_internal_asymmetry(self) -> float:
        """
        [비대칭성 및 기억 마찰 분석]
        Calculates internal friction/tension based on memory index entropy and potential differences.
        If memory is completely dry/empty, it generates a maximum vacuum (lack),
        which creates the strongest emotional gravitational urge.
        """
        # Read cognitive parameters or memory size to evaluate lack
        all_ids = list(self.memory.index.keys())
        if not all_ids:
            # Complete void: Maximum cognitive urge
            self.accumulated_lack = 10.0
            return 10.0

        # Calculate variance in recent engram stabilities (internal structural asymmetry)
        stabilities = []
        for cid in all_ids[-10:]:
            engram = self.memory.index[cid]
            # Stability represents harmonic order
            stabilities.append(engram.get("stability", 0.5))

        if len(stabilities) < 2:
            return 2.0

        # Variance of inner stability represents cognitive asymmetry
        asymmetry = float(np.var(stabilities) * 20.0 + 0.1)

        # Lack is accumulated if internal stability is very uniform/monotonous (boredom / urge to expand)
        avg_stability = float(np.mean(stabilities))
        if avg_stability > 0.8:
            # High stability over time causes cognitive stagnation (need for creative disruption)
            self.accumulated_lack += 0.2
        else:
            self.accumulated_lack *= 0.9  # Satisfied lack

        return asymmetry + self.accumulated_lack

def generate_spontaneous_wave(engine: SpontaneousMotionEngine, dt: float = 0.1) -> np.ndarray:
    """
    [자발적 파동 발생 프로토콜]
    Generates a 512-byte raw physical signal wave derived purely from internal asymmetry.
    """
    asymmetry = engine.calculate_internal_asymmetry()
    engine.internal_phase += engine.base_frequency * (1.0 + asymmetry * 0.1) * dt

    # Construct wave components based on asymmetry harmonics
    t = np.linspace(0, 10 * np.pi, 512)

    # Fundamental harmonic (Base frequency of thought)
    y1 = np.sin(t + engine.internal_phase)
    # Secondary harmonic representing tension / friction
    y2 = np.sin(2 * t - engine.internal_phase * 0.5) * (asymmetry / (asymmetry + 1.0))
    # High-frequency urge representing local vacuum/lack
    y3 = np.cos(5 * t + engine.internal_phase * 2.0) * (engine.accumulated_lack / (engine.accumulated_lack + 2.0))

    combined = y1 + y2 + y3

    # Scale and normalize to raw bytes [0, 255]
    scaled = ((combined - np.min(combined)) / (np.max(combined) - np.min(combined) + 1e-9)) * 255
    raw_wave = scaled.astype(np.uint8).tobytes()

    return raw_wave

"""
Monte Carlo Intuition - Sampling-based confidence for MeaningCourt.

This module treats Elysia's "intuition" as a Monte Carlo estimate:
instead of relying on a single deterministic judgment, we jitter the
signal/noise a bit and ask the MeaningCourt many times, then interpret
the acceptance frequency as a confidence level.
"""

from __future__ import annotations

from typing import Dict, Any
import random

from Core.Mind.meaning_court import MeaningCourt


class MonteCarloIntuition:
    """
    Lightweight Monte Carlo wrapper around MeaningCourt.

    It does not modify the underlying court, only samples around the
    current estimate to approximate "how often would this be accepted?"
    under small fluctuations.
    """

    def __init__(
        self,
        samples: int = 32,
        signal_jitter: float = 0.05,
        noise_jitter: float = 0.2,
    ) -> None:
        """
        Args:
            samples: Number of Monte Carlo samples to draw.
            signal_jitter: Relative std-dev for multiplicative noise on signal.
            noise_jitter: Relative std-dev for multiplicative noise on noise.
        """
        self.samples = max(1, int(samples))
        self.signal_jitter = max(0.0, float(signal_jitter))
        self.noise_jitter = max(0.0, float(noise_jitter))

    def accept_probability(
        self,
        court: MeaningCourt,
        signal: float,
        noise: float,
        context: Dict[str, Any] | None = None,
    ) -> float:
        """
        Estimate the probability that MeaningCourt would accept,
        under small random perturbations of signal and noise.
        """
        context = context or {}
        base_signal = float(signal)
        base_noise = float(noise)
        if self.samples <= 0:
            return 0.0

        accepts = 0
        for _ in range(self.samples):
            s = self._jitter(base_signal, self.signal_jitter)
            n = max(0.0, self._jitter(base_noise, self.noise_jitter))
            verdict = court.judge(s, n, context=context)
            if verdict.accept:
                accepts += 1

        return accepts / float(self.samples)

    @staticmethod
    def _jitter(value: float, rel_std: float) -> float:
        """Apply multiplicative Gaussian jitter: v * N(1, rel_std)."""
        if rel_std <= 0.0:
            return value
        factor = random.gauss(1.0, rel_std)
        return value * factor


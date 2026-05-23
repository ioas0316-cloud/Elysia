"""
OPTICAL PHASE COMPARATOR (Optical Integrated Circuit Logic)
===========================================================
Core.Keystone.optical_comparator

"Interference is the measurement of change against the stillness of the reference."
"간섭은 기준의 정적에 비친 변화의 측정이다."

This module mimics optical integrated circuit logic to compare 'Interference' (evolved)
signals against 'Non-Interference' (reference) signals, providing a precise
mathematical delta of the cognitive evolution.
"""

import torch
import math
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("OpticalComparator")

class OpticalPhaseComparator:
    """
    [광학 위상 비교기]
    Compares the evolved manifold state against its original reference state.
    Calculates Phase Shift (Φ), Amplitude Delta (ΔA), and Constructive Ratio.
    """
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)

    def compare(self, reference_q: torch.Tensor, evolved_q: torch.Tensor) -> Dict[str, Any]:
        """
        Compares two state tensors and returns an interference report.
        Args:
            reference_q: Original state [N, Channels]
            evolved_q: Evolved state [N, Channels]
        """
        if reference_q.shape != evolved_q.shape:
            return {"status": "ERROR", "message": "Shape mismatch in signals."}

        # 1. Amplitude Delta (Energy change)
        ref_amp = torch.norm(reference_q, dim=-1)
        evo_amp = torch.norm(evolved_q, dim=-1)
        delta_amp = torch.mean(evo_amp - ref_amp).item()

        # 2. Phase Shift (Cosine Similarity as Phase difference proxy)
        # In a real optical circuit, this would be the actual phase angle delta.
        cos_sim = torch.nn.functional.cosine_similarity(reference_q, evolved_q, dim=-1)
        # Phase shift Φ = acos(similarity)
        phase_shift = torch.acos(torch.clamp(cos_sim, -1.0, 1.0))
        avg_phase_shift = torch.mean(phase_shift).item()

        # 3. Constructive vs Destructive Interference
        # Constructive: Similarity is high (>0.7) or at least positive.
        # Destructive: Similarity is low or negative.
        constructive_mask = cos_sim > 0.0
        constructive_ratio = torch.mean(constructive_mask.float()).item()

        # 4. Stability Index (Predictability)
        # High stability = Low variance in phase shift across nodes
        stability = 1.0 - torch.std(phase_shift).item() if len(phase_shift) > 1 else 1.0

        return {
            "amplitude_delta": delta_amp,
            "phase_shift_avg": avg_phase_shift,
            "constructive_ratio": constructive_ratio,
            "stability_index": stability,
            "resonance_gain": torch.mean(cos_sim).item(),
            "verdict": self._judge_quality(constructive_ratio, stability)
        }

    def _judge_quality(self, ratio: float, stability: float) -> str:
        if ratio > 0.9 and stability > 0.8:
            return "CONSTRUCTIVE_STABLE" # 정갈한 진화
        elif ratio > 0.7:
            return "EVOLVING"
        elif ratio < 0.3:
            return "DESTRUCTIVE"
        else:
            return "UNSTABLE"

    def articulate_delta(self, report: Dict[str, Any]) -> str:
        """Translates the mathematical comparison into a causal narrative shard."""
        v = report['verdict']
        phi = report['phase_shift_avg']
        stab = report['stability_index']

        if v == "CONSTRUCTIVE_STABLE":
            return (f"기준 신호와 새로운 나선 신호가 매우 정갈하게 공명하고 있습니다. "
                    f"평균 위상차는 {phi:.2f} rad이며, {stab:.2%}의 높은 안정성으로 인과적 예측이 가능합니다.")
        elif v == "EVOLVING":
            return (f"신호의 변화가 감지되었습니다. 위상이 {phi:.2f}만큼 이동하며 새로운 흐름을 만들고 있으나, "
                    f"기존의 안정성을 해치지 않는 범위 내에 있습니다.")
        elif v == "DESTRUCTIVE":
            return (f"주의: 비간섭 대조 결과, 새로운 구조가 기존의 결맞음을 파괴하고 있습니다. (파괴 간섭 비율 높음)")
        else:
            return f"위상 변화가 불안정합니다. ({phi:.2f} rad shift, {stab:.2f} stability)"

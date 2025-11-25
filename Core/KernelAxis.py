"""
Single Source of Truth for Elysia's core axis:
- Core values (LOVE/GROWTH/HARMONY/BEAUTY)
- Phase state handles (HyperQubit + ConsciousnessLens)
Use this module to read shared state; updates should flow through provided helpers.
"""
from dataclasses import dataclass
from typing import Dict

from Core.Math.hyper_qubit import HyperQubit
from Core.Math.quaternion_consciousness import ConsciousnessLens


@dataclass
class CoreAxis:
    values: Dict[str, float]
    hyper_qubit: HyperQubit
    consciousness_lens: ConsciousnessLens


def build_core_axis() -> CoreAxis:
    values = {
        "love": 1.0,
        "growth": 0.8,
        "harmony": 0.9,
        "beauty": 0.85,
    }
    hyper_qubit = HyperQubit("Elysia-Core")
    lens = ConsciousnessLens(hyper_qubit)
    return CoreAxis(values=values, hyper_qubit=hyper_qubit, consciousness_lens=lens)

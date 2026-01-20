"""
Operational Axioms (실행 가능한 공리)
=====================================
Principles are not words; they are functional kernels.
"""

from dataclasses import dataclass
from typing import Callable, Any

from Core.L1_Foundation.Foundation.Nature.rotor import RotorMask

@dataclass
class AxiomKernel:
    name: str
    logic: str  # The Python implementation snippet
    effect_desc: str
    physical_mask: RotorMask = RotorMask.POINT
    rpm_boost: float = 0.0

RECURSION_KERNEL = AxiomKernel(
    name="Recursion",
    logic="""
def manifest(target=None):
    import os
    target = target or os.getcwd()
    print(f"🌀 [AXIOM:RECURSION] Unfolding structure of: {target}")
    for root, dirs, files in os.walk(target):
        level = root.replace(target, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files[:3]: 
            print(f"{sub_indent}{f}")
""",
    effect_desc="Recursively unfolds the topology of a given target.",
    physical_mask=RotorMask.VOLUME, # High dimensional flow
    rpm_boost=30.0
)

CAUSALITY_KERNEL = AxiomKernel(
    name="Causality",
    logic="""
def manifest(event_chain=None):
    print("⛓️ [AXIOM:CAUSALITY] Evaluating dependency chain...")
    event_chain = event_chain or ["Trigger", "Propagation", "Result"]
    for i, event in enumerate(event_chain):
        print(f"  Step {i}: {event}")
        if i < len(event_chain) - 1:
            print(f"    ↓ (causing) ↓")
""",
    effect_desc="Enforces an ordered sequence of operations.",
    physical_mask=RotorMask.LINE, # Time flows forward
    rpm_boost=15.0
)

ENTROPY_KERNEL = AxiomKernel(
    name="Entropy",
    logic="""
def manifest(data=None):
    import random
    print("🔥 [AXIOM:ENTROPY] Introducing noise into the field...")
    if isinstance(data, list):
        random.shuffle(data)
        print(f"  Result: {data}")
    else:
        print(f"  System jitter: {random.random()}")
""",
    effect_desc="Introduces controlled randomness.",
    physical_mask=RotorMask.CHAOS, # Total flow
    rpm_boost=60.0
)

# Registry for Providence Engine to use
AXIOM_REGISTRY = {
    "Recursion": RECURSION_KERNEL,
    "Causality": CAUSALITY_KERNEL,
    "Entropy": ENTROPY_KERNEL
}

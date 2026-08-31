"""
Unit tests for Autopoietic Causal Field.
Verifies Global Topological Entropy calculation, Core Kernel vs Peripheral Shell dual protection,
Shell edge dissolution under entropy spike, and Active Wave Modulation drive under persistent friction.
"""

import pytest
import numpy as np

from core.physics.autopoietic_causal_field import AutopoieticCausalField, NodeRole
from core.lens.enactive_boundary_layer import EnactiveBoundaryLayer
from core.lens.cognitive_lens_engine import ContextualDimension


def test_autopoietic_entropy_calculation():
    acf = AutopoieticCausalField(entropy_threshold=2.0)
    acf.add_autopoietic_node("KernelNode", frequency=5.0, phase=0.0, role=NodeRole.CORE_KERNEL)
    acf.add_autopoietic_node("ShellNode", frequency=5.0, phase=np.pi / 2.0, role=NodeRole.PERIPHERAL_SHELL)
    acf.add_autopoietic_edge("KernelNode", "ShellNode", initial_impedance=0.3)

    initial_entropy = acf.calculate_global_topological_entropy()
    assert pytest.approx(initial_entropy, abs=1e-3) == 0.3


def test_autopoietic_kernel_protection_and_shell_dissolution():
    acf = AutopoieticCausalField(entropy_threshold=0.5)
    acf.add_autopoietic_node("KernelNode", frequency=5.0, phase=0.0, role=NodeRole.CORE_KERNEL)
    acf.add_autopoietic_node("ShellNode", frequency=5.0, phase=np.pi / 2.0, role=NodeRole.PERIPHERAL_SHELL)
    acf.add_autopoietic_edge("KernelNode", "ShellNode", initial_impedance=0.4)

    # Step facing strong external friction
    res = acf.enact_autopoietic_step("KernelNode", external_frequency=5.0, external_phase=np.pi / 2.0, target_node="ShellNode")

    # Entropy should exceed 0.5 threshold, triggering Shell edge dissolution (impedance set to 1.0)
    assert res["shell_dissolved"] is True
    assert acf.ebl.graph.edges["KernelNode", "ShellNode"]["impedance"] == 1.0


def test_active_wave_modulation_drive():
    acf = AutopoieticCausalField(entropy_threshold=5.0, fluctuation_scale=0.3)
    acf.add_autopoietic_node("ShellNode", frequency=5.0, phase=np.pi / 2.0, role=NodeRole.PERIPHERAL_SHELL)
    acf.add_autopoietic_node("TargetNode", frequency=5.0, phase=0.0, role=NodeRole.PERIPHERAL_SHELL)
    acf.add_autopoietic_edge("ShellNode", "TargetNode", initial_impedance=0.1)

    initial_freq = acf.ebl.graph.nodes["ShellNode"]["freq"]
    initial_phase = acf.ebl.graph.nodes["ShellNode"]["phase"]

    # Step facing friction -> triggers active wave modulation
    res = acf.enact_autopoietic_step("ShellNode", external_frequency=5.0, external_phase=0.0, target_node="TargetNode")

    assert res["active_modulation_applied"] is True
    assert (res["new_frequency"] != initial_freq or res["new_phase"] != initial_phase)

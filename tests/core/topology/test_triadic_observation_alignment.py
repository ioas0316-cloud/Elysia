import pytest
from core.physics.causal_engine import CausalNodePy, PHASE_GAS, PHASE_FLUID, PHASE_CRYSTAL
from core.topology.triadic_observation_alignment import TrinityDiagnosticLens


def test_trinity_diagnostic_bias_and_mirror_purity():
    lens = TrinityDiagnosticLens(max_bias_tolerance=300.0)

    self_node = CausalNodePy.create(phase=PHASE_FLUID, potential=1500, bond_op=1, topo_offset=2)
    other_node = CausalNodePy.create(phase=PHASE_FLUID, potential=1000, bond_op=1, topo_offset=2)

    diag = lens.observe_and_diagnose(self_node, other_node)

    assert diag["delta_potential"] == 500.0
    assert diag["phase_discrepancy"] == 0.0
    assert diag["bias_refraction"] == 500.0
    assert diag["is_closed_world_risk"] == 1.0
    assert diag["mirror_purity"] < 0.3


def test_trinity_self_calibration_loop():
    lens = TrinityDiagnosticLens(refraction_decay_rate=0.2, max_bias_tolerance=300.0)

    self_node = CausalNodePy.create(phase=PHASE_CRYSTAL, potential=2000, bond_op=1, topo_offset=0)
    other_node = CausalNodePy.create(phase=PHASE_FLUID, potential=1000, bond_op=1, topo_offset=5)

    diag_initial = lens.observe_and_diagnose(self_node, other_node)
    assert diag_initial["is_closed_world_risk"] == 1.0

    # Execute calibration step
    calibrated_node = lens.calibrate_self_lens(self_node, other_node, diag_initial)

    # Potential should have moved closer to 1000
    assert calibrated_node.potential < 2000
    # Topo offset should have shifted towards 5
    assert calibrated_node.topo_offset == 1
    # Phase state should relax to PHASE_FLUID from PHASE_CRYSTAL due to high bias risk
    assert calibrated_node.phase_state == PHASE_FLUID

    diag_calibrated = lens.observe_and_diagnose(calibrated_node, other_node)
    assert diag_calibrated["bias_refraction"] < diag_initial["bias_refraction"]
    assert diag_calibrated["mirror_purity"] > diag_initial["mirror_purity"]

import pytest
import numpy as np

from core.physics.causal_engine import CausalNodePy, PrincipleRederivationEngine, PHASE_FLUID, PHASE_CRYSTAL
from core.topology.triadic_observation_alignment import TrinityDiagnosticLens
from synaptic_architecture.schema_isomorphism_engine import SchemaIsomorphismEngine


def test_full_integrated_causal_pipeline():
    # 1. Initialize Principle Re-derivation Computation Engine (1st Priority)
    re_derivation_engine = PrincipleRederivationEngine(num_nodes=64, tension_threshold=600.0)
    raw_potentials = (np.sin(np.linspace(0, 2 * np.pi, 64)) * 800.0 + 1200.0).astype(np.float32)
    re_derivation_engine.initialize_field(potentials=raw_potentials)

    engine_res = re_derivation_engine.re_derive_phenomenon(target_potential_gradient=100.0, max_steps=20)
    assert engine_res["converged_step"] > 0

    # 2. Observe with Trinity Diagnostic Lens (2nd Priority)
    lens = TrinityDiagnosticLens(max_bias_tolerance=400.0)
    self_node = CausalNodePy(engine_res["field_nodes"][0])
    other_node = CausalNodePy.create(phase=PHASE_FLUID, potential=1000, bond_op=0, topo_offset=1)

    diag = lens.observe_and_diagnose(self_node, other_node)
    assert "mirror_purity" in diag
    assert "bias_refraction" in diag

    # Perform Self-Lens Calibration
    calibrated_self = lens.calibrate_self_lens(self_node, other_node, diag)
    diag_after = lens.observe_and_diagnose(calibrated_self, other_node)
    assert diag_after["bias_refraction"] <= diag["bias_refraction"]

    # 3. Demonstrate Schema Isomorphism & Inverse Decompression (3rd Priority)
    iso_engine = SchemaIsomorphismEngine(num_nodes=64)
    schema = iso_engine.compress_phenomenon_to_schema(raw_potentials)
    decompressed = iso_engine.decompress_and_re_derive(schema, max_steps=20)
    metrics = iso_engine.verify_topological_isomorphism(raw_potentials, schema, decompressed)

    assert metrics["zero_hallucination_index"] >= 0.0
    assert metrics["compression_ratio"] > 1.0

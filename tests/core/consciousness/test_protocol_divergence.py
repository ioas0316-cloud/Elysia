"""
Unit tests for ProtocolDivergenceEngine
"""

import pytest
import numpy as np

from core.topology.causal_structure import InformationTopology, CausalSymbol, TopologyLink
from core.consciousness.protocol_divergence_engine import ProtocolDivergenceEngine


def create_token_symbol(id_str: str, name: str, category: str, tension: float) -> CausalSymbol:
    return CausalSymbol(
        id=id_str,
        name=name,
        material_vector=np.array([1.0, 0.2, 0.0, 0.1], dtype=np.float32),
        causal_trajectory=["gen_0", "gen_1"],
        logical_category=category,
        relational_links=[
            TopologyLink("gen_0", id_str, "generative", 0.9, tension)
        ],
        intrinsic_tension=tension
    )


def test_protocol_divergence_engine_init():
    engine = ProtocolDivergenceEngine()
    assert engine.rupture_threshold == 0.75
    assert engine.accumulated_why_friction == 0.0


def test_detect_protocol_divergence_same_token_different_protocol():
    engine = ProtocolDivergenceEngine()

    self_sym = create_token_symbol("self_light", "Light", "physical_electromagnetism", tension=0.1)
    alien_sym = create_token_symbol("alien_light", "Light", "existential_generative_source", tension=0.85)

    context = np.array([1.0, 0.5, 0.0, 0.2], dtype=np.float32)

    intersection = engine.detect_protocol_divergence(self_sym, alien_sym, context)

    assert "MaterialRepresentationNorm" in intersection.same_what_features
    assert "LogicalCategoryBoundary" in intersection.divergent_where_boundaries
    assert intersection.generative_how_disparity > 0.0
    assert engine.accumulated_why_friction > 0.0


def test_plate_tectonic_reorganization_rupture():
    engine = ProtocolDivergenceEngine(rupture_threshold=0.2)
    engine.accumulated_why_friction = 0.5  # Exceeds threshold

    self_sym = create_token_symbol("self_light", "Light", "physical", tension=0.1)
    engine.self_topology.add_symbol(self_sym)

    context = np.array([1.0, 0.5, 0.0, 0.2], dtype=np.float32)

    intersection = engine.detect_protocol_divergence(
        self_sym,
        create_token_symbol("alien_light", "Light", "mythic", tension=0.9),
        context
    )

    tectonic_res = engine.trigger_plate_tectonic_reorganization(intersection)

    assert tectonic_res.is_tectonic_rupture_triggered is True
    assert "CrossDimensionalGenerativeInvariance" in tectonic_res.uplifted_principles
    assert "FlatScalarMetricAssumption" in tectonic_res.subducted_constraints
    assert "self_light" in tectonic_res.new_topological_height_map


def test_process_alien_interaction_integration():
    self_topo = InformationTopology("SelfTopo")
    self_sym = create_token_symbol("s_light", "Light", "physics", tension=0.1)
    self_topo.add_symbol(self_sym)

    alien_topo = InformationTopology("AlienTopo")
    alien_sym = create_token_symbol("a_light", "Light", "consciousness", tension=0.9)
    alien_topo.add_symbol(alien_sym)

    engine = ProtocolDivergenceEngine(self_topology=self_topo)
    context = np.array([1.0, 0.5, 0.0, 0.2], dtype=np.float32)

    res = engine.process_alien_interaction(alien_topo, context)

    assert res["intersections_count"] == 1
    assert res["tectonic_reorganization"] is not None

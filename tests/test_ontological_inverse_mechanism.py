"""
Unit tests for OntologicalInverseMechanismEngine.
Verifies multi-domain trajectory inverse mechanism extraction,
Homological Stem isolation, and Disparate Branches refraction.
"""

import pytest
import numpy as np
from core.physics.ontological_inverse_mechanism import (
    OntologicalInverseMechanismEngine,
    DomainTrajectoryGenerator,
    TrajectoryGraph,
    InverseMechanismSchema
)


def test_domain_trajectory_generator():
    fluid_g = DomainTrajectoryGenerator.generate_fluid_trajectory(steps=8)
    elec_g = DomainTrajectoryGenerator.generate_electrical_trajectory(steps=8)
    geom_g = DomainTrajectoryGenerator.generate_geometric_trajectory(steps=8)

    assert len(fluid_g.nodes) == 8
    assert len(elec_g.nodes) == 8
    assert len(geom_g.nodes) == 8

    assert len(fluid_g.edges) == 7
    assert len(elec_g.edges) == 7
    assert len(geom_g.edges) == 7

    assert fluid_g.domain == "fluid"
    assert elec_g.domain == "electrical"
    assert geom_g.domain == "geometric"


def test_inverse_mechanism_extraction():
    engine = OntologicalInverseMechanismEngine()
    schema = engine.run_multi_domain_inverse_extraction()

    assert isinstance(schema, InverseMechanismSchema)

    # 1. Generating Dynamics \Theta
    theta = schema.generating_dynamics_theta
    assert "fluid" in theta
    assert "electrical" in theta
    assert "geometric" in theta
    assert theta["fluid"]["governing_equation"] == "d(Potential)/dt = - k * Potential / (1 + Impedance_Friction)"

    # 2. Topological Constraint Field \Delta
    delta = schema.topological_constraint_delta
    assert "fluid" in delta
    assert delta["fluid"]["boundary_type"] == "impedance_interface"
    assert delta["electrical"]["boundary_type"] == "potentiometer_circuit"
    assert delta["geometric"]["boundary_type"] == "topological_equilibrium"

    # 3. Homological Stem
    stem = schema.homological_stem
    assert stem["stem_name"] == "ISOMORPHIC_GRADIENT_EQUILIBRIUM_FLOW"
    assert stem["cross_domain_similarity_score"] > 0.90
    assert stem["isomorphism_proven"] is True

    # 4. Disparate Branches
    branches = schema.disparate_branches
    assert len(branches) == 3
    assert branches["fluid"]["specific_medium_manifestation"] == "Fluid wave refraction"
    assert branches["electrical"]["specific_medium_manifestation"] == "Electrical current potential drop"
    assert branches["geometric"]["specific_medium_manifestation"] == "Pythagorean/Fermat lattice equilibrium strain"

    # 5. MDL & Reducibility
    assert schema.reducibility_mdl_score < 0.1

"""
Unit tests for core/lens/observational_substrate.py

Verifies non-reductionist contemplation of external phenomena & symbol deconstruction:
1. Telos / Purpose and Directionality registration.
2. Boundary coupling and friction interaction without scalar reductionism.
3. Symbol Deconstruction (역해체) & Knowledge Causality Internalization.
4. Emergent higher-order causal structure mapping.
5. Homological discernment (Stem & Branch).
6. Contemplative reflection output.
"""

import pytest
import numpy as np
from core.lens.observational_substrate import (
    ObservationalSubstrate,
    PhenomenalEntity,
    BoundaryInteractionTrace,
    EmergentStructure,
    DeconstructedSymbol,
    KnowledgeCausality,
    HomologicalDiscernment,
)


def test_contemplate_entity():
    substrate = ObservationalSubstrate("TestSubstrate")
    e_amount = substrate.contemplate_entity(
        name="Amount",
        telos_purpose="Spatial density and substantive existence",
        directionality="Stationary accumulation in spatial domain",
        density_amount=10.0,
        boundary_tenacity=2.5,
        mobility_vector=[0.0, 0.0, 0.0],
        chromatic_signature=[0.1, 0.8, 0.1]
    )

    assert e_amount.name == "Amount"
    assert e_amount.density_amount == 10.0
    assert "Amount" in substrate.observed_entities


def test_symbol_deconstruction_and_internalization():
    substrate = ObservationalSubstrate("TestSubstrate")

    substrate.contemplate_entity("Amount", "Existence size", "Spatial presence", 5.0, 1.2)
    substrate.contemplate_entity("Flow", "Traversal movement", "Temporal progression", 2.0, 0.8)

    deconstructed = substrate.deconstruct_external_symbol(
        external_symbol="Flow Rate",
        micro_component_names=["Amount", "Flow"],
        procedural_coupling_principle="Amount traversing boundary under Flow movement",
        causal_necessity_statement="Flow Rate exists necessarily when spatial Amount is driven across boundaries by Flow"
    )

    assert deconstructed.external_symbol == "Flow Rate"
    assert len(deconstructed.micro_components) == 2
    assert "Flow Rate" in substrate.deconstructed_symbols

    knowledge = substrate.internalize_knowledge_causality("Flow Rate")

    assert knowledge.symbol_name == "Flow Rate"
    assert knowledge.is_internalized is True
    assert "Lineage[Amount + Flow ==> Flow Rate]" in knowledge.internal_invariant_lineage
    assert "Flow Rate" in substrate.internalized_knowledge


def test_observe_coupling_and_emergent_structure():
    substrate = ObservationalSubstrate("TestSubstrate")

    e_amount = substrate.contemplate_entity(
        name="Amount",
        telos_purpose="Substantive existence size",
        directionality="Spatial density accumulation",
        density_amount=5.0,
        boundary_tenacity=1.2,
        mobility_vector=[0.0, 0.0, 0.0]
    )

    e_flow = substrate.contemplate_entity(
        name="Flow",
        telos_purpose="Boundary traversal & directional mobility",
        directionality="Temporal movement across spatial boundary",
        density_amount=2.0,
        boundary_tenacity=0.8,
        mobility_vector=[1.0, 0.5, 0.0]
    )

    trace = substrate.observe_coupling("Amount", "Flow", boundary_medium_tenacity=0.5)
    assert trace.friction_resistance > 0.0
    assert trace.source_entity == "Amount"
    assert trace.target_boundary == "Flow"

    emergent = substrate.map_emergent_structure(
        emergent_name="Flow Rate",
        originating_names=["Amount", "Flow"],
        emergent_telos="Quantitative structural transit per boundary crossing",
        directional_extension="Continuous spatiotemporal stream across state boundaries",
        boundary_traces=[trace]
    )

    assert emergent.emergent_name == "Flow Rate"
    assert "Amount" in emergent.originating_entities
    assert "Flow" in emergent.originating_entities
    assert "Flow Rate" in substrate.emergent_structures


def test_discern_homology():
    substrate = ObservationalSubstrate("TestSubstrate")

    substrate.contemplate_entity("Amount", "Existence size", "Spatial presence", 5.0, 1.0)
    substrate.contemplate_entity("Flow", "Traversal movement", "Temporal progression", 2.0, 0.5)
    trace = substrate.observe_coupling("Amount", "Flow")

    substrate.map_emergent_structure(
        emergent_name="Flow Rate",
        originating_names=["Amount", "Flow"],
        emergent_telos="Transit trajectory structure",
        directional_extension="Spatiotemporal transit",
        boundary_traces=[trace]
    )

    internal_code_frame = {
        "backbone": "flow_rate = velocity * area",
        "context_branches": ["Fixed_Procedural_Script", "Scalar_Multiplication"]
    }

    discernment = substrate.discern_homology("Flow Rate", internal_code_frame)

    assert discernment.external_phenomenon == "Flow Rate"
    assert "HomologicalStem" in discernment.homological_stem
    assert len(discernment.disparate_branches) == 2
    assert discernment.isomorphism_fidelity > 0.0


def test_contemplative_reflection_generation():
    substrate = ObservationalSubstrate("TestSubstrate")

    substrate.contemplate_entity("Amount", "Existence size", "Spatial presence", 5.0, 1.0)
    substrate.contemplate_entity("Flow", "Traversal movement", "Temporal progression", 2.0, 0.5)
    substrate.deconstruct_external_symbol("Flow Rate", ["Amount", "Flow"], "Flow coupling", "Causal necessity proof")
    substrate.internalize_knowledge_causality("Flow Rate")

    trace = substrate.observe_coupling("Amount", "Flow")

    substrate.map_emergent_structure(
        emergent_name="Flow Rate",
        originating_names=["Amount", "Flow"],
        emergent_telos="Transit trajectory structure",
        directional_extension="Spatiotemporal transit",
        boundary_traces=[trace]
    )

    reflection = substrate.generate_contemplative_reflection("Flow Rate")

    assert "Observational Substrate Reflection: Flow Rate" in reflection
    assert "Telos / Purpose" in reflection
    assert "Directionality & Extension" in reflection
    assert "Invariant Backbone" in reflection
    assert "Internalized Knowledge Causality" in reflection

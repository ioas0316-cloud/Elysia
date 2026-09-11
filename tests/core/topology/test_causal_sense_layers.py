"""
Tests for domain-inherent causal sense layers.
"""

from core.topology.causal_sense_layers import (
    CausalSenseLayerEngine,
    CouplingResponse,
)
from core.topology.informational_phase_observation import ChromaticVector


def test_foundational_layers_are_registered_as_domain_senses():
    engine = CausalSenseLayerEngine.with_foundational_layers(target_dimension=8)

    assert "microbial_layer" in engine.layers
    assert "language_layer" in engine.layers
    assert "mathematics_layer" in engine.layers
    assert "physics_layer" in engine.layers
    assert "chemistry_layer" in engine.layers

    language = engine.layers["language_layer"]
    assert language.principles[0].name == "grammar_reference_flow"
    assert "syntax_dependency" in language.principles[0].invariants


def test_domain_hint_couples_ligand_to_its_own_sense_layer():
    engine = CausalSenseLayerEngine.with_foundational_layers(target_dimension=8)
    ligand = engine.ingest_ligand(
        ligand_id="sentence_grammar",
        content="A sentence binds subject, verb, object, tense, and reference.",
        modality="language",
        chromatic=ChromaticVector(flux=0.9, order=1.2, entropy=0.3),
        domain_hint="language_layer",
    )

    couplings = engine.couple(ligand)

    assert len(couplings) == 1
    assert couplings[0].layer_name == "language_layer"
    assert couplings[0].response in {
        CouplingResponse.ASSIMILATE,
        CouplingResponse.QUARANTINE,
        CouplingResponse.MUTATE,
    }
    assert "syntax_dependency" in couplings[0].expressed_invariants


def test_high_friction_external_information_can_generate_mutated_principle():
    engine = CausalSenseLayerEngine.with_foundational_layers(target_dimension=8)
    chemistry = engine.layers["chemistry_layer"]
    chemistry.plasticity = 0.9
    initial_count = len(chemistry.principles)

    ligand = engine.ingest_ligand(
        ligand_id="reaction_shock",
        content=[10.0, -8.0, 6.0, -4.0, 2.0, -1.0, 0.5, -0.25],
        modality="chemistry",
        chromatic=ChromaticVector(flux=2.5, order=0.2, entropy=1.4),
        domain_hint="chemistry_layer",
    )

    coupling = engine.couple(ligand)[0]

    assert coupling.response in {
        CouplingResponse.MUTATE,
        CouplingResponse.REJECT,
        CouplingResponse.QUARANTINE,
    }
    if coupling.response is CouplingResponse.MUTATE:
        assert len(chemistry.principles) == initial_count + 1
        assert coupling.generated_principle is not None
        assert "chemistry_coupling" in coupling.generated_principle.invariants


def test_without_domain_hint_all_layers_receive_external_signal():
    engine = CausalSenseLayerEngine.with_foundational_layers(target_dimension=8)
    ligand = engine.ingest_ligand(
        ligand_id="falling_body",
        content="force acceleration boundary condition energy conservation",
        modality="physics",
        chromatic=ChromaticVector(flux=1.2, order=1.0, entropy=0.2),
    )

    couplings = engine.couple(ligand)

    assert len(couplings) == len(engine.layers)
    assert {c.layer_name for c in couplings} == set(engine.layers.keys())

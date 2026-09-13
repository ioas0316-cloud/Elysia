import pytest
from core.engine.assimilation_loop import (
    CausalConservationNode,
    ExternalCausalSignal,
    IsomorphicKnowledgeAssimilationLoop,
    PhaseTopologicalReconstructionEngine,
    PhaseTopologicalReconstructionEngineExt,
    SealedAttractor,
    SealedAttractorRestructureLoop,
)


@pytest.fixture
def default_assimilation_loop():
    """Default assimilation loop with 0.05 threshold."""
    return IsomorphicKnowledgeAssimilationLoop(resonance_threshold=0.05)


@pytest.fixture
def reconstruction_engine():
    """Main PhaseTopologicalReconstructionEngine instance."""
    return PhaseTopologicalReconstructionEngine()


@pytest.fixture
def reconstruction_engine_ext():
    """Advanced PhaseTopologicalReconstructionEngineExt instance with SealedAttractor loop."""
    return PhaseTopologicalReconstructionEngineExt()


@pytest.fixture
def valid_wave_signal():
    """Valid wave mechanics external signal that reaches resonance."""
    return ExternalCausalSignal(
        signal_id="WAVE_001",
        boundary_conditions=["CONTINUOUS_MEDIUM_REQUIRED", "DISCONTINUOUS_IMPULSE_OK"],
        causal_relationships={
            "Spatial_Displacement": "Restoring_Force",
            "Restoring_Force": "Phase_Oscillation",
        },
    )


@pytest.fixture
def ecosystem_trophic_cascade_signal():
    """Multi-layer ecosystem trophic cascade signal."""
    return ExternalCausalSignal(
        signal_id="ECO_TROPHIC_CASCADE_01",
        boundary_conditions=[
            "ENERGY_CONSERVATION_BETWEEN_TROPHIC_LEVELS",
            "NEGATIVE_FEEDBACK_ATTRACTOR_REQUIRED",
            "DISCONTINUOUS_EXTINCTION_PREVENTED",
        ],
        causal_relationships={
            "Apex_Predator_Population": "Herbivore_Foraging_Behavior",
            "Herbivore_Foraging_Behavior": "Vegetation_Root_Density",
            "Vegetation_Root_Density": "Soil_Erosion_Resistance",
            "Soil_Erosion_Resistance": "Drought_Resilience_Capacity",
            "Drought_Resilience_Capacity": "Apex_Predator_Population",
        },
    )


@pytest.fixture
def high_tension_invalid_signal():
    """Invalid signal triggering high friction tension V_t."""
    return ExternalCausalSignal(
        signal_id="BROKEN_SIGNAL_999",
        boundary_conditions=["UNSTABLE_BOUNDARY_CONDITION"],
        causal_relationships={},  # Empty graph -> V_t spike
    )


class TestIsomorphicKnowledgeAssimilationLoop:

    def test_step1_deconstruct_and_extract(self, default_assimilation_loop, valid_wave_signal):
        """Step 1: Deconstruction & Invariant Extraction."""
        invariant = default_assimilation_loop.deconstruct_and_extract(valid_wave_signal)

        assert invariant["invariant_id"] == "INV_WAVE_001"
        assert "Spatial_Displacement" in invariant["core_morphisms"]
        assert "CONTINUOUS_MEDIUM_REQUIRED" in invariant["boundary_rules"]

    def test_step2_isomorphic_map(self, default_assimilation_loop, valid_wave_signal):
        """Step 2: Isomorphic Mapping (G_ext -> G_int)."""
        invariant = default_assimilation_loop.deconstruct_and_extract(valid_wave_signal)
        mapped = default_assimilation_loop.isomorphic_map(invariant)

        assert "INT_Spatial_Displacement" in mapped["mapped_graph"]
        assert mapped["mapped_graph"]["INT_Spatial_Displacement"] == "INT_Restoring_Force"

    def test_step3_run_internal_simulation_tension_calculation(
        self, default_assimilation_loop, valid_wave_signal, high_tension_invalid_signal
    ):
        """Step 3: Internal Generative Simulation & Friction Tension (V_t) Calculation."""
        inv_valid = default_assimilation_loop.deconstruct_and_extract(valid_wave_signal)
        mapped_valid = default_assimilation_loop.isomorphic_map(inv_valid)
        tension_valid = default_assimilation_loop.run_internal_simulation(mapped_valid)

        assert tension_valid <= default_assimilation_loop.resonance_threshold

        inv_broken = default_assimilation_loop.deconstruct_and_extract(high_tension_invalid_signal)
        mapped_broken = default_assimilation_loop.isomorphic_map(inv_broken)
        tension_broken = default_assimilation_loop.run_internal_simulation(mapped_broken)

        assert tension_broken > default_assimilation_loop.resonance_threshold

    def test_step4_phase_resonance_and_cc_node_freeze(
        self, default_assimilation_loop, valid_wave_signal
    ):
        """Step 4: Phase Resonance & CC-Node Freeze."""
        node = default_assimilation_loop.assimilate_knowledge(valid_wave_signal)

        assert isinstance(node, CausalConservationNode)
        assert node.node_id == "CCNODE_WAVE_001"
        assert "CCNODE_WAVE_001" in default_assimilation_loop.assimilated_cc_nodes
        assert node.stabilized_tension <= default_assimilation_loop.resonance_threshold

    def test_multi_layer_ecosystem_trophic_cascade_assimilation(
        self, default_assimilation_loop, ecosystem_trophic_cascade_signal
    ):
        """Multi-layer ecosystem trophic cascade scenario verification."""
        node = default_assimilation_loop.assimilate_knowledge(ecosystem_trophic_cascade_signal)

        assert node is not None
        assert node.node_id == "CCNODE_ECO_TROPHIC_CASCADE_01"
        assert "INT_Apex_Predator_Population" in node.invariants
        assert node.stabilized_tension <= default_assimilation_loop.resonance_threshold

    def test_tension_exceeded_returns_none(
        self, default_assimilation_loop, high_tension_invalid_signal
    ):
        """Tension overflow returns None."""
        node = default_assimilation_loop.assimilate_knowledge(high_tension_invalid_signal)

        assert node is None
        assert "CCNODE_BROKEN_SIGNAL_999" not in default_assimilation_loop.assimilated_cc_nodes


class TestPhaseTopologicalReconstructionEngineIntegration:

    def test_engine_process_successful_assimilation(self, reconstruction_engine, valid_wave_signal):
        """Core Engine: Successful assimilation transition."""
        result = reconstruction_engine.process_external_phenomenon(valid_wave_signal)

        assert result["status"] == "ASSIMILATED_AND_RESONATED"
        assert result["active_cc_nodes_count"] == 1
        assert result["node_details"].node_id == "CCNODE_WAVE_001"

    def test_engine_process_tension_exceeded_restructure_trigger(
        self, reconstruction_engine, high_tension_invalid_signal
    ):
        """Core Engine: Tension overflow trigger."""
        result = reconstruction_engine.process_external_phenomenon(high_tension_invalid_signal)

        assert result["status"] == "TENSION_EXCEEDED_REQUIRES_RESTRUCTURE"
        assert result["active_cc_nodes_count"] == 0
        assert result["node_details"] is None

    def test_engine_ext_sealed_attractor_recovery(
        self, reconstruction_engine_ext, high_tension_invalid_signal
    ):
        """Extended Engine: SealedAttractor quarantine and autonomous recovery."""
        result = reconstruction_engine_ext.process_external_phenomenon(high_tension_invalid_signal)

        assert result["status"] == "RESTRUCTURED_AND_RECOVERED"
        assert result["attractor_id"] == "SEALED_BROKEN_SIGNAL_999"
        assert result["node"].node_id == "CCNODE_BROKEN_SIGNAL_999_RECOVERED"

"""
Tests for Elysia Core Execution Pipeline, Causal Primitives, Self-Evident Naming,
and Meta-Evolutionary Pipeline.
"""

import pytest
from core.lens.causal_primitives import (
    CausalPrimitive,
    CausalOperator,
    CausalWordPrimitive,
    SemanticCouplingEngine
)
from core.lens.topological_terrain_lens import (
    UnknownStimulus,
    SelfEvidentNamingEngine,
    MetaEvolutionLoop,
    TopologicalTerrain3DMapper,
    CognitiveLensSpec,
    InvariantGatekeeper
)
from core.lens.elysia_core_pipeline import ElysiaCorePipeline


class TestCausalPrimitivesAndOperators:
    """Tests Causal Primitives and Dynamic Operator under rigid, fluid, and friction fields."""

    def test_rigid_condition_operator(self):
        op = CausalOperator()
        e1 = CausalPrimitive(boundary_tension=1.0, potential=2.0, invariant_id="One_A")
        e2 = CausalPrimitive(boundary_tension=1.0, potential=2.0, invariant_id="One_B")
        env_rigid = {"fluidity": 0.0, "friction": 0.1, "coalesce_threshold": 1.0}

        result = op.evaluate(e1, e2, env_rigid)
        assert result.B >= 1.0
        assert "Discrete_Sum" in result.I
        assert result.P == pytest.approx(3.9, 0.1)

    def test_fluid_condition_operator(self):
        op = CausalOperator()
        e1 = CausalPrimitive(boundary_tension=0.4, potential=1.5, invariant_id="Drop_1")
        e2 = CausalPrimitive(boundary_tension=0.4, potential=1.5, invariant_id="Drop_2")
        env_fluid = {"fluidity": 0.8, "friction": 0.0, "coalesce_threshold": 1.0}

        result = op.evaluate(e1, e2, env_fluid)
        assert "Coalesced_One" in result.I
        assert result.B == pytest.approx(0.6, 0.1)


class TestSemanticCouplingEngine:
    """Tests Causal Word Primitives and Spandex boundary stretching."""

    def test_literal_valid_coupling(self):
        engine = SemanticCouplingEngine()
        w1 = CausalWordPrimitive("Fire", boundary_tension=2.0, potential=5.0, core_lineage="ThermalReaction")
        w2 = CausalWordPrimitive("Ignite", boundary_tension=1.5, potential=4.5, core_lineage="Combustion")
        env = {"abstract_level": 0.0, "reality_friction": 1.0, "max_allowable_tension": 5.0}

        res = engine.synthesize(w1, w2, env)
        assert res["verdict"] == "VALID_PHENOMENON"
        assert res["verdict_kr"] == "그렇다"
        assert res["stretched_state"] == "LITERAL"

    def test_literal_invalid_conflict(self):
        engine = SemanticCouplingEngine()
        w1 = CausalWordPrimitive("Fire", boundary_tension=2.0, potential=10.0, core_lineage="ThermalReaction")
        w2 = CausalWordPrimitive("Flow", boundary_tension=1.5, potential=1.0, core_lineage="FluidHydrodynamics")
        env = {"abstract_level": 0.0, "reality_friction": 1.0, "max_allowable_tension": 5.0}

        res = engine.synthesize(w1, w2, env)
        assert res["verdict"] == "FALSE_NOT_CAUSAL"
        assert res["verdict_kr"] == "아니다"

    def test_metaphorical_spandex_stretching(self):
        engine = SemanticCouplingEngine()
        w1 = CausalWordPrimitive("Passion", boundary_tension=2.0, potential=8.0, core_lineage="InternalMotivation")
        w2 = CausalWordPrimitive("Fire", boundary_tension=1.5, potential=2.0, core_lineage="ThermalReaction")
        env = {"abstract_level": 0.8, "reality_friction": 1.0, "max_allowable_tension": 5.0}

        res = engine.synthesize(w1, w2, env)
        assert res["verdict"] == "VALID_PHENOMENON"
        assert res["verdict_kr"] == "그렇다"
        assert res["stretched_state"] == "METAPHORIC"


class TestSelfEvidentNamingEngine:
    """Tests discernment ("그렇다" vs "아니다") and self-evident naming for unlabelled stimuli."""

    def test_valid_stimulus_naming(self):
        naming_engine = SelfEvidentNamingEngine()
        stimulus = UnknownStimulus(
            tension_vector=[1.1, 0.4, 0.2],
            potential=5.2,
            raw_friction=1.5
        )
        env = {"noise_threshold": 0.2}

        res = naming_engine.process_stimulus(stimulus, env)
        assert res["verdict"] == "VALID_PHENOMENON"
        assert res["verdict_kr"] == "그렇다"
        assert "GravitationalConvergence" in res["self_given_name"]

    def test_noise_stimulus_rejection(self):
        naming_engine = SelfEvidentNamingEngine()
        stimulus = UnknownStimulus(
            tension_vector=[0.01, 0.01, 0.01],
            potential=0.01,
            raw_friction=0.01
        )
        env = {"noise_threshold": 0.5}

        res = naming_engine.process_stimulus(stimulus, env)
        assert res["verdict"] == "FALSE_NOISE"
        assert res["verdict_kr"] == "아니다"


class TestMetaEvolutionLoopAnd3DTerrain:
    """Tests 3D terrain projection and procedural lens re-ordering."""

    def test_3d_terrain_projection(self):
        friction_field = {
            "raw_friction": 1.2,
            "potential": 3.0,
            "tension_vector": [1.0, 2.0, 2.0]
        }
        lenses = [CognitiveLensSpec("L1", "D1"), CognitiveLensSpec("L2", "D2")]
        terrain = TopologicalTerrain3DMapper.project_to_3d_terrain(friction_field, lenses)

        assert terrain["elevation"] > 3.0
        assert terrain["geodesic_distance"] == pytest.approx(3.0, 0.1)
        assert terrain["information_density"] > 2.0

    def test_meta_evolution_reordering(self):
        loop = MetaEvolutionLoop()
        initial_order = [l.name for l in loop.get_sorted_lenses()]

        concept_event = {
            "verdict": "VALID_PHENOMENON",
            "causal_provenance": {
                "parent_anchor": "ThermalDiffusion",
                "invariant_integrity": 10.0
            }
        }
        loop.accumulate_concept(concept_event)
        new_order = loop.trigger_evolution({})

        assert new_order[0] == "SemanticMetaphorLens"


class TestElysiaCorePipeline:
    """Tests end-to-end processing of ElysiaCorePipeline."""

    def test_pipeline_end_to_end_evolution(self):
        pipeline = ElysiaCorePipeline()

        stimulus = {
            "fluid_potential_diff": 2.5,
            "boundary_tension": 1.2,
            "shear_stress": 0.8,
            "collision_momentum": 0.5
        }
        context = {"noise_threshold": 0.1}

        res = pipeline.process_stimulus(stimulus, context)
        assert res["verdict_kr"] in ["그렇다", "아니다"]
        assert "causal_terrain" in res
        assert "pipeline_state" in res

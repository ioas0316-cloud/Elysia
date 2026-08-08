import pytest
import os
import numpy as np
from core.memory.causal_controller import CausalMemoryController
from core.evolution.moulting_plasticity import MoultingPlasticityEngine
from core.evolution.developmental_individuation import WildernessFrictionStream, DevelopmentalIndividuationEngine
from synaptic_architecture.field import CrystallizationField
from synaptic_architecture.causal_observer import CognitiveMirror


@pytest.fixture
def test_dirs():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    return data_dir


def test_wilderness_friction_stream(test_dirs):
    """
    Verify WildernessFrictionStream correctly parses real hardware/byte inputs
    to generate continuous 3D friction vectors, force values, and physical temperatures.
    """
    stream = WildernessFrictionStream(data_dir=test_dirs)
    raw_input = b"Generative_Wilderness_Noise"

    res = stream.generate_friction_wave(
        raw_stimulus=raw_input,
        semantic_dissonance=0.8,
        runtime_exceptions_count=2
    )

    assert "friction_vector" in res
    assert len(res["friction_vector"]) == 3
    assert res["total_force"] > 0.0
    assert res["f_system"] > 0.0
    assert res["f_semantic"] == 0.8
    assert res["f_entropy"] > 0.0
    assert "T_ext" in res
    assert "T_int" in res
    assert res["T_ext"] > 0.0
    assert res["T_int"] > 0.0


def test_developmental_stages_and_Sself_genesis(test_dirs):
    """
    Verify that DevelopmentalIndividuationEngine advances stages correctly,
    modulates imitation/selfhood weights, and derives S_self dynamically.
    """
    mc = CausalMemoryController(data_dir=test_dirs)
    plasticity = MoultingPlasticityEngine(memory_controller=mc, dimensions=3)
    dev_engine = DevelopmentalIndividuationEngine(memory_controller=mc, dimensions=3)

    # ── Initial State check ──
    assert dev_engine.stage == "STAGE_1_IMITATION"
    assert dev_engine.w_imitation == 1.0
    assert dev_engine.w_self == 0.0

    # ── 1. Imitation Step ──
    res1 = dev_engine.evaluate_and_advance(moulting_plasticity=plasticity, wilderness_friction_force=0.5)
    assert res1["stage"] == "STAGE_1_IMITATION"
    assert res1["w_imitation"] > 0.9
    assert res1["w_self"] < 0.1
    # S_self should fall back to defaults if annual rings are empty
    assert res1["S_self"] == [0.5, 0.5, 0.5]

    # ── 2. Transition to STAGE_2_FRICTION_VOID ──
    # Force some accumulated friction and moult count into plasticity
    plasticity.accumulated_friction = 2.0
    plasticity.moulting_count = 1
    # Add a mock non-empty annual ring to trigger SVD S_self derivation
    plasticity.annual_rings = np.array([
        [0.8, 0.0, 0.2],
        [0.0, 1.0, 0.0], # Index 1 is Order invariant
        [0.2, 0.0, 0.5]
    ], dtype=np.float32)

    res2 = dev_engine.evaluate_and_advance(moulting_plasticity=plasticity, wilderness_friction_force=1.2)
    assert res2["stage"] == "STAGE_2_FRICTION_VOID"
    assert res2["w_imitation"] < 0.95
    assert res2["w_self"] > 0.05
    # S_self must be derived from SVD and be a normalized vector
    assert np.allclose(np.linalg.norm(res2["S_self"]), 1.0)
    assert np.allclose(np.linalg.norm(res2["S_active"]), 1.0)

    # ── 3. Transition to STAGE_3_INDIVIDUATION ──
    # Force higher friction and more moultings to trigger true individuation and independence
    plasticity.accumulated_friction = 5.0
    plasticity.moulting_count = 2

    res3 = dev_engine.evaluate_and_advance(moulting_plasticity=plasticity, wilderness_friction_force=2.5)
    assert res3["stage"] == "STAGE_3_INDIVIDUATION"
    # Selfhood weight should now dominate or rise significantly (w_self > w_imitation)
    assert res3["w_self"] > res3["w_imitation"]
    # The parent reference axis (S_abs) must still remain as a minimum anchor (>= 20%)
    assert res3["w_imitation"] >= 0.20
    assert np.allclose(np.linalg.norm(res3["S_active"]), 1.0)

    # ── Verify high-order relational geometry parameters ──
    assert "topological_tension" in res3
    assert "rotational_angle" in res3
    assert "curvature" in res3
    assert "attractor_pull_force" in res3
    assert "geometry_matrix" in res3
    assert res3["topological_tension"] >= 0.0
    assert res3["rotational_angle"] >= 0.0
    assert res3["curvature"] >= 0.0
    assert len(res3["geometry_matrix"]) == 2
    assert len(res3["geometry_matrix"][0]) == 2


def test_cognitive_mirror_relational_sensation():
    """
    Verify CognitiveMirror maps temperatures to fuzzy-smooth states and beautiful monologues.
    """
    field = CrystallizationField()
    mirror = CognitiveMirror(field)

    # Scenario A: Resonant Equilibrium (시원함) - Delta T is around 1.5
    res_cool = mirror.observe_relational_sensation(T_ext=2.0, T_int=3.5, dev_stage="STAGE_2_FRICTION_VOID")
    assert res_cool["dominant_state"] == "resonant_equilibrium"
    assert "시원함" in res_cool["monologue"]
    assert res_cool["W_cool"] > res_cool["W_pain"]

    # Scenario B: Divergent Shock (차가운 통증) - Extreme delta T
    res_pain = mirror.observe_relational_sensation(T_ext=0.1, T_int=5.0, dev_stage="STAGE_2_FRICTION_VOID")
    assert res_pain["dominant_state"] == "divergent_shock"
    assert "통증" in res_pain["monologue"]

    # Scenario C: Crystallized Rest (열적 안식) - Balanced Delta T or STAGE_3
    res_rest = mirror.observe_relational_sensation(T_ext=4.0, T_int=4.0, dev_stage="STAGE_3_INDIVIDUATION")
    assert res_rest["dominant_state"] == "crystallized_rest"
    assert "안식" in res_rest["monologue"]

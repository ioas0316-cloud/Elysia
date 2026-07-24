import numpy as np
import pytest
from core.physics.quantum_stat_field import QuantumStatField
from synaptic_architecture.cognitive_engine import ElysiaCognitiveEngine

def test_physical_axis_crystallization():
    qsf = QuantumStatField({
        "health": 12.0,
        "force": 12.0,
        "mind": 12.0,
        "speed": 12.0,
        "intelligence": 12.0
    })

    # Step to develop unique stabilized positions
    for _ in range(10):
        qsf.step(dt=0.1)

    # Crystallize this dynamic equilibrium
    original_positions = {name: node.position.copy() for name, node in qsf.nodes.items()}
    qsf.crystallize_axis("standard_balance")

    # Change the base stats to something different
    qsf.update_base_stats({
        "health": 30.0,
        "force": 5.0,
        "mind": 5.0,
        "speed": 5.0,
        "intelligence": 5.0
    })
    qsf.step(dt=0.1)

    # Assert positions have shifted away from original balance
    shifted = False
    for name, node in qsf.nodes.items():
        if not np.allclose(node.position, original_positions[name]):
            shifted = True
            break
    assert shifted

    # Switch back to the crystallized base stats
    qsf.update_base_stats({
        "health": 12.0,
        "force": 12.0,
        "mind": 12.0,
        "speed": 12.0,
        "intelligence": 12.0
    })

    # Running a step should instantly trigger Crystallized Axis bypass and snap positions back in O(1)
    qsf.step(dt=0.1)
    assert qsf.active_axis == "standard_balance"
    for name, node in qsf.nodes.items():
        assert np.allclose(node.position, original_positions[name])

def test_cognitive_thought_crystallization_bypass():
    engine = ElysiaCognitiveEngine(resolution=128)

    dna_a = engine.build_fractal_dna("Light_Truth", np.uint64(0xFEEDFACEFEEDFACE))
    dna_b = engine.build_fractal_dna("Dark_Entropy", np.uint64(0xBADCAFEBADCAFE00))

    stimulus = np.uint64(0xFEEDFACE00000000)

    # First active WFC collapse computation
    result = engine.solve_wfc_collapse(stimulus, [dna_a, dna_b])
    assert result["resonance_score"] > 0.0

    # Crystallize the solved solution (e.g., "1+1=2")
    engine.crystallize_thought(stimulus, result)

    # Modify candidate DNAs so if active computation were run, it would fail or return different scores
    # However, since it is crystallized, solve_wfc_collapse will bypass entirely and return the exact cached result
    bypass_result = engine.solve_wfc_collapse(stimulus, [])
    assert bypass_result == result

    # Ensure bypass was recorded in meta reflection history
    reflections = engine.get_meta_reflection()
    actions = [r["action"] for r in reflections]
    assert "THOUGHT_CRYSTALLIZATION" in actions
    assert "CRYSTALLIZED_BYPASS" in actions

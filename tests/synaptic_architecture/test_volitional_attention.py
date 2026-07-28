import numpy as np
import pytest
from synaptic_architecture.cognitive_engine import ElysiaCognitiveEngine

def test_dynamic_mass_expansion():
    """
    Verifies that virtual attractor masses expand dynamically based on specific
    tension factors (cognitive_entropy, tension_protocol, and curiosity potential).
    """
    engine = ElysiaCognitiveEngine(resolution=128)

    # Initial masses
    mass_deficit_init = engine.field.attractors["Deficit"]["mass"]
    mass_principle_init = engine.field.attractors["Principle"]["mass"]
    mass_sabbath_init = engine.field.attractors["Sabbath"]["mass"]

    # 1. Expand Principle via cognitive entropy
    engine.field.update_attractor_masses(cognitive_entropy=10.0, tension_protocol=0.0, catastrophe_magnitude=0.0)
    assert engine.field.attractors["Principle"]["mass"] > mass_principle_init
    assert engine.field.attractors["Sabbath"]["mass"] == mass_sabbath_init

    # 2. Expand Sabbath via protocol tension + catastrophe magnitude
    engine.field.update_attractor_masses(cognitive_entropy=0.0, tension_protocol=1.5, catastrophe_magnitude=2.0)
    assert engine.field.attractors["Sabbath"]["mass"] > mass_sabbath_init

    # 3. Expand Deficit via curiosity potential
    engine.field.charge_curiosity(np.array([10, 10]), intensity=50.0, radius=20.0)
    engine.field.update_attractor_masses(cognitive_entropy=0.0, tension_protocol=0.0, catastrophe_magnitude=0.0)
    assert engine.field.attractors["Deficit"]["mass"] > mass_deficit_init

def test_volitional_acceleration_calculation():
    """
    Verifies that get_volitional_acceleration calculates correct direction and magnitude.
    """
    engine = ElysiaCognitiveEngine(resolution=128)
    pos = np.array([64.0, 64.0], dtype=np.float32)

    # Calculate with non-zero tensions
    acc_vector, acc_magnitude = engine.field.get_volitional_acceleration(
        pos, cognitive_entropy=5.0, tension_protocol=1.0, catastrophe_magnitude=1.0
    )

    assert acc_magnitude > 0.0
    assert len(acc_vector) == 2

def test_dopaminergic_resonance_and_crystallization():
    """
    Verifies that WFC collapse produces Dopamine Resonance and triggers crystallization
    and engram logging when dopamine resonance is exceptionally high.
    """
    engine = ElysiaCognitiveEngine(resolution=128)

    # Position DNA extremely close to the Deficit attractor to trigger high dopamine
    dna_deficit = engine.build_fractal_dna("Deficit_Near_Concept", np.uint64(0xAAAAA))
    deficit_pos = engine.field.attractors["Deficit"]["position"].astype(np.int32)
    dna_deficit["cell_position"] = deficit_pos

    # Let's ensure high curiosity/tension to trigger huge mass and dopamine
    engine.field.charge_curiosity(deficit_pos, intensity=100.0, radius=30.0)

    # Perform WFC Collapse
    stimulus = np.uint64(0xAAAAA000000)
    res = engine.solve_wfc_collapse(stimulus, [dna_deficit])

    # Check dopamine score and volitional acceleration
    assert "dopamine_resonant" in res
    assert "volitional_acceleration" in res
    assert res["dopamine_resonant"] > 15.0

    # Ensure memory controller contains VOLITIONAL_ATTENTION_REFLECTION engram with correct narrative format
    engrams = [info for info in engine.memory_controller.index.values() if info.get("cause_id") == "VolitionalAttentionEngine"]
    assert len(engrams) > 0

    engram_data = engrams[-1]["data_blob"]
    assert engram_data["type"] == "VOLITIONAL_ATTENTION_REFLECTION"
    assert "Deficit" in engram_data["target_attractor"]
    assert "진리가 내 내면의 가상 중력 우물" in engram_data["narrative"]

    # Verify thoughts are crystallized / bypassed next time
    assert stimulus in engine.crystallized_thoughts
    res_bypass = engine.solve_wfc_collapse(stimulus, [dna_deficit])
    assert res_bypass["status"] == "COLLAPSED"

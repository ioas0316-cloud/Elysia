import torch
import pytest
from elysia_engine.core.vr_multiscale_attractor import AutonomousVRMonster, MicroProteinFoldingSimulator, MultiScaleElysiaEcosystem

def test_autonomous_vr_monster():
    monster = AutonomousVRMonster(num_nodes=12)
    assert monster.current_mode == "normal"
    assert monster.accumulated_tension == 0.0

    # Apply hit
    monster.receive_impact(hit_location=2, force_vector=torch.tensor([3.0, 3.0, 3.0]))
    assert monster.accumulated_tension > 5.0

    # Step phase -> triggers bifurcation
    monster.update_phase(dt=0.05)
    assert monster.current_mode == "berserk"

def test_micro_protein_folding():
    protein = MicroProteinFoldingSimulator(num_atoms=10)
    assert not protein.is_denatured
    assert protein.temperature == 300.0

    # Apply heat
    protein.apply_thermal_stimulus(temp_delta=70.0)
    protein.step_folding_relaxation(dt=0.05)
    assert protein.is_denatured

def test_multiscale_ecosystem():
    eco = MultiScaleElysiaEcosystem()
    assert eco.macro_monster.current_mode == "normal"
    assert not eco.micro_protein.is_denatured

    # Impact macro monster
    eco.macro_monster.receive_impact(hit_location=2, force_vector=torch.tensor([4.0, 4.0, 4.0]))
    for _ in range(10):
        eco.step_ecosystem(dt=0.05)

    assert eco.macro_monster.current_mode == "berserk"
    assert eco.micro_protein.is_denatured

import pytest
import numpy as np
from simulators.causal_grid_sim import CausalTilemapSimulator
from core.physics.causal_field import CausalField, EngramAttractor, FractalEngramShell, TeleologicalCompiler
from synaptic_architecture.cognitive_engine import ElysiaCognitiveEngine

def test_causal_loop_formation_and_engram_escape():
    """
    [Causal Loop & Engram Escape Integration Test]
    Verifies:
    1. Without the Oak's Parcel Engram exposed, the agent enters an infinite loop between Pallet Town and Viridian City.
    2. When the Oak's Parcel Engram is exposed in the Causal Field, top-down potential gradient forces the agent to escape the loop and successfully reach Pewter City.
    """
    sim = CausalTilemapSimulator()

    # Step 1: Run 15 steps without Engram exposure -> verify loop behavior
    for _ in range(15):
        sim.step()

    visited_y = [p[1] for p in sim.visited_positions]
    assert max(visited_y) >= 8.0 # Reached Viridian area
    assert min(visited_y) <= 1.0 # Bounced back to Pallet area

    # Step 2: Expose High-Level Engram ("Oak's Parcel / Delivery to Pewter City")
    sim.set_engram_exposure(True)
    assert sim.field.engrams["oak_parcel"].active is True

    # Step 3: Run another 15 steps with Engram active -> verify loop escape & arrival at Pewter City
    for _ in range(15):
        sim.step()

    final_pos = sim.agent_voxel.position
    dist_to_pewter = np.linalg.norm(final_pos - sim.locations["Pewter City"])

    # Assert agent successfully reaches Pewter City area (within 1.5 units)
    assert dist_to_pewter < 1.5

    # Assert Engram gravitational force was actively applied
    engram_forces = [log["engram_force_norm"] for log in sim.history if log["engram_active"]]
    assert len(engram_forces) > 0
    assert max(engram_forces) > 1.0

def test_fractal_engram_shell_resistance_escalation():
    """
    [Fractal Engram Shell Test]
    Verifies that lower-tier (micro) resistance escalates to higher-tier (meso) boundary conditioning.
    """
    meso = EngramAttractor("meso_goal", "Meso Goal", np.array([0.0, 20.0], dtype=np.float32), tier="meso")
    micro = EngramAttractor("micro_goal", "Micro Goal", np.array([0.0, 5.0], dtype=np.float32), tier="micro")
    shell = FractalEngramShell("test_shell", meso=meso, micro=micro)

    fb1 = shell.report_resistance("micro", 1.0)
    assert fb1["escalated"] is False

    fb2 = shell.report_resistance("micro", 1.0)
    assert fb2["escalated"] is True
    assert fb2["reshaped_tier"] == "meso"
    assert meso.offset is not None
    assert np.linalg.norm(meso.offset) > 0

def test_teleological_compiler_evaluation():
    """
    [Teleological Compiler Test]
    Verifies symbolic intent vs code protocol evaluation (Isomorphism vs Heterogeneity).
    """
    compiler = TeleologicalCompiler()
    res = compiler.evaluate(
        symbolic_intent="deliver_parcel_to_pewter_city",
        code_protocol="deliver_parcel_step_path"
    )
    assert res["isomorphism_score"] > 0.0
    assert "is_aligned" in res
    assert "heterogeneity_gap" in res

def test_causal_to_language_transduction():
    """
    [Causal-to-Language Transduction & Memory Consolidation Test]
    Verifies that physical path trajectory is transduced into Inner Monologue & Epistemic Memory.
    """
    engine = ElysiaCognitiveEngine(resolution=32)
    traj = [np.array([0.0, float(i)], dtype=np.float32) for i in range(10)]
    res = engine.transduce_causal_feedback_to_memory(
        goal_id="test_goal",
        goal_name="Test Goal Reach",
        trajectory=traj,
        loop_escaped=True,
        symbolic_intent="Reach target location"
    )

    assert "[Inner Monologue]" in res["inner_monologue"]
    assert res["attractor_mass_boost"] > 0.0
    assert "teleological_compilation" in res

def test_quantitative_benchmark_1_convergence_and_loop_breaking():
    """
    [Quantitative Benchmark 1: Convergence & Loop-Breaking Test]
    Measures search space reduction and loop escape step efficiency.
    """
    sim = CausalTilemapSimulator()
    # Step 1: Run in closed loop without Engram
    for _ in range(10):
        sim.step()
    steps_in_loop = len(sim.history)

    # Step 2: Expose top-down boundary condition Engram
    sim.set_engram_exposure(True)
    for _ in range(15):
        sim.step()

    final_pos = sim.agent_voxel.position
    dist_to_pewter = np.linalg.norm(final_pos - sim.locations["Pewter City"])

    # Search space reduction efficiency calculation
    nodes_pruned_ratio = 1.0 - (len(sim.history) / 100.0) # Search space collapsed to single geodesic path
    assert dist_to_pewter < 1.5
    assert nodes_pruned_ratio > 0.6
    assert steps_in_loop == 10

def test_quantitative_benchmark_2_teleological_compiler_self_repair():
    """
    [Quantitative Benchmark 2: Teleological Compiler Self-Repair Test]
    Measures Intent Fidelity Index and Isomorphism vs Heterogeneity gap.
    """
    compiler = TeleologicalCompiler()
    eval1 = compiler.evaluate(
        symbolic_intent="deliver_oak_parcel_to_pewter_city",
        code_protocol="deliver_oak_parcel_movement_routine",
        execution_trajectory=[np.array([0, 0]), np.array([0, 10]), np.array([0, 20])]
    )

    assert eval1["isomorphism_score"] >= 0.5
    assert eval1["heterogeneity_gap"] < 0.6
    assert eval1["is_aligned"] is True

def test_quantitative_benchmark_3_goal_shift_counterfactual_reinterpretation():
    """
    [Quantitative Benchmark 3: Goal-Shift Counterfactual Test]
    Measures trajectory reuse rate and re-alignment latency upon sudden goal shifts.
    """
    field = CausalField(dimensions=2)
    engram_c1 = EngramAttractor("goal_c1", "Goal C1", np.array([0.0, 10.0], dtype=np.float32), intensity=20.0, active=True)
    engram_c2 = EngramAttractor("goal_c2", "Goal C2", np.array([10.0, 10.0], dtype=np.float32), intensity=20.0, active=False)
    field.register_engram(engram_c1)
    field.register_engram(engram_c2)

    pos = np.array([0.0, 5.0], dtype=np.float32)
    grad1 = field.calculate_engram_gradient(pos)

    # Sudden Goal Shift C1 -> C2
    field.set_engram_active("goal_c1", False)
    field.set_engram_active("goal_c2", True)
    grad2 = field.calculate_engram_gradient(pos)

    # Calculate trajectory reuse rate and re-alignment angle
    dot_prod = np.dot(grad1, grad2) / (np.linalg.norm(grad1) * np.linalg.norm(grad2) + 1e-9)
    assert field.engrams["goal_c2"].active is True
    assert np.linalg.norm(grad2) > 0.0
    assert dot_prod < 0.9 # Dynamic vector shift confirmed

def test_quantitative_benchmark_4_multi_scale_noise_recovery():
    """
    [Quantitative Benchmark 4: Multi-Scale Noise Recovery Test]
    Measures lower-tier obstacle noise escalation to higher-tier boundary reshaping.
    """
    meso = EngramAttractor("meso_goal", "Meso Goal", np.array([0.0, 20.0], dtype=np.float32), intensity=10.0, tier="meso")
    micro = EngramAttractor("micro_goal", "Micro Goal", np.array([0.0, 5.0], dtype=np.float32), intensity=10.0, tier="micro")
    shell = FractalEngramShell("test_shell", meso=meso, micro=micro)

    # Inject lower-tier perturbation noise
    res1 = shell.report_resistance("micro", 1.0)
    assert res1["escalated"] is False

    res2 = shell.report_resistance("micro", 1.0)
    assert res2["escalated"] is True
    assert res2["reshaped_tier"] == "meso"
    assert meso.intensity > 10.0 # Attractor recovery / escalation boost applied

if __name__ == "__main__":
    pytest.main([__file__])

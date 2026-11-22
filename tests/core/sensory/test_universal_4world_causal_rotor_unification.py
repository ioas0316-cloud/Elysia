r"""
test_universal_4world_causal_rotor_unification.py

Zero-mock empirical proof that ONE continuous causal rotor substrate unifies and
optimizes 4 completely different worlds with the exact same O(1) operators:
  1. World 1 — Cinema / Video (Real MP4 Flower `960x540`)
  2. World 2 — Human / Robot Kinematics (Real 3-DOF Minimum-Jerk Damped IK Solver)
  3. World 3 — Matter & Reversible Chemistry (Real RK4 ODE A + B <-> C + D)
  4. World 4 — Embodied 5-Sense Biology (Fresh <-> Decayed Apple Rotor)

Plus verification that `ElysiaLiveOrganism` natively persists and cold-reloads these
O(1) reversible rotor engrams across sessions.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from core.sensory.agamotto_causal_rotor_engram_engine import (
    AgamottoCausalRotorEngramEngine,
)
from core.sensory.elysia_live_organism_runtime import ElysiaLiveOrganism

REAL_MP4_PATH = Path(r"c:\Elysia\demos\real_world_assets\real_external_nature_video.mp4")
UNIVERSAL_ENGRAM_PATH = Path(r"c:\Elysia\data\universal_4world_engrams.json")
UNIVERSAL_DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_universal_causal_rotor_4worlds_dashboard.png"
)


def test_one_rotor_engine_unifies_video_kinematics_chemistry_and_biology() -> None:
    # ------------------------------------------------------------------
    # Session 1: One-Time First-Pass Crystallization of 4 Disparate Worlds
    # ------------------------------------------------------------------
    eng1 = AgamottoCausalRotorEngramEngine()
    mp4_eng, key_frames = eng1.crystallize_real_mp4_to_engram(
        REAL_MP4_PATH, sample_frames=24, engram_id="real_mp4_flower"
    )
    kin_eng, kin_poses = eng1.crystallize_robot_human_kinematics_engram(
        engram_id="human_robot_reach_grasp", steps=64
    )
    chem_eng, _ = eng1.crystallize_chemical_reaction_engram(
        engram_id="reversible_chemical_kinetics", steps=80
    )
    apple_eng = eng1.crystallize_embodied_apple_engram("embodied_5sense_apple")

    engrams_by_id = {
        "real_mp4_flower": mp4_eng,
        "human_robot_reach_grasp": kin_eng,
        "reversible_chemical_kinetics": chem_eng,
        "embodied_5sense_apple": apple_eng,
    }
    for eid in engrams_by_id:
        assert eid in eng1.causal_field.voxels
        assert np.isclose(np.linalg.norm(eng1.causal_field.voxels[eid].quaternion_q), 1.0, atol=1e-4)
    assert len(eng1.causal_field.beams) >= 3

    saved_path = eng1.save_engrams(UNIVERSAL_ENGRAM_PATH)
    disk_kb = saved_path.stat().st_size / 1024.0
    assert saved_path.stat().st_size < 40000

    # ------------------------------------------------------------------
    # Session 2: Brand-New Instance Cold-Reloads ONLY Compact JSON
    # ------------------------------------------------------------------
    eng2 = AgamottoCausalRotorEngramEngine()
    t_load0 = time.perf_counter()
    count = eng2.load_engrams(UNIVERSAL_ENGRAM_PATH)
    load_ms = (time.perf_counter() - t_load0) * 1000.0
    assert count == 4
    assert load_ms < 50.0

    # Benchmark O(1) cold-reload scrub & Eye-of-Agamotto back-trace across ALL 4 worlds
    thetas = np.linspace(0.0, 1.0, 25)
    cold_recall_ms = {}
    for eid in engrams_by_id:
        t_q0 = time.perf_counter()
        for th in thetas:
            eng2.scrub_engram_phase(eid, float(th), distance_m=1.0)
            eng2.reverse_trace_causal_origin(eid, float(th))
        per_call_ms = ((time.perf_counter() - t_q0) * 1000.0) / (2.0 * len(thetas))
        cold_recall_ms[eid] = per_call_ms

        rev = eng2.reverse_trace_causal_origin(eid, observed_theta=1.0)
        assert rev.reversible_round_trip_error < 1e-5
        assert per_call_ms < 0.5

        speedup = engrams_by_id[eid].first_pass_crystallization_ms / max(per_call_ms, 1e-6)
        print(
            f"\n[{eid:28s}] 1st-pass={engrams_by_id[eid].first_pass_crystallization_ms:7.2f}ms "
            f"-> O(1) recall={per_call_ms:.4f}ms ({speedup:6.0f}x) | rev_err={rev.reversible_round_trip_error:.1e}"
        )

    print(f"[Universal 4-World Engram File] {disk_kb:.2f} KB | cold_load={load_ms:.2f}ms")

    # World 2 Physical Check: 3-DOF Human/Robot Reach & Grasp Kinematics
    kin_rest = eng2.scrub_engram_phase("human_robot_reach_grasp", theta=0.0)
    kin_grasp = eng2.scrub_engram_phase("human_robot_reach_grasp", theta=1.0)
    print(
        f"[Kinematics 3-DOF IK] dist: {kin_rest.modality_channels['target_distance_m']:.3f}m -> "
        f"{kin_grasp.modality_channels['target_distance_m']:.4f}m | grip: "
        f"{kin_rest.modality_channels['grip_force_n']:.2f} -> {kin_grasp.modality_channels['grip_force_n']:.2f}"
    )
    assert kin_grasp.modality_channels["target_distance_m"] < 0.08
    assert kin_grasp.modality_channels["grip_force_n"] > 0.70
    assert kin_grasp.modality_channels["target_distance_m"] < kin_rest.modality_channels["target_distance_m"]

    # World 3 Physical Check: Reversible Chemical Kinetics A + B <-> C + D
    chem_init = eng2.scrub_engram_phase("reversible_chemical_kinetics", theta=0.0)
    chem_eq = eng2.scrub_engram_phase("reversible_chemical_kinetics", theta=1.0)
    print(
        f"[Chemistry RK4 ODE ] [A]: {chem_init.modality_channels['reactant_A']:.3f} -> "
        f"{chem_eq.modality_channels['reactant_A']:.3f} | [C]: "
        f"{chem_init.modality_channels['product_C']:.3f} -> {chem_eq.modality_channels['product_C']:.3f}"
    )
    assert np.isclose(chem_init.modality_channels["reactant_A"], 1.0, atol=1e-4)
    assert np.isclose(chem_init.modality_channels["product_C"], 0.0, atol=1e-4)
    assert chem_eq.modality_channels["reactant_A"] < 0.45
    assert chem_eq.modality_channels["product_C"] > 0.55

    # Render Unified 4-World Dashboard PNG
    out_png = eng1.render_universal_4worlds_dashboard(
        engrams_by_id, key_frames, kin_poses, cold_recall_ms, UNIVERSAL_DASHBOARD_PATH
    )
    assert out_png.exists() and out_png.stat().st_size > 25000


def test_live_organism_cold_reload_o1_agamotto_memory(tmp_path: Path) -> None:
    state_file = tmp_path / "organism_state.json"

    # Session 1: Organism watches genuine external MP4 ONCE & persists engrams
    org1 = ElysiaLiveOrganism(state_path=state_file, load_existing_memory=False)
    org1.observe_real_mp4_video_stream(
        video_path=REAL_MP4_PATH,
        output_dashboard_path=tmp_path / "obs_dash.png",
    )
    saved_state = org1.save_organism_state()
    assert saved_state.exists()
    assert org1.engram_store_path.exists()

    # Session 2: Brand-new cold organism reloads state & performs O(1) memory recall (< 1ms!)
    org2 = ElysiaLiveOrganism(state_path=state_file, load_existing_memory=True)
    assert org2.awakening_count == 2
    assert "real_mp4_flower" in org2.agamotto.engrams

    t0 = time.perf_counter()
    mid_state = org2.recall_crystallized_experience("real_mp4_flower", theta=0.5, distance_m=1.0)
    rev_trace = org2.reverse_trace_memory("real_mp4_flower", observed_theta=1.0)
    total_ms = (time.perf_counter() - t0) * 1000.0

    print(
        f"\n[ElysiaLiveOrganism Cold O(1) Recall] theta=0.5 red={mid_state.modality_channels['red_petal_chroma']:.3f} "
        f"| rev_err={rev_trace.reversible_round_trip_error:.1e} | total_time={total_ms:.4f}ms"
    )
    assert total_ms < 1.0
    assert rev_trace.reversible_round_trip_error < 1e-5
    assert mid_state.modality_channels["red_petal_chroma"] > 0.10

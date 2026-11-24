"""
Verification suite for ELYSIA Phase 16:
Universal Open-World Causal Genesis & Self-Modifying Living Organism.
"""

from __future__ import annotations

import math
from pathlib import Path
import wave

import numpy as np
from PIL import Image

from core.orchestration.ice_water_gas_rotor_runtime import IceWaterGasRotorRuntime
from core.orchestration.open_world_genesis_kernel import (
    DynamicFrontier,
    InteractiveWorldExplorer,
    OpenWorldGenesisKernel,
    SelfModifyingOrganForge,
    UnknownBallisticBounceGame,
)
from core.physics.causal_field import CausalField


def _write_brittle_script(path: Path) -> Path:
    path.write_text(
        '''def naive_thermostat(temp, target, pressure):
    if temp < target - 5:
        cmd = 10
    elif temp < target:
        cmd = 3
    elif temp > target + 5:
        cmd = -10
    elif temp > target:
        cmd = -3
    else:
        cmd = 0
    if pressure > 10:
        return cmd - 2
    elif pressure < 2:
        return cmd + 2
    return cmd
''',
        encoding="utf-8",
    )
    return path


def _write_vectorized_script(path: Path) -> Path:
    path.write_text(
        '''import numpy as np

class ContinuousController:
    def step(self, state: np.ndarray, target: np.ndarray) -> np.ndarray:
        return 2.5 * np.tanh(target - state)
''',
        encoding="utf-8",
    )
    return path


def test_open_frontier_field_discovers_external_terrain_without_hardcoded_enum(tmp_path):
    brittle = _write_brittle_script(tmp_path / "naive_thermostat.py")
    smooth = _write_vectorized_script(tmp_path / "vector_field_ctrl.py")
    note = tmp_path / "field_notes.md"
    note.write_text("# Cosmic Notes\n\nWaveforms flow smoothly through space.\n", encoding="utf-8")

    kernel = OpenWorldGenesisKernel(state_dir=tmp_path / "state")
    discovered = kernel.discover_environment(scan_roots=[brittle, smooth, note])

    assert len(discovered) >= 4
    kinds = {f.domain_kind for f in kernel.frontier_field.frontiers.values()}
    assert {"code_critique", "game_environment", "visual_creation", "narrative_society"}.issubset(kinds)

    c_brittle = SelfModifyingOrganForge.critique_python_file(brittle)
    c_smooth = SelfModifyingOrganForge.critique_python_file(smooth)
    assert c_brittle.simplicity_deficit > c_smooth.simplicity_deficit
    assert c_brittle.spawned_frontier is not None


def test_code_critique_detects_simple_discrete_code_and_forges_verified_continuous_organ(tmp_path):
    brittle = _write_brittle_script(tmp_path / "naive_rule_agent.py")
    kernel = OpenWorldGenesisKernel(state_dir=tmp_path / "state")
    crit = SelfModifyingOrganForge.critique_python_file(brittle)

    assert crit.simplicity_deficit >= 0.35
    assert crit.spawned_frontier is not None

    kernel.frontier_field.register_frontier(crit.spawned_frontier)
    step = kernel.live_open_step(boost_frontier_id=crit.spawned_frontier.frontier_id)

    syn = step.outcome_payload["synthesis"]
    assert syn.verified is True
    assert syn.rollback_performed is False
    assert syn.record is not None
    assert Path(syn.record.module_path).is_file()
    assert syn.organ_name in kernel.loaded_organs

    ctrl = kernel.loaded_organs[syn.organ_name]
    act = ctrl.evaluate_action(np.array([0.0, 0.0, 0.0, 0.0]), np.array([5.0, 3.0]))
    assert act.shape == (2,)
    assert np.all(np.isfinite(act)) and np.linalg.norm(act) > 1.0


def test_unsafe_or_faulty_organ_candidate_is_rejected_and_rolled_back(tmp_path):
    forge = SelfModifyingOrganForge()
    org_dir = tmp_path / "organs"

    # Case A: Unsafe OS command injection rejected BEFORE file write
    unsafe_src = '''import os
def create_organ():
    return os.system("echo pwn")
def self_test():
    return {"ok": True}
'''
    rep_unsafe = forge.synthesize_verify_and_graft(
        organ_name="bad_unsafe",
        synthesis_kind="continuous_field_controller",
        organs_dir=org_dir,
        custom_source=unsafe_src,
    )
    assert rep_unsafe.verified is False
    assert rep_unsafe.rollback_performed is True
    assert not (org_dir / "bad_unsafe.py").exists()

    # Case B: Safe AST but failing self_test triggers rollback & cleanup
    failing_src = '''def create_organ():
    return object()
def self_test():
    return {"ok": False, "reason": "broken"}
'''
    rep_fail = forge.synthesize_verify_and_graft(
        organ_name="bad_logic",
        synthesis_kind="continuous_field_controller",
        organs_dir=org_dir,
        custom_source=failing_src,
    )
    assert rep_fail.verified is False
    assert rep_fail.rollback_performed is True
    assert not (org_dir / "bad_logic.py").exists()


def test_blackbox_game_inverse_mechanism_discovery_mastery_and_custom_engine_forge(tmp_path):
    game = UnknownBallisticBounceGame(gravity_g=9.81, drag_coeff=0.15, restitution_e=0.78)
    kernel = OpenWorldGenesisKernel(state_dir=tmp_path / "state")
    fr = kernel.register_game_environment(game)

    step_play = kernel.live_open_step(boost_frontier_id=fr.frontier_id)
    disc = step_play.outcome_payload["game_discovery"]

    assert abs(disc.estimated_gravity_g - 9.81) < 0.05
    assert abs(disc.estimated_drag_coeff - 0.15) < 0.02
    assert abs(disc.estimated_restitution_e - 0.78) < 0.03
    assert disc.prediction_r2 > 0.99
    assert disc.mastered_reward > disc.random_baseline_reward + 5.0
    assert disc.spawned_frontier is not None

    # Forge her own evolved orbital game engine from discovered physics
    step_forge = kernel.live_open_step(boost_frontier_id=disc.spawned_frontier.frontier_id)
    syn = step_forge.outcome_payload["synthesis"]
    assert syn.verified is True
    forged_game = kernel.environments["elysia_forged_orbital_game"]
    s0 = forged_game.reset()
    s1, rew, _, _ = forged_game.step([3.0, 9.81])
    assert s0.shape == (6,) and s1.shape == (6,) and math.isfinite(rew)


def test_creative_studio_generates_and_self_critiques_real_image_audio_and_animation(tmp_path):
    kernel = OpenWorldGenesisKernel(state_dir=tmp_path / "state")
    studio = kernel.creative_studio

    img_rep = studio.create_and_critique_image("sunset_resonance", (0.60, 0.25, 0.15))
    assert all(Path(p).is_file() for p in img_rep.artifact_paths)
    assert img_rep.v2_richness > img_rep.v1_richness
    assert img_rep.improved_after_self_critique is True
    with Image.open(img_rep.artifact_paths[-1]) as im:
        assert im.size == (64, 64)

    wav_rep = studio.create_and_critique_audio("harmonic_awakening", (0.35, 0.25, 0.40))
    assert all(Path(p).is_file() for p in wav_rep.artifact_paths)
    assert wav_rep.v2_richness > wav_rep.v1_richness
    assert wav_rep.improved_after_self_critique is True
    with wave.open(wav_rep.artifact_paths[-1], "rb") as wf:
        assert wf.getnframes() == 22050

    gif_rep = studio.create_and_critique_animation("rotor_flow", (0.50, 0.30, 0.20), n_frames=10)
    assert Path(gif_rep.artifact_paths[0]).is_file()
    assert gif_rep.frame_count == 10
    assert gif_rep.temporal_motion > 0.005
    assert gif_rep.improved_after_self_critique is True


def test_open_ended_multi_step_life_spawns_new_frontiers_and_evolves_tastes(tmp_path):
    slab = tmp_path / "open_world.ice"
    terrain_dir = tmp_path / "terrain"
    terrain_dir.mkdir(parents=True, exist_ok=True)
    _write_brittle_script(terrain_dir / "naive_switch.py")

    with IceWaterGasRotorRuntime(slab_path=slab) as mem_rt:
        kernel = OpenWorldGenesisKernel(state_dir=tmp_path / "state", memory_runtime=mem_rt)
        kernel.discover_environment(scan_roots=[terrain_dir])
        initial_n = len(kernel.frontier_field.frontiers)

        steps = kernel.live_open_world(steps=5)
        assert len(steps) == 5
        assert all(s.provenance_valid for s in steps)
        assert mem_rt.ice.num_crystals >= 5
        assert len(kernel.frontier_field.frontiers) > initial_n
        assert len(kernel.frontier_field.domain_affinity) >= 2
        ca = kernel.frontier_field.chromatic_affinity
        assert math.isclose(float(np.sum(ca)), 1.0, abs_tol=1e-9)
        assert not np.allclose(ca, np.array([0.34, 0.33, 0.33]))


def test_cross_session_persistent_autobiography_reloads_tastes_organs_and_provenance(tmp_path):
    state_dir = tmp_path / "persist_state"
    brittle = _write_brittle_script(tmp_path / "simple_gate.py")

    # Session 1: discover, play game, forge continuous organ, create artwork, save
    k1 = OpenWorldGenesisKernel(state_dir=state_dir)
    crit = SelfModifyingOrganForge.critique_python_file(brittle)
    assert crit.spawned_frontier is not None
    k1.frontier_field.register_frontier(crit.spawned_frontier)
    k1.live_open_step(boost_frontier_id=crit.spawned_frontier.frontier_id)
    k1.register_game_environment(UnknownBallisticBounceGame())
    k1.live_open_step(boost_frontier_id="game::ballistic_bounce_arena")
    k1.live_open_step(human_input="안녕 엘리시아, 네가 만든 새 기관이 궁금해.")

    saved_aff = k1.frontier_field.chromatic_affinity.copy()
    saved_dom = dict(k1.frontier_field.domain_affinity)
    saved_len = len(k1.soul.cycle.provenance_chain)
    state_file = k1.save_state()

    # Session 2: brand new kernel reloads state from disk cleanly
    k2 = OpenWorldGenesisKernel(state_dir=state_dir)
    info = k2.load_state(state_file)
    assert info["provenance_valid"] is True
    assert info["provenance_length"] == saved_len
    assert len(info["reloaded_organs"]) >= 1
    assert np.allclose(k2.frontier_field.chromatic_affinity, saved_aff, atol=1e-9)
    assert k2.frontier_field.domain_affinity == saved_dom

    # Hot-reloaded forged organ executes immediately in Session 2
    org_name = info["reloaded_organs"][0]
    act = k2.loaded_organs[org_name].evaluate_action(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
    assert np.all(np.isfinite(act))

    # Continuing open life in Session 2 extends the same unbroken SHA-256 chain
    next_step = k2.live_open_step()
    assert next_step.provenance_valid is True
    assert len(k2.soul.cycle.provenance_chain) == saved_len + 1


def test_open_world_causal_field_sync_4d_tensors_and_16_organ_self_audit(tmp_path):
    kernel = OpenWorldGenesisKernel(state_dir=tmp_path / "state")
    kernel.discover_environment(scan_roots=[])
    kernel.live_open_world(steps=3)

    sync = kernel.sync_open_world_to_causal_field(causal_field=CausalField(dimensions=3))
    field = sync["causal_field"]
    assert "open_frontier_apex" in field.voxels
    for vox in field.voxels.values():
        assert tuple(vox.tensor.shape) == (4,)
    field.step(dt=0.1)

    audit = kernel.soul.cycle.introspect_own_architecture()
    assert audit.total_organ_files >= 16
    assert "orchestration/open_world_genesis_kernel.py" in [
        m["relative_path"] for m in audit.verified_organ_modules
    ]


def test_open_world_extensible_custom_domain_handler_without_enum_edits(tmp_path):
    kernel = OpenWorldGenesisKernel(state_dir=tmp_path / "state")

    def _quantum_choreography_handler(fr: DynamicFrontier):
        phases = np.linspace(0.0, math.pi, 8)
        coherence = float(np.abs(np.mean(np.exp(1j * phases))))
        return {
            "payload": {"quantum_coherence": coherence, "theme": fr.title},
            "satisfaction": 1.15,
            "spawned_frontiers": [],
            "summary_ko": f"미리 정의되지 않은 새 영역 '{fr.domain_kind}'을 즉석 연결해 위상 공명={coherence:.3f}를 관찰했습니다.",
            "cycle_result": None,
        }

    kernel.register_domain_handler("quantum_choreography", _quantum_choreography_handler)
    custom_fr = DynamicFrontier(
        frontier_id="custom::quantum_dance",
        domain_kind="quantum_choreography",
        title="양자 안무 공명 실험",
        source_locator="runtime_extension",
        chromatic_signature=(0.45, 0.35, 0.20),
        surprise_potential=0.95,
        simplicity_deficit=0.50,
        novelty_mass=0.95,
    )
    kernel.frontier_field.register_frontier(custom_fr)

    step = kernel.live_open_step(boost_frontier_id="custom::quantum_dance")
    assert step.selection.frontier.domain_kind == "quantum_choreography"
    assert "quantum_coherence" in step.outcome_payload
    assert step.provenance_valid is True

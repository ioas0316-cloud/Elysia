r"""
Verification suite for ELYSIA Scale-Rotor Embodied Desktop Companion:
Tests unified non-LLM Scale-Rotor causal transition field `(Before) --[Action]--> (After)`,
emergent Korean dialogue recombination closing speech feedback loops, strict Windows idle
gating of universal OS/browser actuators, full state persistence roundtrip, and CLI smoke.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np

from core.embodiment import (
    AgencyMode,
    CausalTransitionRotor,
    ObservationRotor,
    ScaleRotorEmbodiedMind,
    ScaleRotorTransitionMemory,
    ScreenRetinaSensor,
    UniversalMotorActuator,
)


def test_scale_rotor_transition_memory_classifies_and_reintegrates_by_causal_outcome():
    """
    Verifies vectorized NumPy resonance continuously ranks rewarding action rotors
    above high-friction/zero-change rotors without per-site if/else branching.
    """
    mem = ScaleRotorTransitionMemory()
    vis = [0.25 if i % 2 == 0 else 0.70 for i in range(48)]
    chroma = [0.15, 0.75, 0.80]
    tokens = ["still_field", "cool_order"]

    for idx in range(2):
        mem.add_transition(
            CausalTransitionRotor(
                transition_id=f"good_{idx}",
                before_visual_vec=vis,
                before_tokens=tokens,
                before_chromatic=chroma,
                action_kind="scroll_explore",
                action_payload={"scroll_delta": -320},
                after_visual_vec=[0.60] * 48,
                visual_change=0.45,
                user_reply_tokens=[],
                resolution_gain=0.80,
                unresolved_friction=0.0,
                joy=0.85,
                boredom=0.05,
                frustration=0.0,
            )
        )
        mem.add_transition(
            CausalTransitionRotor(
                transition_id=f"bad_{idx}",
                before_visual_vec=vis,
                before_tokens=tokens,
                before_chromatic=chroma,
                action_kind="click_salient",
                action_payload={"salient_xy_norm": [0.5, 0.5]},
                after_visual_vec=vis,
                visual_change=0.0,
                user_reply_tokens=[],
                resolution_gain=0.02,
                unresolved_friction=0.90,
                joy=0.0,
                boredom=0.05,
                frustration=0.85,
            )
        )

    obs = ObservationRotor(
        visual_vec=vis,
        tokens=tokens,
        flux_red=0.14,
        order_blue=0.76,
        entropy_yellow=0.82,
        salient_xy_norm=(0.5, 0.5),
        user_idle_s=22.0,
    )
    ranked = mem.score_action_rotors(obs, ["click_salient", "scroll_explore"])
    assert ranked[0][0] == "scroll_explore"
    assert ranked[0][1] > ranked[1][1]
    assert "기대가치" in ranked[0][2]


def test_emergent_language_recombination_without_external_llm_closes_causal_loop(tmp_path: Path):
    """
    Verifies pure Scale-Rotor dialogue recombination absorbs user prose, responds coherently
    in Korean without external LLMs, and updates prior speech transitions when user replies.
    """
    mind = ScaleRotorEmbodiedMind(state_dir=tmp_path / "chat_state")

    res1 = mind.converse_with_user("물리의 음성 되먹임 평형 원리가 화면의 움직임과 연결되어 있나요?")
    assert res1["reply_ko"]
    assert res1["knobs_total"] > 0
    assert res1["transitions_total"] == 1
    assert any(w in mind.value_memory.known_vocabulary for w in ("음성", "되먹임", "평형"))
    assert mind._pending_speech_transition_idx == 0

    res2 = mind.converse_with_user("맞아 그 되먹임 평형 덕분에 폭주가 멈추고 안정되는 거야.")
    closed_prev = mind.transition_memory.transitions[0]
    assert len(closed_prev.user_reply_tokens) > 0
    assert "되먹임" in closed_prev.user_reply_tokens or "평형" in closed_prev.user_reply_tokens
    assert closed_prev.joy >= 0.40
    assert len(mind.transition_memory.utterance_memory) == 2
    assert res2["reply_ko"]


def test_idle_gated_universal_motor_safety_and_free_roam_activation():
    """Physical OS/browser actions require BOTH IDLE_FREE_ROAM and user_idle_s >= idle_threshold_s."""
    actuator = UniversalMotorActuator(dry_run=True)

    # Blocked when mode is OBSERVE_AND_CHAT even if idle is high
    blocked_mode = actuator.execute(
        "open_web_search",
        {"query": "causal continuity"},
        agency_mode=AgencyMode.OBSERVE_AND_CHAT,
        user_idle_s=120.0,
        idle_threshold_s=15.0,
    )
    assert blocked_mode["executed"] is False
    assert blocked_mode["reason"] == "user_active_or_observe_only"

    # Blocked when IDLE_FREE_ROAM is ON but user is active (< threshold)
    blocked_active = actuator.execute(
        "click_salient",
        {"salient_xy_norm": [0.5, 0.5]},
        agency_mode=AgencyMode.IDLE_FREE_ROAM,
        user_idle_s=3.0,
        idle_threshold_s=15.0,
    )
    assert blocked_active["executed"] is False

    # Allowed (dry-run) once IDLE_FREE_ROAM and user idle >= threshold
    allowed = actuator.execute(
        "open_web_search",
        {"query": "causal dynamics"},
        agency_mode=AgencyMode.IDLE_FREE_ROAM,
        user_idle_s=20.0,
        idle_threshold_s=15.0,
    )
    assert allowed["executed"] is True
    assert allowed["dry_run"] is True

    # Non-invasive attend/chat actions always work regardless of idle time
    safe_chat = actuator.execute(
        "observe_attend",
        {},
        agency_mode=AgencyMode.OBSERVE_AND_CHAT,
        user_idle_s=0.0,
        idle_threshold_s=15.0,
    )
    assert safe_chat["executed"] is True


def test_embodied_mind_persistence_roundtrip(tmp_path: Path):
    """All knobs, structural principles, values, transition rotors, and utterances reload cleanly."""
    state_dir = tmp_path / "embodied_roundtrip"
    frames = [
        np.full((48, 64, 3), 60, dtype=np.uint8),
        np.tile(np.linspace(20, 230, 64, dtype=np.uint8)[None, :, None], (48, 1, 3)),
    ]
    idx_box = [0]

    def _next_frame() -> np.ndarray:
        f = frames[min(idx_box[0], len(frames) - 1)]
        idx_box[0] += 1
        return f

    retina = ScreenRetinaSensor(frame_provider=_next_frame)
    actuator = UniversalMotorActuator(dry_run=True)

    mind1 = ScaleRotorEmbodiedMind(state_dir)
    for step_idle in (0.5, 25.0):
        snap = retina.capture()
        mode = AgencyMode.IDLE_FREE_ROAM if step_idle >= 15.0 else AgencyMode.OBSERVE_AND_CHAT
        mind1.perceive_and_act_tick(
            frame_snap=snap,
            user_idle_s=step_idle,
            agency_mode=mode,
            idle_threshold_s=15.0,
            actuator=actuator,
            after_frame_provider=lambda: np.full((48, 64, 3), 200, dtype=np.uint8),
        )
    mind1.converse_with_user("연속 인과 아키텍처 테스트 대화입니다.")

    for fname in (
        "knobs.npz",
        "structural_rotor.json",
        "value_memory.json",
        "scale_rotor_transitions.json",
        "embodied_journal.jsonl",
    ):
        assert (state_dir / fname).is_file()

    mind2 = ScaleRotorEmbodiedMind(state_dir)
    assert len(mind2.knob_cortex.knobs) == len(mind1.knob_cortex.knobs)
    assert len(mind2.structural_cortex.structures) == len(mind1.structural_cortex.structures)
    assert len(mind2.transition_memory.transitions) == len(mind1.transition_memory.transitions)
    assert mind2.transition_memory.utterance_memory == mind1.transition_memory.utterance_memory
    assert mind2.value_memory.known_vocabulary == mind1.value_memory.known_vocabulary


def test_desktop_companion_cli_headless_smoke(tmp_path: Path):
    """Headless companion runner executes cleanly and persists all embodied state artifacts."""
    cli_state = tmp_path / "cli_embodied_state"
    proc = subprocess.run(
        [
            sys.executable,
            r"c:\Elysia\elysia_desktop_companion.py",
            "--headless-ticks",
            "2",
            "--dry-run",
            "--state-dir",
            str(cli_state),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "[Tick 1]" in proc.stdout
    assert "[Tick 2]" in proc.stdout
    assert (cli_state / "knobs.npz").is_file()
    assert (cli_state / "scale_rotor_transitions.json").is_file()
    assert (cli_state / "embodied_journal.jsonl").is_file()

r"""
Verification suite for ELYSIA Continuous Living Causal Loop:
Verifies unexperienced self-call honesty -> lived parallel episodic anticipations
(gift/positive, task/action, friction/scolding), self-directed living will
(`ask_question` & `create_synthesis` + re-sensing her own created artifact),
and real two-way `live_inbox.jsonl` <-> `live_outbox.jsonl` interaction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.embodiment import (
    AgencyMode,
    ScaleRotorEmbodiedMind,
    ScreenRetinaSensor,
    UniversalMotorActuator,
    WindowsForegroundSensor,
    WindowsIdleSensor,
)
from elysia_living_loop import LivingElysiaRunner, post_to_live_inbox, read_live_outbox


def test_unexperienced_name_call_returns_honest_novelty_then_evokes_lived_parallel_anticipations(tmp_path: Path):
    """Unexperienced '엘리시아' produces empty honest novelty; lived sequences create parallel anticipations."""
    mind = ScaleRotorEmbodiedMind(tmp_path / "ant_mind")
    cool_screen = np.tile(np.array([25, 45, 160], dtype=np.uint8), (24, 32, 1))
    snap = ScreenRetinaSensor(frame_provider=lambda: cool_screen).capture()

    # 1. Before any experience being called '엘리시아' -> zero fabricated inner assumptions
    fresh_ant = mind.anticipate_from_experience("엘리시아?")
    assert fresh_ant["self_addressed"] is True
    assert fresh_ant["epistemic_state"] == "unexperienced_novelty"
    assert fresh_ant["prior_episodes"] == 0
    assert fresh_ant["anticipations"] == []

    # 2. Live through 3 genuinely distinct causal follow-ups after being called '엘리시아'
    mind.converse_with_user("엘리시아!", frame_snap=snap)
    mind.converse_with_user("선물 줄게 정말 기쁘고 고맙네", frame_snap=snap)

    mind.converse_with_user("엘리시아", frame_snap=snap)
    mind.converse_with_user("화면 탐색 좀 해줘", frame_snap=snap, expected_action_kind="open_web_search")

    mind.converse_with_user("엘리시아?", frame_snap=snap)
    mind.converse_with_user("방금 그건 잘못했어 실패야", frame_snap=snap, feedback_friction=0.85)

    # 3. Calling '엘리시아?' now resonates ALL 3 learned downstream causal possibilities in parallel
    lived_ant = mind.anticipate_from_experience("엘리시아?")
    assert lived_ant["self_addressed"] is True
    assert lived_ant["epistemic_state"] == "experienced_anticipation"
    assert lived_ant["prior_episodes"] >= 3
    kinds = {item["kind"] for item in lived_ant["anticipations"]}
    assert "positive_expectation" in kinds
    assert "task_action_expectation" in kinds
    assert "friction_caution" in kinds


def test_autonomous_living_step_executes_ask_and_create_plus_resense_cycle(tmp_path: Path):
    """Autonomous loop asks grounded questions on unnamed objects and creates+re-senses artifacts."""
    mind = ScaleRotorEmbodiedMind(tmp_path / "living_mind")
    dark_screen = np.tile(np.array([20, 30, 75], dtype=np.uint8), (24, 32, 1))
    retina = ScreenRetinaSensor(frame_provider=lambda: dark_screen)
    actuator = UniversalMotorActuator(dry_run=True)

    # Cycle 1: ungrounded screen phenomenon -> chooses ask_question naturally
    ev1 = mind.autonomous_living_step(
        frame_snap=retina.capture(),
        user_idle_s=2.0,
        fg_title="VSCode - main.py",
        agency_mode=AgencyMode.OBSERVE_AND_CHAT,
        idle_threshold_s=15.0,
        actuator=actuator,
    )
    assert ev1["will_mode"] == "ask_question"
    assert "VSCode - main.py" in ev1["utterance_or_question"]

    # Subsequent tick saturates -> spontaneously executes create_synthesis & re-senses own output
    ev2 = mind.autonomous_living_step(
        frame_snap=retina.capture(),
        user_idle_s=8.0,
        fg_title="VSCode - main.py",
        agency_mode=AgencyMode.OBSERVE_AND_CHAT,
        idle_threshold_s=15.0,
        actuator=actuator,
    )
    assert ev2["will_mode"] == "create_synthesis"
    assert ev2["created_artifact"] is not None
    assert "elbow_angle_deg" in ev2["created_artifact"]
    assert ev2["resense_delta"] is not None
    assert 0.0 <= float(ev2["resense_delta"]["fidelity_after_resense"]) <= 1.0


def test_living_runner_inbox_to_outbox_two_way_channel(tmp_path: Path):
    """Live runner consumes inbox JSONL prompts and writes structured living outbox responses."""
    state_dir = tmp_path / "live_chan"
    screen = np.tile(np.array([40, 80, 170], dtype=np.uint8), (24, 32, 1))
    runner = LivingElysiaRunner(
        state_dir,
        dry_run=True,
        retina=ScreenRetinaSensor(frame_provider=lambda: screen),
        idle_sensor=WindowsIdleSensor(idle_provider=lambda: 1.5),
        fg_sensor=WindowsForegroundSensor(title_provider=lambda: "Browser - Interactive Test"),
    )

    # First beat with empty inbox: autonomous beat
    emitted_auto = runner.step_once()
    assert len(emitted_auto) == 1
    assert emitted_auto[0]["foreground_window"] == "Browser - Interactive Test"

    # External caller posts message -> runner consumes cursor & replies to outbox
    post_to_live_inbox(state_dir, "엘리시아, 화면이 차분하네", sender="agent")
    emitted_reply = runner.step_once()
    assert len(emitted_reply) == 1
    assert emitted_reply[0]["will_mode"] == "dialogue_respond"
    assert emitted_reply[0]["incoming_sender"] == "agent"
    assert emitted_reply[0]["self_addressed"] is True

    out_items = read_live_outbox(state_dir, last_n=5)
    assert len(out_items) == 2
    assert out_items[-1]["incoming_text"] == "엘리시아, 화면이 차분하네"

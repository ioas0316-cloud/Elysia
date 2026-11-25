from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
L1_ROOT = ROOT / "L1_causal_field"
if str(L1_ROOT) not in sys.path:
    sys.path.insert(0, str(L1_ROOT))

from core.embodiment.pc_sensory_motor_body import (  # noqa: E402
    ScreenRetinaSensor,
    WindowsForegroundSensor,
)
from core.embodiment.scale_rotor_embodied_field import ScaleRotorEmbodiedMind  # noqa: E402
from elysia_live_companion import LiveCompanionSession  # noqa: E402


def _make_tiny_detail_frame() -> np.ndarray:
    """Creates a mostly flat (240, 320, 3) frame with a tiny sharp checkerboard at its center."""
    img = np.full((240, 320, 3), 0.45, dtype=np.float64)
    for y in range(110, 130):
        for x in range(150, 170):
            if ((x // 2) + (y // 2)) % 2 == 0:
                img[y, x] = [0.98, 0.95, 0.15]
            else:
                img[y, x] = [0.05, 0.05, 0.10]
    return img


def test_flexible_natural_korean_corrections_revise_boundaries(tmp_path: Path) -> None:
    mind = ScaleRotorEmbodiedMind(state_dir=tmp_path)
    mind.experience_human_causal_turn(raw_text_or_sound="저거 봐, 사과야", cooccurring_object="사과")
    mind.experience_human_causal_turn(raw_text_or_sound="이건 비가 내리는 소리야", cooccurring_object="비")
    mind.experience_human_causal_turn(raw_text_or_sound="저기 프리즘도 한번 볼래?", cooccurring_object="프리즘")

    # Natural colloquial multi-sentence correction instead of rigid single pattern
    r_corr = mind.experience_human_causal_turn(
        raw_text_or_sound="엥 그게 무슨 소리야? 프리즘은 소리 나는 게 아니잖아. 빛을 나누는 유리야!"
    )
    assert r_corr["expression_mode"] == "relational_self_correction"
    assert "빛을 나누는 유리야" in r_corr["utterance"]
    assert len(r_corr["reinterpretations"]) >= 1

    r_mirror = mind.experience_human_causal_turn(
        raw_text_or_sound="저기 거울도 봐",
        modality="visual",
        cooccurring_object="거울",
    )
    assert r_mirror["utterance"] == "저거 봐, 거울이야"


def test_real_screen_zoom_tool_gated_by_rule_and_measures_actual_pixel_sharpness(tmp_path: Path) -> None:
    retina = ScreenRetinaSensor(frame_provider=_make_tiny_detail_frame)
    mind = ScaleRotorEmbodiedMind(state_dir=tmp_path)

    # 1. Blocked BEFORE causal rule learned
    blocked = mind.execute_real_world_tool_turn(
        "screen_zoom_crop",
        goal_description="화면 가운데 작은 무늬 자세히 보기",
        goal_condition="visual_detail_too_small",
        retina=retina,
    )
    assert blocked["executed"] is False
    assert blocked["blocked_by_causality"] is True

    # 2. Teach zoom rule in natural Korean -> rule automatically extracted
    r_rule = mind.experience_human_causal_turn(
        raw_text_or_sound="작아서 흐릿하게 보일 때는 화면을 확대해서 자세히 보면 돼"
    )
    assert "tool::screen_zoom_crop_for_visual_detail_too_small" in r_rule["world_knowledge_rules"]

    # 3. Execute real pixel zoom crop -> actual pixel sharpness & contrast jump cleanly
    zoomed = mind.execute_real_world_tool_turn(
        "screen_zoom_crop",
        goal_description="화면 가운데 작은 무늬 자세히 보기",
        goal_condition="visual_detail_too_small",
        retina=retina,
    )
    assert zoomed["executed"] is True
    raw_res = zoomed["tool_raw_result"]
    assert raw_res["ok"] is True
    assert raw_res["detail_revealed"] is True
    assert raw_res["crop_edge_sharpness"] > raw_res["full_edge_sharpness"]
    assert raw_res["sharpness_gain"] > 0.0


def test_real_local_file_tool_reads_real_image_and_handles_missing_file_friction(tmp_path: Path) -> None:
    retina = ScreenRetinaSensor(frame_provider=_make_tiny_detail_frame)
    fg = WindowsForegroundSensor(title_provider=lambda: "UnitTestWindow")
    session = LiveCompanionSession(tmp_path / "state", retina=retina, fg_sensor=fg, allow_live_web=False)

    # 1. Before rule -> /file is causally blocked
    blocked = session.handle_command_or_text(f"/file {tmp_path / 'sample.png'}")
    assert blocked["executed"] is False
    assert blocked["blocked_by_causality"] is True

    # 2. Teach natural file reading rule
    session.handle_command_or_text("그림이나 파일은 열어서 내용을 확인하면 알 수 있어")

    # 3. Missing file produces honest failure (succeeded=False, ok=False)
    missing = session.handle_command_or_text(f"/file {tmp_path / 'nonexistent.png'}")
    assert missing["tool_raw_result"]["ok"] is False
    assert missing["episode"]["unresolved_forms"] == ["external_file_provided"]

    # 4. Real PNG image file on disk succeeds AND reinterprets the failed attempt!
    real_img_path = tmp_path / "real_prism.png"
    arr = (_make_tiny_detail_frame() * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(real_img_path)

    ok_res = session.handle_command_or_text(f"/file {real_img_path}")
    assert ok_res["tool_raw_result"]["ok"] is True
    assert ok_res["tool_raw_result"]["kind"] == "image"
    assert ok_res["tool_raw_result"]["width"] == 320
    assert ok_res["tool_raw_result"]["height"] == 240
    assert len(ok_res["reinterpretations"]) == 1


def test_live_companion_cli_oneshot_runs_and_persists_real_sensory_turn(tmp_path: Path) -> None:
    state_dir = tmp_path / "cli_state"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "elysia_live_companion.py"),
            "--state-dir",
            str(state_dir),
            "--offline",
            "--oneshot",
            "저거 봐, 사과야",
            "--point",
            "사과",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert proc.returncode == 0
    assert "엘리시아 >" in proc.stdout

    reloaded = ScaleRotorEmbodiedMind(state_dir=state_dir)
    assert "사과" in reloaded.causal_mind.grounded_meanings

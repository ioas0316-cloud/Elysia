import base64
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from core.embodiment import (
    AgencyMode,
    LiveAudioEarSensor,
    ScaleRotorEmbodiedMind,
    ScreenRetinaSensor,
    UniversalMotorActuator,
)


def _moving_stimulus_frame(step_idx: int) -> np.ndarray:
    arr = np.full((120, 160, 3), 25, dtype=np.uint8)
    if step_idx == 0:
        arr[10:45, 10:55, :] = [235, 220, 40]
    else:
        arr[70:110, 105:152, :] = [40, 225, 240]
        arr[80:98, 118:140, :] = [250, 50, 50]
    return arr


def test_real_eye_gaze_steering_fovea_contours_and_audio_ear(tmp_path: Path) -> None:
    idx = {"n": 0}

    def _provider() -> np.ndarray:
        frame = _moving_stimulus_frame(idx["n"])
        idx["n"] += 1
        return frame

    retina = ScreenRetinaSensor(frame_provider=_provider)
    ear = LiveAudioEarSensor(audio_provider=lambda: 0.25 * np.sin(np.linspace(0, 40 * np.pi, 512)))

    snap1 = retina.capture()
    snap2 = retina.capture()
    aud = ear.listen()

    # Gaze physically moves from top-left toward bottom-right stimulus with real fovea contours
    assert snap1.gaze_xy_norm[0] < snap2.gaze_xy_norm[0]
    assert snap1.gaze_xy_norm[1] < snap2.gaze_xy_norm[1]
    assert snap2.gaze_shift > 0.1
    assert snap2.fovea_contours >= 1
    assert aud.capture_ok is True
    assert aud.active_sound is True
    assert aud.rms_energy > 0.05

    mind = ScaleRotorEmbodiedMind(tmp_path / "state")
    act = UniversalMotorActuator(dry_run=True)
    res1 = mind.perceive_and_act_tick(
        frame_snap=snap1,
        user_idle_s=1.0,
        agency_mode=AgencyMode.OBSERVE_AND_CHAT,
        idle_threshold_s=15.0,
        actuator=act,
        audio_snap=aud,
    )
    res2 = mind.perceive_and_act_tick(
        frame_snap=snap2,
        user_idle_s=1.0,
        agency_mode=AgencyMode.OBSERVE_AND_CHAT,
        idle_threshold_s=15.0,
        actuator=act,
        audio_snap=aud,
    )

    # Persistent body responds directly to real gaze change (not identical static toy angles)
    assert res1["body_reality"]["target_gaze_xy"] != res2["body_reality"]["target_gaze_xy"]
    assert res1["body_reality"]["elbow_deg"] != res2["body_reality"]["elbow_deg"]
    assert res2["audio_ear"]["active_sound"] is True
    assert res2["eye_fovea"]["fovea_contours"] >= 1


def test_unknown_github_two_way_interaction_drives_fabric_and_body(tmp_path: Path) -> None:
    foreign_rust = "fn main() {\n    let resonance: f64 = 3.14;\n    println!(\"{}\", resonance);\n}\n"
    rust_b64 = base64.b64encode(foreign_rust.encode("utf-8")).decode("ascii")
    readme_b64 = base64.b64encode(b"# Foreign Rust Project\nHandles realtime wave routing.\n").decode("ascii")

    def _mock_github(url: str, timeout: float, headers: Optional[Dict[str, str]]) -> Tuple[int, bytes, str]:
        if "/events" in url:
            payload = [{"id": "e1", "type": "PushEvent", "repo": {"name": "unknown-org/wave-rs"}, "actor": {"login": "dev"}}]
            return 200, json.dumps(payload).encode("utf-8"), "application/json"
        if url.endswith("/repos/unknown-org/wave-rs"):
            payload = {"full_name": "unknown-org/wave-rs", "description": "Real-time wave router", "language": "Rust", "stargazers_count": 42}
            return 200, json.dumps(payload).encode("utf-8"), "application/json"
        if url.endswith("/readme"):
            return 200, json.dumps({"content": readme_b64}).encode("utf-8"), "application/json"
        if url.endswith("/contents"):
            entries = [
                {"name": "README.md", "path": "README.md", "type": "file", "size": 50},
                {"name": "main.rs", "path": "main.rs", "type": "file", "size": len(foreign_rust)},
            ]
            return 200, json.dumps(entries).encode("utf-8"), "application/json"
        if url.endswith("/contents/main.rs"):
            return 200, json.dumps({"content": rust_b64, "size": len(foreign_rust)}).encode("utf-8"), "application/json"
        if "/issues" in url:
            issues = [{"number": 7, "title": "Fix phase drift under high load", "state": "open", "body": "Observed drift"}]
            return 200, json.dumps(issues).encode("utf-8"), "application/json"
        return 404, b"{}", "application/json"

    mind = ScaleRotorEmbodiedMind(tmp_path / "state")
    gh_res = mind.interact_with_live_github_world(fetcher=_mock_github, allow_outward_enact=False)

    assert gh_res["ok"] is True
    assert gh_res["repo"] == "unknown-org/wave-rs"
    assert gh_res["language"] == "Rust"
    assert gh_res["inspected_file"] == "main.rs"
    assert gh_res["issues_count"] == 1
    # Foreign Rust code clashed against Python compiler protocol -> genuine repulsion boundary & body resistance!
    assert gh_res["new_repulsion_boundaries"] >= 1
    assert gh_res["body_pose"]["external_resistance"] >= 0.7
    assert len(gh_res["outward_contact_chains"]) >= 1

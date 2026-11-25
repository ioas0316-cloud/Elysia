r"""
Verification suite for ELYSIA Past-Present-Future Limit Projection, Outward World Excursion,
and Cross-Scale Reinterpretation (Micro <-> Meso <-> Macro):
Tests stagnation-limit projection -> purpose/query synthesis, deterministic mock web excursion
expanding all 3 scales cleanly, and non-trivial `reinterpretation_shift` when re-viewing
her own pre-excursion observation through newly expanded cognitive axes.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from core.embodiment import (
    AgencyMode,
    ScaleRotorEmbodiedMind,
    ScreenRetinaSensor,
    UniversalMotorActuator,
)


def _mock_world_fetcher(url: str, timeout: float, headers: Optional[Dict[str, str]]) -> Tuple[int, bytes, str]:
    if "wikipedia.org/api/rest_v1/page/summary/" in url:
        payload = {
            "title": "Self-organization" if "en." in url else "창발",
            "description": "Process where global order arises from local interactions",
            "extract": (
                "Self-organization creates coherent global patterns through continuous feedback, "
                "local docking resonance, and phase equilibrium across boundary conditions. "
                "Information flows from micro interactions to emergent macro order."
            ),
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Self-organization"}},
        }
        return 200, json.dumps(payload).encode("utf-8"), "application/json"

    if "api.github.com/search/repositories" in url:
        return (
            200,
            json.dumps(
                {
                    "items": [
                        {"html_url": "https://github.com/elysia-test/causal-field-sim"},
                        {"html_url": "https://github.com/elysia-test/scale-rotor-morph"},
                    ]
                }
            ).encode("utf-8"),
            "application/json",
        )

    if "api.github.com/repos/" in url and url.endswith("/readme"):
        readme = (
            "# Causal Field Simulator\n\n"
            "Models continuous interactions across micro, meso, and macro rotors.\n\n"
            "```python\n"
            "def evolve_field(psi, dt):\n"
            "    return psi + dt * divergence(psi)\n"
            "```\n"
        )
        return 200, json.dumps({"content": base64.b64encode(readme.encode("utf-8")).decode("ascii")}).encode("utf-8"), "application/json"

    if "api.github.com/repos/" in url:
        return (
            200,
            json.dumps(
                {
                    "full_name": "elysia-test/causal-field-sim",
                    "description": "Continuous causal topological engine",
                    "language": "Python",
                    "stargazers_count": 420,
                    "topics": ["physics", "topology"],
                }
            ).encode("utf-8"),
            "application/json",
        )

    if "wikisource.org/w/api.php" in url and "list=search" in url:
        return (
            200,
            json.dumps(
                {
                    "query": {
                        "search": [
                            {"title": "운수 좋은 날"},
                        ]
                    }
                }
            ).encode("utf-8"),
            "application/json",
        )

    if "wikisource.org/w/api.php" in url and "prop=extracts" in url:
        sample_story = (
            "차가운 비가 끝없이 내리는 골목길을 걸으며 김 첨지는 깊게 한숨을 쉬었다.\n"
            "집 안으로 들어서며 김 첨지는 아픈 아내를 바라보고 조용히 물었다. "
            '"오늘은 몸과 마음이 조금 편하시오?"\n'
            "아내는 촛불 아래에서 희미하게 웃으며 대답했다. "
            '"당신이 돌아와서 이제야 마음이 한결 놓여요."'
        )
        return (
            200,
            json.dumps(
                {
                    "query": {
                        "pages": {
                            "1": {"title": "운수 좋은 날", "extract": sample_story}
                        }
                    }
                }
            ).encode("utf-8"),
            "application/json",
        )

    return 404, b"not found", "text/plain"


def test_future_limit_projection_and_excursion_cross_scale_reinterpretation(tmp_path: Path):
    """Static ticks trigger stagnation limit projection; excursion expands all 3 scales & reinterprets old state."""
    mind = ScaleRotorEmbodiedMind(tmp_path / "exc_mind")
    static_screen = np.tile(np.array([25, 35, 65], dtype=np.uint8), (24, 32, 1))
    retina = ScreenRetinaSensor(frame_provider=lambda: static_screen)
    actuator = UniversalMotorActuator(dry_run=True)

    # 3 static ticks build stagnation ceiling -> drives future limit awareness
    for _ in range(3):
        mind.perceive_and_act_tick(
            frame_snap=retina.capture(),
            user_idle_s=0.5,
            agency_mode=AgencyMode.OBSERVE_AND_CHAT,
            idle_threshold_s=15.0,
            actuator=actuator,
            foreground_window="Static Terminal",
        )

    proj = mind.project_future_limit_and_purpose()
    assert proj["projected_limit"] in ("stagnation_ceiling", "epistemic_naming_limit", "recreation_gap_limit")
    assert proj["purpose_ko"]
    assert len(proj["wiki_topics"]) >= 1

    report = mind.embark_external_world_excursion(
        fetch_web=True,
        max_concepts=2,
        max_repos=2,
        fetcher=_mock_world_fetcher,
    )
    b = report["before"]
    a = report["after"]
    reinterp = report["reinterpretation"]

    assert a["nodes"] > b["nodes"]
    assert a["docked_links"] > b["docked_links"]
    assert a["prism_prototypes"] > b["prism_prototypes"]
    assert a["language_stems"] > b["language_stems"]
    assert a["structures"] > b["structures"]
    assert reinterp["deltas"]["micro"]["discovered_axes"] >= 1
    assert reinterp["reinterpretation_shift"] > 0.0

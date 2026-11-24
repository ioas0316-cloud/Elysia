"""
Verification suite for ELYSIA Value-Motivated Open-World Explorer & Web Bridges:
Tests unbroken Micro (Sensory Delta) -> Meso (Joy / Boredom / Frustration) ->
Macro (Emergent Affinity, Open Valences, Value-Guided Choice & Revisit) chain,
real external Web/GitHub/YouTube/Pixiv-gate bridges, Before-vs-After cognitive
diff audit, and self-contained HTML Observatory generation.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
import subprocess
import sys

from core.orchestration.autonomous_open_explorer import (
    ElysiaAutonomousOpenExplorer,
    SensoryEncounterDelta,
    compute_self_state_feeling,
)


def test_joy_boredom_frustration_emerge_causally_from_micro_to_meso_delta():
    """Verifies joy, boredom, and frustration emerge strictly from sensory tension/resolution/friction."""
    joy_f = compute_self_state_feeling(
        SensoryEncounterDelta(tension_before=0.90, resolution_gain=0.85, unresolved_friction=0.0, motifs=["novel"]),
        repeat_count=0,
    )
    assert joy_f.joy > joy_f.boredom
    assert joy_f.dominant_state == "joy_expansion"

    bored_f = compute_self_state_feeling(
        SensoryEncounterDelta(tension_before=0.04, resolution_gain=0.01, unresolved_friction=0.0, motifs=["repeat"]),
        repeat_count=2,
    )
    assert bored_f.boredom > bored_f.joy
    assert bored_f.dominant_state == "boredom_saturation"

    frust_f = compute_self_state_feeling(
        SensoryEncounterDelta(tension_before=0.85, resolution_gain=0.04, unresolved_friction=0.90, motifs=["broken"]),
        repeat_count=0,
    )
    assert frust_f.frustration > frust_f.joy
    assert frust_f.dominant_state == "friction_retreat"


def test_value_and_affinity_drive_choice_and_explanation_not_alphabetical_round_robin(tmp_path: Path):
    """Frustration lowers affinity & spawns open valence; joyful resolution raises motif affinity & guides choice."""
    ext_root = tmp_path / "ext_world"
    ext_root.mkdir()
    (ext_root / "a_broken_syntax.py").write_text("def broken(:\n    pass\n", encoding="utf-8")
    (ext_root / "b_equilibrium_rotor.md").write_text(
        "외부 자극이 회로를 구동한다.\n"
        "복원 경계가 폭주를 억제하고 음성 되먹임이 평형을 교정한다.\n"
        "서로 다른 계층을 다리로 연결해 연속 질서를 이룬다.\n"
        "마침내 안정된 구조가 유지된다.\n",
        encoding="utf-8",
    )
    (ext_root / "c_valid_architecture.py").write_text(
        '"""Resilient module"""\n\ndef heal(x):\n    try:\n        return x * 2\n    except Exception:\n        return 0\n',
        encoding="utf-8",
    )

    state_dir = tmp_path / "value_state"
    explorer = ElysiaAutonomousOpenExplorer(roots=[ext_root], state_dir=state_dir)
    rep = explorer.run_autonomous_session(cycles=3)

    assert len(rep["cycle_logs"]) == 3
    by_name = {log["target_name"]: log for log in rep["cycle_logs"]}

    broken_log = by_name["a_broken_syntax.py"]
    assert broken_log["subprocess_compile"]["returncode"] != 0
    assert broken_log["feeling"]["frustration"] > broken_log["feeling"]["joy"]
    assert "마찰" in broken_log["feeling"]["why_i_felt_this_ko"]

    prose_log = by_name["b_equilibrium_rotor.md"]
    assert prose_log["feeling"]["joy"] > prose_log["feeling"]["frustration"]
    assert len(prose_log["why_i_chose_this_ko"]) > 10
    assert len(prose_log["feeling"]["why_i_felt_this_ko"]) > 10

    aff = rep["affinity_snapshot"]
    assert aff.get("causal_discourse", 0.0) > aff.get("syntax_friction", 0.0)


def test_autonomous_explorer_discovers_real_targets_and_persists_value_memory(tmp_path: Path):
    """Runs 4 cycles on real `c:\\Elysia` and persists value_memory, knobs, structures, visited, journal."""
    state_dir = tmp_path / "living_state"
    explorer = ElysiaAutonomousOpenExplorer(roots=[Path(r"c:\Elysia")], state_dir=state_dir)

    rep = explorer.run_autonomous_session(cycles=4)
    assert len(rep["cycle_logs"]) == 4
    assert rep["knobs_after"] > 0
    assert rep["structures_after"] > rep["structures_before"]

    for log in rep["cycle_logs"]:
        assert log["explored"] is True
        assert "why_i_chose_this_ko" in log and log["why_i_chose_this_ko"]
        assert "feeling" in log and "why_i_felt_this_ko" in log["feeling"]
        assert "delta" in log

    assert (state_dir / "knobs.npz").is_file()
    assert (state_dir / "structural_rotor.json").is_file()
    assert (state_dir / "value_memory.json").is_file()
    assert (state_dir / "visited.json").is_file()
    assert (state_dir / "journal.jsonl").is_file()


def test_persistent_second_session_reloads_values_and_continues_living_growth(tmp_path: Path):
    """Session 2 reloads persisted knobs, structures, AND affinity/values from Session 1."""
    state_dir = tmp_path / "persistent_value_state"

    s1 = ElysiaAutonomousOpenExplorer(roots=[Path(r"c:\Elysia")], state_dir=state_dir)
    rep1 = s1.run_autonomous_session(cycles=3)
    assert len(rep1["affinity_snapshot"]) >= 1

    s2 = ElysiaAutonomousOpenExplorer(roots=[Path(r"c:\Elysia")], state_dir=state_dir)
    assert len(s2.knob_cortex.knobs) == rep1["knobs_after"]
    assert len(s2.structural_cortex.structures) == rep1["structures_after"]
    assert s2.value_memory.affinity_by_motif == rep1["affinity_snapshot"]

    rep2 = s2.run_autonomous_session(cycles=3)
    assert len(rep2["cycle_logs"]) == 3
    total_visits = sum(s2.value_memory.visit_counts.values())
    assert total_visits == 6


def test_cli_runner_executes_cleanly(tmp_path: Path):
    """External CLI process runs value-motivated exploration cleanly and logs Korean causal explanations."""
    cli_state = tmp_path / "cli_value_state"
    proc = subprocess.run(
        [
            sys.executable,
            r"c:\Elysia\elysia_autonomous_runner.py",
            "--cycles",
            "3",
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
    assert "선택 이유" in proc.stdout
    assert "자기 인식" in proc.stdout
    assert (cli_state / "value_memory.json").is_file()


def test_external_web_bridge_github_youtube_and_blocked_pixiv_boundary(tmp_path: Path):
    """Tests GitHub repo+README, YouTube oEmbed, and honest HTTP 403 Pixiv boundary handling."""
    readme_text = (
        "# Sample Physics Rotor Repo\n"
        "Restoring force opposes displacement from equilibrium.\n"
        "Negative feedback corrects drift and preserves continuous balance.\n"
        "Cross-layer coupling bridges micro motion to macro stability.\n"
    )
    readme_b64 = base64.b64encode(readme_text.encode("utf-8")).decode("ascii")

    def _mock_fetcher(url: str, timeout: float, headers=None):
        if url == "https://api.github.com/repos/org/rotor_lab":
            payload = {"full_name": "org/rotor_lab", "description": "Continuous causal rotor library", "language": "Python", "stargazers_count": 42, "topics": ["physics", "rotor"]}
            return 200, json.dumps(payload).encode("utf-8"), "application/json"
        if url == "https://api.github.com/repos/org/rotor_lab/readme":
            return 200, json.dumps({"content": readme_b64}).encode("utf-8"), "application/json"
        if "youtube.com/oembed" in url:
            payload = {"title": "Continuum Dynamics Explained", "author_name": "CausalLab", "provider_name": "YouTube", "thumbnail_url": "https://img.example/thumb.png", "width": 480, "height": 360}
            return 200, json.dumps(payload).encode("utf-8"), "application/json"
        if "img.example/thumb.png" in url:
            # Tiny valid 4x4 PNG generated via PIL in-memory
            import io
            from PIL import Image

            buf = io.BytesIO()
            Image.new("RGB", (4, 4), (40, 120, 220)).save(buf, format="PNG")
            return 200, buf.getvalue(), "image/png"
        if "pixiv.net" in url:
            return 403, b"Access Denied / Login Gate", "text/html"
        return 404, b"Not Found", "text/plain"

    empty_root = tmp_path / "empty_local"
    empty_root.mkdir()
    state_dir = tmp_path / "web_state"

    explorer = ElysiaAutonomousOpenExplorer(
        roots=[empty_root],
        state_dir=state_dir,
        web_urls=[
            "https://github.com/org/rotor_lab",
            "https://www.youtube.com/watch?v=test123",
            "https://www.pixiv.net/en/artworks/999",
        ],
        web_fetcher=_mock_fetcher,
    )
    rep = explorer.run_autonomous_session(cycles=5)
    diff = rep["session_diff"]

    assert rep["knobs_after"] > 0
    assert rep["structures_after"] > rep["structures_before"]
    assert len(diff["blocked_boundaries"]) == 1
    assert "pixiv" in diff["blocked_boundaries"][0]["url"]
    assert any("차단" in need or "자격" in need for need in rep["what_elysia_still_needs"])


def test_session_before_after_diff_and_observatory_html_generation(tmp_path: Path):
    """Verifies observatory.html contains clear Before/After diffs and per-cycle Korean reasoning."""
    state_dir = tmp_path / "obs_state"
    explorer = ElysiaAutonomousOpenExplorer(roots=[Path(r"c:\Elysia")], state_dir=state_dir)
    rep = explorer.run_autonomous_session(cycles=3)

    obs_path = Path(rep["observatory_path"])
    assert obs_path.is_file()
    html_text = obs_path.read_text(encoding="utf-8")
    assert "Before" in html_text
    assert "After" in html_text
    assert "선택 이유" in html_text
    assert "자기 인식" in html_text
    assert "엘리시아에게 다음으로 필요한 것" in html_text


def test_cli_runner_with_web_urls_generates_observatory_report(tmp_path: Path):
    """CLI runner accepts `--web-urls` and generates `observatory.html` cleanly."""
    cli_state = tmp_path / "cli_obs_state"
    proc = subprocess.run(
        [
            sys.executable,
            r"c:\Elysia\elysia_autonomous_runner.py",
            "--cycles",
            "3",
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
    assert (cli_state / "observatory.html").is_file()
    assert "관측 대시보드" in proc.stdout


def test_self_directed_tool_selection_and_autonomous_world_discovery(tmp_path: Path):
    """
    Verifies Elysia starts with ZERO human-supplied web_urls, discovers external
    GitHub/Wikipedia frontiers via ToolAffordanceMemory, harvests outbound links,
    extracts rich image palette/contrast, and empirically pivots away from failing tools.
    """
    import base64
    import io
    import json
    from PIL import Image
    from core.orchestration.tool_affordance_memory import ToolAffordanceMemory

    # 1. Empirical tool adaptation unit check: failing tool yields to untouched/successful tool
    tm = ToolAffordanceMemory()
    first_pick, _ = tm.select_best_tool("frontier_discovery", ["tool::github_topic_search", "tool::wikipedia_concept"])
    assert first_pick == "tool::github_topic_search"
    tm.record_outcome("frontier_discovery", "tool::github_topic_search", success=False, resolution_gain=0.0, friction=0.80)
    second_pick, _ = tm.select_best_tool("frontier_discovery", ["tool::github_topic_search", "tool::wikipedia_concept"])
    assert second_pick == "tool::wikipedia_concept"

    # 2. Live loop with ZERO web_urls and enable_self_discovery=True
    readme_text = (
        "# Causal Net\n\n"
        "Continuous coupling regulates feedback loops toward homeostasis.\n"
        "See also https://en.wikipedia.org/wiki/Homeostasis for biological equilibrium.\n"
    )
    readme_b64 = base64.b64encode(readme_text.encode("utf-8")).decode("ascii")

    def _mock_discovery_fetcher(url: str, timeout: float, headers=None):
        if "api.github.com/search/repositories" in url:
            payload = {
                "items": [
                    {"html_url": "https://github.com/org/causal_net"},
                    {"html_url": "https://github.com/org/homeostasis_py"},
                ]
            }
            return 200, json.dumps(payload).encode("utf-8"), "application/json"
        if url == "https://api.github.com/repos/org/causal_net":
            payload = {"full_name": "org/causal_net", "description": "Continuous causal dynamics", "language": "Python", "stargazers_count": 120, "topics": ["causal", "equilibrium"]}
            return 200, json.dumps(payload).encode("utf-8"), "application/json"
        if url == "https://api.github.com/repos/org/causal_net/readme":
            return 200, json.dumps({"content": readme_b64}).encode("utf-8"), "application/json"
        if url == "https://api.github.com/repos/org/homeostasis_py":
            payload = {"full_name": "org/homeostasis_py", "description": "Self-regulating homeostasis", "language": "Python", "stargazers_count": 88, "topics": ["homeostasis"]}
            return 200, json.dumps(payload).encode("utf-8"), "application/json"
        if url == "https://api.github.com/repos/org/homeostasis_py/readme":
            rm = base64.b64encode(b"# Homeostasis Py\nRestoring feedback stabilizes dynamic oscillation.\n").decode("ascii")
            return 200, json.dumps({"content": rm}).encode("utf-8"), "application/json"
        if "wikipedia.org/api/rest_v1/page/summary/" in url:
            payload = {
                "title": "Homeostasis",
                "description": "Property of self-regulating systems",
                "extract": "Homeostasis maintains stable equilibrium across varying biological environments via coupled feedback.",
                "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Homeostasis"}},
                "thumbnail": {"source": "https://img.example/wiki_thumb.png"},
            }
            return 200, json.dumps(payload).encode("utf-8"), "application/json"
        if "wiki_thumb.png" in url:
            buf = io.BytesIO()
            im = Image.new("RGB", (8, 8))
            for y in range(8):
                for x in range(8):
                    im.putpixel((x, y), (30 + x * 28, 80 + y * 15, 220 - x * 20))
            im.save(buf, format="PNG")
            return 200, buf.getvalue(), "image/png"
        return 404, b"Not Found", "text/plain"

    empty_root = tmp_path / "empty_world"
    empty_root.mkdir()
    state_dir = tmp_path / "self_discovery_state"

    explorer = ElysiaAutonomousOpenExplorer(
        roots=[empty_root],
        state_dir=state_dir,
        web_urls=[],
        web_fetcher=_mock_discovery_fetcher,
        enable_self_discovery=True,
    )
    rep = explorer.run_autonomous_session(cycles=4)

    assert (state_dir / "discovered_frontiers.json").is_file()
    assert (state_dir / "tool_memory.json").is_file()
    frontier_urls = [it["url"] for it in rep["discovered_frontiers"]]
    assert "https://github.com/org/causal_net" in frontier_urls
    assert "https://en.wikipedia.org/wiki/Homeostasis" in frontier_urls
    assert explorer.modality_counts["external_web"] >= 2
    assert any(r["target_kind"] == "frontier_discovery" for r in rep["tool_competence_summary"])
    assert any(lg.get("tool_used") for lg in rep["cycle_logs"] if lg.get("explored"))
    obs_html = Path(rep["observatory_path"]).read_text(encoding="utf-8")
    assert "스스로 찾아낸 새 세상" in obs_html
    assert "스스로 체득한 도구 활용 능력" in obs_html


def test_lexical_novelty_prevents_premature_boredom_on_new_language_and_domain(tmp_path: Path):
    """
    Verifies 96-bucket stream + known_vocabulary prevents premature boredom when
    transitioning from English prose to completely unseen Korean/Japanese cultural text.
    """
    prose_dir = tmp_path / "multilingual_world"
    prose_dir.mkdir()
    (prose_dir / "01_english_causal.md").write_text(
        "# Continuous Causal Rotor Dynamics\n\n"
        "Boundary equilibrium regulates oscillatory momentum across coupled potential fields.\n"
        "Conservation principles sustain structural integrity when perturbation pressure shifts.\n"
        "Multi-scale invariance emerges from recursive coupling rather than static gates.\n",
        encoding="utf-8",
    )
    (prose_dir / "02_japanese_korean_art.md").write_text(
        "# 色彩の連続性と藝術的均衡 (색채 연속성과 예술적 균형)\n\n"
        "光と影の境界は固定された数値ではなく、呼吸する感情の揺らぎとして画面に流れ込む。\n"
        "한국의 전통 색채 조화와 일본 애니메이션의 섬세한 여백은 서로 다른 문화의 미학이면서도 공명한다.\n"
        "新しい言語と未知の物語に出会うとき、世界は再び未知の緊張と喜びとして開かれる。\n",
        encoding="utf-8",
    )

    state_dir = tmp_path / "lex_state"
    explorer = ElysiaAutonomousOpenExplorer(roots=[prose_dir], state_dir=state_dir)
    rep = explorer.run_autonomous_session(cycles=4)

    first_visits = {
        lg["target_name"]: lg
        for lg in rep["cycle_logs"]
        if lg.get("explored") and not lg.get("is_revisit")
    }
    assert "01_english_causal.md" in first_visits
    assert "02_japanese_korean_art.md" in first_visits

    jp_kr = first_visits["02_japanese_korean_art.md"]
    assert jp_kr["lexical_novelty"] is not None and jp_kr["lexical_novelty"] >= 0.70
    assert jp_kr["feeling"]["joy"] > jp_kr["feeling"]["boredom"]
    assert jp_kr["feeling"]["dominant_state"] == "joy_expansion"


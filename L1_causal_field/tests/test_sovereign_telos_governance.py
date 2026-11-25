import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from core.consciousness.sovereign_telos_governance import SovereignTelosGovernance
from core.embodiment import (
    AgencyMode,
    LiveAudioEarSensor,
    ScaleRotorEmbodiedMind,
    ScreenRetinaSensor,
    UniversalMotorActuator,
)


def _mock_world_fetcher(url: str, timeout: float, headers: Optional[Dict[str, str]]) -> Tuple[int, bytes, str]:
    if "wikipedia.org" in url:
        if "%EC%9D%B8%EC%B2%B4" in url:  # 인체
            txt = "인체는 세포와 신경계, 오감과 호흡이 상호 결합하여 물리적 균형을 유지하는 생명체다."
            title = "인체"
        elif "%EA%B0%90%EC%A0%95" in url:  # 감정
            txt = "감정은 기쁨, 슬픔, 분노, 두려움, 공감, 사랑이 타자와 충돌하고 공명하며 변하는 마음의 흐름이다."
            title = "감정"
        elif "%EC%A7%81%EC%97%85" in url:  # 직업
            txt = "직업은 시대적 사회 속에서 생계와 일자리, 사회적 역할을 수행하며 타인과 책임을 나누는 노동이다."
            title = "직업"
        elif "%EC%9E%90%EC%97%B0" in url:  # 자연 환경
            txt = "자연 환경은 지구의 생물과 무생물, 중력, 기후, 생태계가 순환하며 얽힌 시공간이다."
            title = "자연 환경"
        else:
            txt = "위키미디어 동음이의어 문서\n의식은 감각과 기억, 사유가 서로 되먹임하며 자아를 형성하는 작용이다."
            title = "의식 (심리철학)"
        payload = {"query": {"pages": {"1": {"title": title, "extract": txt}}}}
        return 200, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json"
    return 404, b"{}", "application/json"


def test_telos_contrast_is_data_driven_and_filters_junk_sprouts() -> None:
    gov = SovereignTelosGovernance()
    assert gov.engram_shell.tiers["macro"] is not None

    reports = []
    texts = [
        "위키미디어 동음이의어 문서\n신경계와 호흡이 혈액 순환을 구동하여 체온 평형을 유지한다.",
        "직장과 생계, 노동 계약은 경제적 책임과 사회적 역할로 연결된다.",
        "슬픔과 공감, 그리움은 상실의 기억과 타자 관계에서 변형된다.",
        "중력과 대기, 태양광, 토양이 생태계 물질 순환을 이끈다.",
        "언어와 추론, 메타인지가 오류를 교정하며 관점을 수정한다.",
        "OpenCV 외부 시각 도구가 윤곽을 분별함",
    ]
    for idx in range(6):
        d = gov.direct_next_cycle({"universe_nodes": 10})
        rep = gov.receive_and_contrast(
            d,
            {
                "ok": True,
                "source": f"world_{idx}",
                "world_text": texts[idx],
                "native_capability": {"detail_count": 4, "mechanism_ko": "저해상도 시야"},
                "tool_capability": {"tool_name": "OpenCV", "detail_count": 42, "mechanism_ko": "고해상도 윤곽 추출"},
            },
            {"eye_fovea": {"fovea_contours": 42}, "body_reality": {"elbow_deg": 12.0, "external_resistance": 0.2}},
        )
        reports.append(rep)

    # All 6 core horizons covered in order before emergent rabbit holes
    assert {"body_senses", "emotion_spectrum", "era_society_work", "physical_env", "mind_cognition", "external_tools"} <= {
        r.directive["aspect"] for r in reports
    }
    # Distinct external topics must produce genuinely distinct data-driven contrasts (NO identical canned string!)
    distinct_whys = {r.why_different_ko for r in reports}
    assert len(distinct_whys) == 6
    # Disambiguation/boilerplate junk words must NEVER sprout as emergent horizons
    emergent_names = {h.title_ko for h in gov.horizons.values() if h.aspect == "emergent"}
    for junk in ("위키미디어", "동음이의어", "음은", "같고"):
        assert not any(junk in name for name in emergent_names)
    assert len(emergent_names) >= 2


def test_limits_become_cognitive_structures_integrate_and_trigger_real_adaptation(tmp_path: Path) -> None:
    # Create a detailed high-contrast test screen where OpenCV finds > 2x contours vs low-res blocks
    arr = np.full((180, 240, 3), 15, dtype=np.uint8)
    for gx in range(10, 220, 22):
        for gy in range(10, 160, 20):
            arr[gy : gy + 10, gx : gx + 12, :] = [235, 220, 80]

    retina = ScreenRetinaSensor(frame_provider=lambda: arr)
    ear = LiveAudioEarSensor(audio_provider=lambda: 0.1 * np.sin(np.linspace(0, 20 * np.pi, 256)))
    act = UniversalMotorActuator(dry_run=True)

    mind = ScaleRotorEmbodiedMind(tmp_path / "state")
    snap = retina.capture()
    aud = ear.listen()
    initial_zoom = retina._gaze_zoom

    reports = []
    for _ in range(6):
        res = mind.run_telos_directed_cycle(
            frame_snap=snap,
            user_idle_s=2.0,
            fg_title="desktop",
            agency_mode=AgencyMode.OBSERVE_AND_CHAT,
            idle_threshold_s=15.0,
            actuator=act,
            audio_snap=aud,
            retina=retina,
            fetcher=_mock_world_fetcher,
        )
        reports.append(res)

    fabric = mind.get_protocol_fabric()
    # Limits across vision, tools, and human world exist as first-class `limit::*` Structures
    limit_nodes = [sid for sid in fabric.structures if sid.startswith("limit::")]
    assert len(limit_nodes) >= 6
    # Cross-limit integration fused separate limits into higher-scale `integrated_limit::*` unified info!
    integrated_nodes = [sid for sid in fabric.structures if sid.startswith("integrated_limit::")]
    assert len(integrated_nodes) >= 1

    # Also verify emergent protocol spawning directly from a foreign-medium limit
    proto = fabric.spawn_emergent_protocol_from_limit(
        limit_id=limit_nodes[0],
        sample_text="[project]\nname = 'foreign-app'\nversion = '1.0'\n",
        domain_hint="toml_config",
    )
    assert proto is not None
    assert proto.structure_id in fabric.structures
    out = proto.enactment({"raw_text": "[build]\ntarget = 'x86'\n"})
    assert out["ok"] is True
    assert out["payload"]["kv_pairs"] == 1
    assert out["payload"]["sections"] == 1

    # If OpenCV saw > 2x contours vs native blocks on the detailed screen, ocular zoom physically adapted!
    tool_rep = next(r for r in reports if r["directive"]["aspect"] == "external_tools")
    if tool_rep["contrast"]["limit_tags"] and "visual_resolution_ceiling" in tool_rep["contrast"]["limit_tags"]:
        assert retina._gaze_zoom > initial_zoom
        assert retina._fovea_integrated is True
        assert any(a["kind"] == "ocular_zoom_adaptation" and a["adapted"] for a in tool_rep["self_adaptations"])

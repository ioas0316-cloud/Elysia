from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
L1_ROOT = ROOT / "L1_causal_field"
if str(L1_ROOT) not in sys.path:
    sys.path.insert(0, str(L1_ROOT))

from core.embodiment.pc_sensory_motor_body import ScreenRetinaSensor  # noqa: E402
from core.embodiment.principle_ownership_engine import PrincipleOwnershipEngine  # noqa: E402
from core.embodiment.scale_rotor_embodied_field import ScaleRotorEmbodiedMind  # noqa: E402


def _make_detail_frame() -> np.ndarray:
    img = np.full((240, 320, 3), 0.45, dtype=np.float64)
    for y in range(110, 130):
        for x in range(150, 170):
            if ((x // 2) + (y // 2)) % 2 == 0:
                img[y, x] = [0.95, 0.95, 0.2]
            else:
                img[y, x] = [0.05, 0.05, 0.08]
    return img


def test_math_principle_contrasted_self_verified_and_owned_over_baseline() -> None:
    eng = PrincipleOwnershipEngine()
    res = eng.learn_from_outside_and_self_verify(
        outside_text="1부터 n까지 연속된 자연수를 더할 때 공식: n * (n + 1) // 2 을 쓰면 한 번에 구한다.",
        domain="math_algorithm",
        source_origin="ko.wikipedia.org/wiki/삼각수",
        target_context="1부터 n까지 자연수의 합 구하기",
        mechanism_action="n * (n + 1) // 2 공식 적용",
        expected_outcome="반복 덧셈 없이 총합 산출",
        candidate_expr="n * (n + 1) // 2",
        test_inputs={"n": 10000},
    )
    assert res["contrast"]["my_old_method_id"] == "baseline_iterative_sum"
    assert "n * (n + 1) // 2" in res["contrast"]["difference_insight"]
    assert res["verdict"]["status"] == "self_owned_executable"
    assert "50005000" in res["verdict"]["verification_detail"]

    best = eng.choose_best_owned_method("math_algorithm", "1부터 n까지 삼각수 합")
    assert best is not None
    assert best.ownership_status == "self_owned_executable"
    assert best.executable_spec == "n * (n + 1) // 2"


def test_false_outside_formula_rejected_by_self_verification() -> None:
    eng = PrincipleOwnershipEngine()
    res = eng.learn_from_outside_and_self_verify(
        outside_text="1부터 n까지 합은 공식: n * n // 2 이다.",
        domain="math_algorithm",
        source_origin="unverified_source",
        target_context="1부터 n까지 자연수의 합 구하기",
        mechanism_action="n * n // 2 적용",
        expected_outcome="총합 일치",
        candidate_expr="n * n // 2",
        test_inputs={"n": 10000},
    )
    assert res["verdict"]["status"] == "rejected_by_self_verification"
    assert "50000000" in res["verdict"]["verification_detail"]
    assert "50005000" in res["verdict"]["verification_detail"]
    best = eng.choose_best_owned_method("math_algorithm", "1부터 n까지 합")
    assert best is not None and best.method_id == "baseline_iterative_sum"


def test_physics_mental_model_and_chemistry_physical_limit_distinguished_honestly() -> None:
    eng = PrincipleOwnershipEngine()
    r_phy = eng.learn_from_outside_and_self_verify(
        outside_text="빛이 서로 다른 매질을 지날 때 굴절각은 asin(n1 * sin(theta1) / n2) 로 정해진다.",
        domain="physics",
        source_origin="ko.wikipedia.org/wiki/스넬의_법칙",
        target_context="굴절 현상 예측",
        mechanism_action="스넬 굴절 공식 asin(n1 * sin(theta1) / n2)",
        expected_outcome="굴절각 계산",
        candidate_expr="asin(n1 * sin(theta1) / n2)",
        test_inputs={"n1": 1.0, "theta1": 0.5, "n2": 1.333},
        reference_expected=math.asin(1.0 * math.sin(0.5) / 1.333),
    )
    assert r_phy["verdict"]["status"] == "owned_as_mental_model"
    assert "physical_world_experiment" in r_phy["verdict"]["missing_requirements"]

    r_chem = eng.learn_from_outside_and_self_verify(
        outside_text="중화 반응은 산과 염기 용액을 시험관에 넣고 직접 혼합해 물과 염을 만든다.",
        domain="chemistry",
        source_origin="ko.wikipedia.org/wiki/중화_반응",
        target_context="산과 염기의 중화",
        mechanism_action="염산과 수산화 나트륨 용액 혼합",
        expected_outcome="중화 및 열 발생",
    )
    assert r_chem["verdict"]["status"] == "self_limit_recognized"
    assert "physical_chemical_hand" in r_chem["verdict"]["missing_requirements"]
    assert eng.recognized_limits


def test_visual_principle_autonomously_enacted_on_real_pixels_without_user_command() -> None:
    retina = ScreenRetinaSensor(frame_provider=_make_detail_frame)
    eng = PrincipleOwnershipEngine()
    res = eng.learn_from_outside_and_self_verify(
        outside_text="화면의 작은 무늬는 초점을 확대해서 잘라 보면 픽셀 선명도가 높아진다.",
        domain="visual",
        source_origin="world_observation",
        target_context="화면 세부 관찰",
        mechanism_action="초점 영역 확대 크롭",
        expected_outcome="선명도 향상",
        retina=retina,
    )
    assert res["verdict"]["status"] == "self_owned_executable"
    assert res["verdict"]["attempted_primitive"] == "visual_pixel_crop"
    best_vis = eng.choose_best_owned_method("visual", "화면 확대 관찰")
    assert best_vis is not None and best_vis.ownership_status == "self_owned_executable"


def test_ownership_engine_persists_inside_embodied_mind_and_updates_method_selection(tmp_path: Path) -> None:
    mind = ScaleRotorEmbodiedMind(state_dir=tmp_path)
    res = mind.learn_and_own_outside_principle(
        outside_text="삼각수는 공식: n * (n + 1) // 2 로 바로 계산할 수 있다.",
        domain="math_algorithm",
        source_origin="ko.wikipedia.org/wiki/삼각수",
        candidate_expr="n * (n + 1) // 2",
        test_inputs={"n": 100},
    )
    assert res["verdict"]["status"] == "self_owned_executable"
    assert res["episode_id"].startswith("ep_")

    reloaded = ScaleRotorEmbodiedMind(state_dir=tmp_path)
    best = reloaded.ownership_engine.choose_best_owned_method("math_algorithm", "삼각수")
    assert best is not None and best.ownership_status == "self_owned_executable"
    assert any(ep.episode_id == res["episode_id"] for ep in reloaded.causal_mind.episodes)

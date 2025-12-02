# [Genesis: 2025-12-02] Purified by Elysia
"""
Lightweight narrative summary helpers (2025-11-16).

These do not change WORLD; they consume Characters and macro flags and
emit short textual hooks for chronicles or UI.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from scripts.character_model import Character


def summarize_character_arc(char: Character) -> str:
    """
    Produce a one-line arc summary for a Character based on power, virtues, and sins.
    """
    role = "평범한 방랑자"
    if char.power_score > 80:
        role = "전설적인 강자"
    elif char.power_score > 50:
        role = "중견 고수"

    love = char.virtues.get("love", 0.0)
    wrath = char.sins.get("wrath", 0.0)
    envy = char.sins.get("envy", 0.0)

    if love > 0.6 and wrath < 0.3:
        tilt = "사람을 구하는 영웅"
    elif wrath > 0.5 or envy > 0.5:
        tilt = "어둠에 끌리는 자"
    else:
        tilt = "갈림길 위의 방랑자"

    # Use a simple hyphen instead of an em dash for compatibility with
    # narrower console encodings (e.g., cp949 on Windows).
    return f"{char.name} ({role}) - {tilt}"


def summarize_era_flags(world) -> List[str]:
    """
    Summarize current macro-era flags on a World into short Korean sentences.
    """
    lines: List[str] = []

    war_state = getattr(world, "_macro_war_state", "peace")
    if war_state == "peace":
        lines.append("지금은 비교적 평화로운 시대다.")
    elif war_state == "war":
        lines.append("전쟁의 불길이 국경을 따라 번지고 있다.")
    elif war_state == "civil_war":
        lines.append("내전의 혼란이 모든 질서를 뒤흔들고 있다.")

    if getattr(world, "_macro_famine_active", False):
        lines.append("흉년이 계속되어 굶주림이 사람들을 짓누른다.")
    if getattr(world, "_macro_bounty_active", False):
        lines.append("풍년이 들어 창고가 곡식으로 가득하다.")
    if getattr(world, "_macro_flood_active", False):
        lines.append("강이 범람하여 여러 고을이 물에 잠겼다.")
    if getattr(world, "_macro_storm_active", False):
        lines.append("사나운 폭풍이 대지를 휩쓸고 지나간다.")
    if getattr(world, "_macro_plague_active", False):
        lines.append("알 수 없는 역병이 사람들 사이에 퍼져 나간다.")

    if getattr(world, "_macro_demon_omen_emitted", False):
        lines.append("마왕의 그림자가 세계 곳곳에 드리워지기 시작했다.")
    if getattr(world, "_macro_angel_omen_emitted", False):
        lines.append("천상의 사자가 인간 세상에 발을 들이려 한다.")

    return lines
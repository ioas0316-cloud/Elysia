"""
Character and relationship model for Elysia (2025-11-16)

This module does not depend on the running WORLD. It provides:
- A `Character` dataclass with origin, faction, class/role, tiers, and
  sin/virtue vectors.
- A `CharacterRelation` dataclass for pairwise relations
  (like/trust/grudge/respect/desire).
- Simple helper functions to:
  - map a continuous power_score to martial/knight/adventurer tiers
  - compute rough scores for master/hero/beauty style rankings
  - rank characters by a chosen scoring function

These helpers are intentionally lightweight so different simulations or
worlds can plug them in without tight coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple, Callable


SIN_KEYS = ["lust", "gluttony", "greed", "sloth", "wrath", "envy", "pride"]
VIRTUE_KEYS = [
    "love",
    "temperance",
    "generosity",
    "diligence",
    "patience",
    "kindness",
    "humility",
]


@dataclass
class Character:
    # Identity
    id: str
    name: str
    epithet: Optional[str] = None  # 별호 (e.g., "백야의 기사")

    # World identity
    origin_civ: str = "unknown"  # e.g., "HumanKingdom", "Ming", "DwarfHold"
    era: str = "unknown"  # e.g., "post_calamity", "gate_opening"
    birth_place_tags: List[str] = field(default_factory=list)  # ["capital", "border_town"]

    # Social position
    faction: Optional[str] = None  # e.g., "Wudang", "RoyalGuard", "AdventurerGuild"
    rank_in_faction: Optional[str] = None  # "disciple", "captain", "elder"

    # Role / class
    class_role: str = "commoner"  # "warrior", "swordsman", "mage", ...
    party_role: str = "flex"  # "tank", "dps", "support", "scout", ...

    # Power / tiers
    power_score: float = 0.0  # continuous; interpretation depends on world
    martial_tier: Optional[str] = None  # "삼류", "이류", "일류", "절정", "초절정", "화경"
    adventurer_rank: Optional[str] = None  # "흑철", "청동", "백은", "황금", "백금"
    knight_rank: Optional[str] = None  # "병사", "정예", "기사", ...

    # Inner tendencies
    sins: Dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in SIN_KEYS})
    virtues: Dict[str, float] = field(
        default_factory=lambda: {k: 0.0 for k in VIRTUE_KEYS}
    )
    personality_tags: List[str] = field(default_factory=list)  # ["온화", "충동적", ...]

    # Bookkeeping for world-side aggregates (optional)
    battles_fought: int = 0
    battles_won: int = 0
    major_quests_completed: int = 0
    civilians_saved: int = 0
    times_betrayed: int = 0
    times_betrayed_others: int = 0


@dataclass
class CharacterRelation:
    src_id: str
    dst_id: str
    like: float = 0.0  # -1..1
    trust: float = 0.0
    grudge: float = 0.0
    respect: float = 0.0
    desire: float = 0.0  # attraction / 집착 / 동경의 정도


# --- Party model -------------------------------------------------------------


@dataclass
class Party:
    """Minimal party model for WORLD-agnostic reasoning.

    WORLD implementations can extend this concept with concrete member objects.
    """

    id: str
    member_ids: List[str]
    # Optional, per-member role override; usually matches Character.party_role.
    roles: Dict[str, str] = field(default_factory=dict)
    origin_tags: List[str] = field(default_factory=list)  # e.g., ["심연관", "HumanKingdom"]
    notoriety: float = 0.0  # how widely known this party is


# --- Tier mapping helpers ----------------------------------------------------


def compute_martial_tier(power_score: float) -> str:
    """
    Map a continuous power_score to a 무협식 경지.

    The thresholds are intentionally simple and can be tuned per world.
    """
    s = max(0.0, power_score)
    if s < 20:
        return "삼류"
    if s < 40:
        return "이류"
    if s < 60:
        return "일류"
    if s < 80:
        return "절정"
    if s < 95:
        return "초절정"
    return "화경"


def compute_adventurer_rank(power_score: float) -> str:
    """
    Map a continuous power_score to 모험가 길드 위계.
    """
    s = max(0.0, power_score)
    if s < 15:
        return "흑철"
    if s < 35:
        return "청동"
    if s < 60:
        return "백은"
    if s < 85:
        return "황금"
    return "백금"


def compute_knight_rank(power_score: float) -> str:
    """
    Map a continuous power_score to 서방식 기사 위계.
    """
    s = max(0.0, power_score)
    if s < 10:
        return "병사"
    if s < 30:
        return "정예병"
    if s < 55:
        return "기사"
    if s < 80:
        return "기사단장"
        return "전설 용사"


def assign_tiers(
    char: Character,
    use_martial: bool = True,
    use_adventurer: bool = True,
    use_knight: bool = True,
) -> None:
    """Fill tier strings on a Character based on its power_score.

    Callers can disable tiers that are not relevant in the current world.
    """
    if use_martial:
        char.martial_tier = compute_martial_tier(char.power_score)
    if use_adventurer:
        char.adventurer_rank = compute_adventurer_rank(char.power_score)
    if use_knight:
        char.knight_rank = compute_knight_rank(char.power_score)


# --- Ranking helpers (십대고수/영웅/미녀 스코어링) ---------------------------


def score_master(char: Character) -> float:
    """
    Rough score for '십대고수' 스타일 랭킹.

    - 전투 기여와 power_score를 가장 크게 본다.
    - 기연/각성(major_quests_completed)도 가산점.
    """
    base = char.power_score
    battle_factor = (char.battles_won / max(1, char.battles_fought)) if char.battles_fought else 0.0
    quest_bonus = min(20.0, char.major_quests_completed * 2.0)
    return base + 10.0 * battle_factor + quest_bonus


def score_hero(char: Character) -> float:
    """
    Rough score for '십대영웅' 스타일 랭킹.

    - 공적(민간인 구출, 큰 퀘스트 완료)을 가장 크게 본다.
    - 전투력은 보조 지표.
    """
    public_good = char.civilians_saved * 0.5 + char.major_quests_completed * 3.0
    betrayal_penalty = char.times_betrayed_others * 5.0
    return public_good + 0.5 * char.power_score - betrayal_penalty


def score_beauty(char: Character, relations: Iterable[CharacterRelation]) -> float:
    """
    Rough score for '십대미녀/미남' 스타일 랭킹.

    - 외형/매력은 직접 모르므로, 대신:
      - 타인에게서 받는 호감/존경/욕망 평균을 사용.
      - 어느 정도 power_score/문화 지표를 보조로 본다.
    """
    incoming: List[Tuple[float, float, float]] = []
    for rel in relations:
        if rel.dst_id == char.id:
            incoming.append((rel.like, rel.respect, rel.desire))
    if incoming:
        avg_like = sum(x[0] for x in incoming) / len(incoming)
        avg_respect = sum(x[1] for x in incoming) / len(incoming)
        avg_desire = sum(x[2] for x in incoming) / len(incoming)
    else:
        avg_like = avg_respect = avg_desire = 0.0

    social_score = 30.0 * avg_like + 20.0 * avg_respect + 25.0 * avg_desire
    # 문화/예술 쪽은 virtue 기반으로 간단히 보정 (love/kindness 중심)
    love = char.virtues.get("love", 0.0)
    kindness = char.virtues.get("kindness", 0.0)
    culture_bonus = 10.0 * (love + kindness)

    return social_score + culture_bonus + 0.2 * char.power_score


# --- Jealousy / comparison helpers ------------------------------------------


def compute_status_score(char: Character) -> float:
    """Coarse 'status' metric for jealousy/comparison reasoning."""
    return (
        char.power_score
        + char.major_quests_completed * 3.0
        + char.civilians_saved * 0.5
    )


def compute_jealousy(
    a: Character,
    b: Character,
    relations: Iterable[CharacterRelation],
) -> float:
    """Estimate A가 B에게 느끼는 질투 정도.

    - envy 성향이 높을수록,
    - B의 status가 A보다 높을수록,
    - A가 B에게 가지는 desire가 클수록
      jealousy가 커지는 형태로 설계한다.
    """
    envy = max(0.0, a.sins.get("envy", 0.0))
    status_a = compute_status_score(a)
    status_b = compute_status_score(b)
    status_gap = max(0.0, status_b - status_a)

    desire = 0.0
    for rel in relations:
        if rel.src_id == a.id and rel.dst_id == b.id:
            desire = max(desire, rel.desire)

    # Normalize status gap to a soft 0..1 scale (assuming scores rarely exceed ~200).
    norm_gap = min(1.0, status_gap / 200.0)
    base = envy * (0.4 * norm_gap + 0.6 * desire)
    return max(0.0, min(1.0, base))


def rank_characters(
    chars: Iterable[Character],
    relations: Iterable[CharacterRelation],
    scorer: Callable[[Character], float],
    top_n: int = 10,
) -> List[Tuple[Character, float]]:
    """
    Rank characters using a given scoring function that depends only on Character.

    For beauty-like rankings that need relations, wrap scorer with a closure.
    """
    scored: List[Tuple[Character, float]] = []
    for ch in chars:
        scored.append((ch, scorer(ch)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def rank_beauties(
    chars: Iterable[Character],
    relations: Iterable[CharacterRelation],
    top_n: int = 10,
) -> List[Tuple[Character, float]]:
    """
    Convenience wrapper for 십대미녀/미남 랭킹.
    """
    rel_list = list(relations)
    scored: List[Tuple[Character, float]] = []
    for ch in chars:
        scored.append((ch, score_beauty(ch, rel_list)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# --- Demo (optional) ---------------------------------------------------------


def _demo() -> None:
    """Quick demo: build a few sample characters and print rankings."""
    # Sample characters around 심연관
    human_knight = Character(
        id="human_knight",
        name="롤랑",
        epithet="백야의 기사",
        origin_civ="HumanKingdom",
        era="gate_opening",
        birth_place_tags=["capital"],
        faction="RoyalGuard",
        rank_in_faction="captain",
        class_role="knight",
        party_role="tank",
        power_score=78.0,
        virtues={"love": 0.4, "temperance": 0.6, "generosity": 0.3, "diligence": 0.8, "patience": 0.5, "kindness": 0.4, "humility": 0.3},
        sins={"lust": 0.1, "gluttony": 0.2, "greed": 0.3, "sloth": 0.1, "wrath": 0.4, "envy": 0.2, "pride": 0.5},
        battles_fought=40,
        battles_won=35,
        major_quests_completed=5,
        civilians_saved=120,
    )

    wuxia_disciple = Character(
        id="wuxia_disciple",
        name="이청",
        epithet="청풍검객",
        origin_civ="Ming",
        era="gate_opening",
        birth_place_tags=["mountain_temple"],
        faction="Wudang",
        rank_in_faction="inner_disciple",
        class_role="swordsman",
        party_role="dps",
        power_score=72.0,
        virtues={"love": 0.5, "temperance": 0.7, "generosity": 0.4, "diligence": 0.9, "patience": 0.7, "kindness": 0.5, "humility": 0.4},
        sins={"lust": 0.1, "gluttony": 0.1, "greed": 0.2, "sloth": 0.1, "wrath": 0.3, "envy": 0.2, "pride": 0.3},
        battles_fought=30,
        battles_won=27,
        major_quests_completed=4,
        civilians_saved=60,
    )

    dwarf_guide = Character(
        id="dwarf_guide",
        name="브란 스톤하트",
        epithet="석심",
        origin_civ="DwarfHold",
        era="gate_opening",
        birth_place_tags=["deep_mine"],
        faction="DeepHoldCouncil",
        rank_in_faction="navigator",
        class_role="guide",
        party_role="support",
        power_score=50.0,
        virtues={"love": 0.3, "temperance": 0.4, "generosity": 0.3, "diligence": 0.8, "patience": 0.6, "kindness": 0.3, "humility": 0.5},
        sins={"lust": 0.1, "gluttony": 0.4, "greed": 0.3, "sloth": 0.1, "wrath": 0.3, "envy": 0.2, "pride": 0.2},
        battles_fought=15,
        battles_won=10,
        major_quests_completed=3,
        civilians_saved=20,
    )

    elf_mage = Character(
        id="elf_mage",
        name="아리엘 린드엘",
        epithet="월화",
        origin_civ="ElfDomain",
        era="gate_opening",
        birth_place_tags=["forest", "moon_temple"],
        faction="MoonPalace",
        rank_in_faction="magus",
        class_role="mage",
        party_role="support",
        power_score=70.0,
        virtues={"love": 0.6, "temperance": 0.7, "generosity": 0.4, "diligence": 0.7, "patience": 0.8, "kindness": 0.7, "humility": 0.4},
        sins={"lust": 0.2, "gluttony": 0.1, "greed": 0.2, "sloth": 0.1, "wrath": 0.2, "envy": 0.2, "pride": 0.3},
        battles_fought=20,
        battles_won=18,
        major_quests_completed=6,
        civilians_saved=80,
    )

    chars = [human_knight, wuxia_disciple, dwarf_guide, elf_mage]
    for ch in chars:
        assign_tiers(ch)

    # Sample relations (who feels what about whom)
    relations = [
        CharacterRelation(src_id="wuxia_disciple", dst_id="elf_mage", like=0.6, trust=0.5, respect=0.7, desire=0.5),
        CharacterRelation(src_id="elf_mage", dst_id="wuxia_disciple", like=0.5, trust=0.6, respect=0.6, desire=0.4),
        CharacterRelation(src_id="human_knight", dst_id="elf_mage", like=0.5, trust=0.5, respect=0.8, desire=0.3),
        CharacterRelation(src_id="dwarf_guide", dst_id="human_knight", like=0.4, trust=0.7, respect=0.6, desire=0.0),
    ]

    # Rankings
    masters = rank_characters(chars, relations, score_master, top_n=3)
    heroes = rank_characters(chars, relations, score_hero, top_n=3)
    beauties = rank_beauties(chars, relations, top_n=3)

    print("[십대고수 후보]")
    for ch, sc in masters:
        print(f"- {ch.name} ({ch.epithet or ''}) | power={ch.power_score:.1f} | score={sc:.1f} | 경지={ch.martial_tier}")

    print("\n[십대영웅 후보]")
    for ch, sc in heroes:
        print(f"- {ch.name} ({ch.epithet or ''}) | 공적 score={sc:.1f}")

    print("\n[십대미녀/미남 후보]")
    for ch, sc in beauties:
        print(f"- {ch.name} ({ch.epithet or ''}) | beauty score={sc:.1f}")


if __name__ == "__main__":
    _demo()

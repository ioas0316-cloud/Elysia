# [Genesis: 2025-12-02] Purified by Elysia
"""
Job definitions and registry for Elysia (2025-11-16).

This module encodes the domain.class.archetype structure from
CORE_12_PEOPLE_AND_CIVILIZATION_TIERS and provides a small set of
canonical jobs that other layers can reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Job:
    id: str
    domain: str
    job_class: str
    archetype: str
    name: str
    tier_base: int
    mobility: float
    risk: float
    training_years: float
    prestige_base: float
    field_affinity: Dict[str, float]
    grade: str = "green"
    virtue_alignment: Dict[str, float] = None
    sin_alignment: Dict[str, float] = None


JOB_REGISTRY: Dict[str, Job] = {}
PROMOTION_TREE: Dict[str, List[str]] = {}


def register_job(job: Job) -> None:
    if job.id in JOB_REGISTRY:
        raise ValueError(f"Duplicate job id: {job.id}")
    # Normalise optional dict fields to empty dicts so callers can rely on them.
    if job.virtue_alignment is None:
        object.__setattr__(job, "virtue_alignment", {})
    if job.sin_alignment is None:
        object.__setattr__(job, "sin_alignment", {})
    JOB_REGISTRY[job.id] = job


def register_promotion_path(path: List[str]) -> None:
    """
    Register a linear promotion path such as:
    ["martial.soldier.guard", "martial.soldier.knight", "govern.noble.lord"].

    Calling this multiple times with shared prefixes creates branching trees.
    """
    if len(path) < 2:
        return
    for jid in path:
        if jid not in JOB_REGISTRY:
            raise ValueError(f"Promotion path references unknown job id: {jid}")
    for current_id, next_id in zip(path, path[1:]):
        next_list = PROMOTION_TREE.setdefault(current_id, [])
        if next_id not in next_list:
            next_list.append(next_id)


def split_job_id(job_id: str) -> Tuple[str, str, str]:
    parts = job_id.split(".")
    if len(parts) != 3:
        return "unknown", "unknown", job_id
    return parts[0], parts[1], parts[2]


def get_job(job_id: str) -> Optional[Job]:
    return JOB_REGISTRY.get(job_id)


def jobs_in_domain(domain: str) -> List[Job]:
    return [job for job in JOB_REGISTRY.values() if job.domain == domain]


def _ja(field_value: float) -> float:
    return max(-1.0, min(1.0, field_value))


# --- Civilization / everyday jobs (Tier 3 aggregate mapping) -----------------
# These jobs are used mainly for macro civ sims and Godot prototypes.

register_job(
    Job(
        id="agri.peasant.farmer",
        domain="agri",
        job_class="peasant",
        archetype="farmer",
        name="농부",
        tier_base=1,
        mobility=0.2,
        risk=0.2,
        training_years=1.0,
        prestige_base=0.2,
        field_affinity={
            "value_mass_delta": _ja(0.05),
            "will_tension_delta": _ja(-0.03),
            "threat_delta": _ja(-0.02),
            "faith_delta": _ja(0.0),
        },
        virtue_alignment={"diligence": 0.6, "temperance": 0.4, "patience": 0.5},
        sin_alignment={"sloth": -0.3, "greed": 0.1},
    )
)

register_job(
    Job(
        id="agri.peasant.hunter",
        domain="agri",
        job_class="peasant",
        archetype="hunter",
        name="사냥꾼",
        tier_base=1,
        mobility=0.6,
        risk=0.5,
        training_years=1.5,
        prestige_base=0.25,
        field_affinity={
            "value_mass_delta": _ja(0.03),
            "will_tension_delta": _ja(0.02),
            "threat_delta": _ja(0.05),
            "faith_delta": _ja(0.0),
        },
    )
)

register_job(
    Job(
        id="craft.artisan.blacksmith",
        domain="craft",
        job_class="artisan",
        archetype="blacksmith",
        name="대장장이",
        tier_base=2,
        mobility=0.3,
        risk=0.3,
        training_years=4.0,
        prestige_base=0.4,
        field_affinity={
            "value_mass_delta": _ja(0.06),
            "will_tension_delta": _ja(0.01),
            "threat_delta": _ja(0.04),
            "faith_delta": _ja(0.0),
        },
    )
)

register_job(
    Job(
        id="craft.artisan.carpenter",
        domain="craft",
        job_class="artisan",
        archetype="carpenter",
        name="목수",
        tier_base=2,
        mobility=0.4,
        risk=0.25,
        training_years=3.0,
        prestige_base=0.35,
        field_affinity={
            "value_mass_delta": _ja(0.05),
            "will_tension_delta": _ja(0.0),
            "threat_delta": _ja(0.0),
            "faith_delta": _ja(0.0),
        },
    )
)

register_job(
    Job(
        id="trade.merchant.peddler",
        domain="trade",
        job_class="merchant",
        archetype="peddler",
        name="행상인",
        tier_base=1,
        mobility=0.8,
        risk=0.5,
        training_years=1.0,
        prestige_base=0.3,
        field_affinity={
            "value_mass_delta": _ja(0.04),
            "will_tension_delta": _ja(0.03),
            "threat_delta": _ja(0.03),
            "faith_delta": _ja(0.0),
        },
    )
)

register_job(
    Job(
        id="trade.merchant.caravan_master",
        domain="trade",
        job_class="merchant",
        archetype="caravan_master",
        name="대상단장",
        tier_base=2,
        mobility=0.9,
        risk=0.6,
        training_years=4.0,
        prestige_base=0.55,
        field_affinity={
            "value_mass_delta": _ja(0.07),
            "will_tension_delta": _ja(0.05),
            "threat_delta": _ja(0.05),
            "faith_delta": _ja(0.0),
        },
    )
)

register_job(
    Job(
        id="martial.soldier.guard",
        domain="martial",
        job_class="soldier",
        archetype="guard",
        name="경비병",
        tier_base=1,
        mobility=0.4,
        risk=0.6,
        training_years=2.0,
        prestige_base=0.35,
        field_affinity={
            "value_mass_delta": _ja(0.03),
            "will_tension_delta": _ja(0.05),
            "threat_delta": _ja(0.08),
            "faith_delta": _ja(0.0),
        },
    )
)

register_job(
    Job(
        id="martial.soldier.knight",
        domain="martial",
        job_class="soldier",
        archetype="knight",
        name="기사",
        tier_base=2,
        mobility=0.6,
        risk=0.7,
        training_years=6.0,
        prestige_base=0.75,
        field_affinity={
            "value_mass_delta": _ja(0.06),
            "will_tension_delta": _ja(0.08),
            "threat_delta": _ja(0.1),
            "faith_delta": _ja(0.02),
        },
    )
)

register_job(
    Job(
        id="faith.priest.acolyte",
        domain="faith",
        job_class="priest",
        archetype="acolyte",
        name="신전 시종",
        tier_base=1,
        mobility=0.3,
        risk=0.2,
        training_years=2.0,
        prestige_base=0.3,
        field_affinity={
            "value_mass_delta": _ja(0.03),
            "will_tension_delta": _ja(-0.02),
            "threat_delta": _ja(-0.01),
            "faith_delta": _ja(0.06),
        },
    )
)

register_job(
    Job(
        id="faith.priest.monk",
        domain="faith",
        job_class="priest",
        archetype="monk",
        name="수도승",
        tier_base=2,
        mobility=0.4,
        risk=0.3,
        training_years=5.0,
        prestige_base=0.5,
        field_affinity={
            "value_mass_delta": _ja(0.04),
            "will_tension_delta": _ja(-0.05),
            "threat_delta": _ja(-0.02),
            "faith_delta": _ja(0.1),
        },
    )
)

register_job(
    Job(
        id="knowledge.scholar.scribe",
        domain="knowledge",
        job_class="scholar",
        archetype="scribe",
        name="서기",
        tier_base=1,
        mobility=0.3,
        risk=0.15,
        training_years=3.0,
        prestige_base=0.4,
        field_affinity={
            "value_mass_delta": _ja(0.04),
            "will_tension_delta": _ja(0.0),
            "threat_delta": _ja(0.0),
            "faith_delta": _ja(0.02),
        },
    )
)

register_job(
    Job(
        id="knowledge.scholar.sage",
        domain="knowledge",
        job_class="scholar",
        archetype="sage",
        name="현자",
        tier_base=3,
        mobility=0.4,
        risk=0.2,
        training_years=10.0,
        prestige_base=0.9,
        field_affinity={
            "value_mass_delta": _ja(0.08),
            "will_tension_delta": _ja(0.02),
            "threat_delta": _ja(0.01),
            "faith_delta": _ja(0.05),
        },
    )
)

register_job(
    Job(
        id="govern.noble.steward",
        domain="govern",
        job_class="noble",
        archetype="steward",
        name="청지기",
        tier_base=2,
        mobility=0.3,
        risk=0.4,
        training_years=5.0,
        prestige_base=0.6,
        field_affinity={
            "value_mass_delta": _ja(0.06),
            "will_tension_delta": _ja(0.03),
            "threat_delta": _ja(0.02),
            "faith_delta": _ja(0.01),
        },
    )
)

register_job(
    Job(
        id="govern.noble.lord",
        domain="govern",
        job_class="noble",
        archetype="lord",
        name="영주",
        tier_base=3,
        mobility=0.4,
        risk=0.5,
        training_years=8.0,
        prestige_base=0.95,
        field_affinity={
            "value_mass_delta": _ja(0.09),
            "will_tension_delta": _ja(0.06),
            "threat_delta": _ja(0.04),
            "faith_delta": _ja(0.03),
        },
    )
)

register_job(
    Job(
        id="art.artisan.bard",
        domain="art",
        job_class="artisan",
        archetype="bard",
        name="음유시인",
        tier_base=1,
        mobility=0.7,
        risk=0.3,
        training_years=2.0,
        prestige_base=0.35,
        field_affinity={
            "value_mass_delta": _ja(0.05),
            "will_tension_delta": _ja(-0.01),
            "threat_delta": _ja(0.0),
            "faith_delta": _ja(0.01),
        },
    )
)

register_job(
    Job(
        id="adventure.adventurer.ranger",
        domain="adventure",
        job_class="adventurer",
        archetype="ranger",
        name="레인저",
        tier_base=1,
        mobility=0.9,
        risk=0.7,
        training_years=3.0,
        prestige_base=0.5,
        field_affinity={
            "value_mass_delta": _ja(0.05),
            "will_tension_delta": _ja(0.04),
            "threat_delta": _ja(0.08),
            "faith_delta": _ja(0.0),
        },
    )
)

register_job(
    Job(
        id="adventure.adventurer.dungeon_delver",
        domain="adventure",
        job_class="adventurer",
        archetype="dungeon_delver",
        name="던전 탐험가",
        tier_base=2,
        mobility=1.0,
        risk=0.9,
        training_years=4.0,
        prestige_base=0.7,
        field_affinity={
            "value_mass_delta": _ja(0.07),
            "will_tension_delta": _ja(0.08),
            "threat_delta": _ja(0.12),
            "faith_delta": _ja(0.0),
        },
    )
)

# --- Adventurer job tree (라그나로크/랑그릿사 스타일 전직 트리) -------------

# Base 1차: 초보자
register_job(
    Job(
        id="adventure.novice.novice",
        domain="adventure",
        job_class="novice",
        archetype="novice",
        name="초보자",
        tier_base=1,
        mobility=0.5,
        risk=0.2,
        training_years=0.5,
        prestige_base=0.1,
        field_affinity={
            "value_mass_delta": _ja(0.01),
            "will_tension_delta": _ja(0.0),
            "threat_delta": _ja(0.0),
            "faith_delta": _ja(0.0),
        },
    )
)

# 1차 분기: 검사 / 마법사 / 궁수 / 도둑 / 복사 / 상인
register_job(
    Job(
        id="martial.soldier.swordsman",
        domain="martial",
        job_class="soldier",
        archetype="swordsman",
        name="검사",
        tier_base=1,
        mobility=0.6,
        risk=0.5,
        training_years=1.0,
        prestige_base=0.3,
        field_affinity={
            "value_mass_delta": _ja(0.03),
            "will_tension_delta": _ja(0.04),
            "threat_delta": _ja(0.06),
            "faith_delta": _ja(0.0),
        },
    )
)

register_job(
    Job(
        id="knowledge.scholar.mage",
        domain="knowledge",
        job_class="scholar",
        archetype="mage",
        name="마법사",
        tier_base=1,
        mobility=0.5,
        risk=0.5,
        training_years=1.5,
        prestige_base=0.35,
        field_affinity={
            "value_mass_delta": _ja(0.04),
            "will_tension_delta": _ja(0.02),
            "threat_delta": _ja(0.05),
            "faith_delta": _ja(0.01),
        },
    )
)

register_job(
    Job(
        id="adventure.adventurer.archer",
        domain="adventure",
        job_class="adventurer",
        archetype="archer",
        name="궁수",
        tier_base=1,
        mobility=0.8,
        risk=0.5,
        training_years=1.0,
        prestige_base=0.3,
        field_affinity={
            "value_mass_delta": _ja(0.03),
            "will_tension_delta": _ja(0.03),
            "threat_delta": _ja(0.06),
            "faith_delta": _ja(0.0),
        },
    )
)

register_job(
    Job(
        id="adventure.adventurer.thief",
        domain="adventure",
        job_class="adventurer",
        archetype="thief",
        name="도둑",
        tier_base=1,
        mobility=0.9,
        risk=0.6,
        training_years=1.0,
        prestige_base=0.25,
        field_affinity={
            "value_mass_delta": _ja(0.02),
            "will_tension_delta": _ja(0.04),
            "threat_delta": _ja(0.05),
            "faith_delta": _ja(0.0),
        },
    )
)

register_job(
    Job(
        id="faith.priest.acolyte_combat",
        domain="faith",
        job_class="priest",
        archetype="acolyte_combat",
        name="복사",
        tier_base=1,
        mobility=0.5,
        risk=0.4,
        training_years=1.0,
        prestige_base=0.3,
        field_affinity={
            "value_mass_delta": _ja(0.03),
            "will_tension_delta": _ja(-0.01),
            "threat_delta": _ja(0.01),
            "faith_delta": _ja(0.05),
        },
    )
)

register_job(
    Job(
        id="trade.merchant.merchant",
        domain="trade",
        job_class="merchant",
        archetype="merchant",
        name="상인",
        tier_base=1,
        mobility=0.6,
        risk=0.4,
        training_years=1.0,
        prestige_base=0.3,
        field_affinity={
            "value_mass_delta": _ja(0.04),
            "will_tension_delta": _ja(0.03),
            "threat_delta": _ja(0.03),
            "faith_delta": _ja(0.0),
        },
    )
)

# 2차: 기사 / 크루세이더, 위자드 / 세이지, 헌터 / 로그, 어새신 / 로우, 프리스트 / 몽크,
#      대장장이 / 연금술사

register_job(
    Job(
        id="martial.soldier.knight_2",
        domain="martial",
        job_class="soldier",
        archetype="knight_2",
        name="기사",
        tier_base=2,
        mobility=0.7,
        risk=0.7,
        training_years=3.0,
        prestige_base=0.6,
        field_affinity={
            "value_mass_delta": _ja(0.05),
            "will_tension_delta": _ja(0.07),
            "threat_delta": _ja(0.09),
            "faith_delta": _ja(0.01),
        },
    )
)

register_job(
    Job(
        id="martial.soldier.crusader_2",
        domain="martial",
        job_class="soldier",
        archetype="crusader_2",
        name="크루세이더",
        tier_base=2,
        mobility=0.6,
        risk=0.7,
        training_years=3.0,
        prestige_base=0.6,
        field_affinity={
            "value_mass_delta": _ja(0.05),
            "will_tension_delta": _ja(0.06),
            "threat_delta": _ja(0.08),
            "faith_delta": _ja(0.03),
        },
    )
)

register_job(
    Job(
        id="knowledge.scholar.wizard",
        domain="knowledge",
        job_class="scholar",
        archetype="wizard",
        name="위자드",
        tier_base=2,
        mobility=0.5,
        risk=0.7,
        training_years=3.0,
        prestige_base=0.65,
        field_affinity={
            "value_mass_delta": _ja(0.06),
            "will_tension_delta": _ja(0.03),
            "threat_delta": _ja(0.1),
            "faith_delta": _ja(0.02),
        },
    )
)

register_job(
    Job(
        id="knowledge.scholar.sage_2",
        domain="knowledge",
        job_class="scholar",
        archetype="sage_2",
        name="세이지",
        tier_base=2,
        mobility=0.5,
        risk=0.6,
        training_years=3.0,
        prestige_base=0.6,
        field_affinity={
            "value_mass_delta": _ja(0.05),
            "will_tension_delta": _ja(0.02),
            "threat_delta": _ja(0.07),
            "faith_delta": _ja(0.03),
        },
    )
)

register_job(
    Job(
        id="adventure.adventurer.hunter",
        domain="adventure",
        job_class="adventurer",
        archetype="hunter",
        name="헌터",
        tier_base=2,
        mobility=0.9,
        risk=0.7,
        training_years=2.0,
        prestige_base=0.55,
        field_affinity={
            "value_mass_delta": _ja(0.05),
            "will_tension_delta": _ja(0.05),
            "threat_delta": _ja(0.1),
            "faith_delta": _ja(0.0),
        },
    )
)

register_job(
    Job(
        id="adventure.adventurer.rogue",
        domain="adventure",
        job_class="adventurer",
        archetype="rogue",
        name="로그",
        tier_base=2,
        mobility=0.9,
        risk=0.7,
        training_years=2.0,
        prestige_base=0.5,
        field_affinity={
            "value_mass_delta": _ja(0.04),
            "will_tension_delta": _ja(0.06),
            "threat_delta": _ja(0.08),
            "faith_delta": _ja(0.0),
        },
    )
)

register_job(
    Job(
        id="adventure.adventurer.assassin",
        domain="adventure",
        job_class="adventurer",
        archetype="assassin",
        name="어새신",
        tier_base=2,
        mobility=1.0,
        risk=0.8,
        training_years=2.0,
        prestige_base=0.55,
        field_affinity={
            "value_mass_delta": _ja(0.04),
            "will_tension_delta": _ja(0.07),
            "threat_delta": _ja(0.11),
            "faith_delta": _ja(-0.02),
        },
    )
)

register_job(
    Job(
        id="faith.priest.priest",
        domain="faith",
        job_class="priest",
        archetype="priest",
        name="프리스트",
        tier_base=2,
        mobility=0.6,
        risk=0.5,
        training_years=2.5,
        prestige_base=0.6,
        field_affinity={
            "value_mass_delta": _ja(0.06),
            "will_tension_delta": _ja(-0.04),
            "threat_delta": _ja(-0.02),
            "faith_delta": _ja(0.12),
        },
    )
)

register_job(
    Job(
        id="faith.priest.monk_combat",
        domain="faith",
        job_class="priest",
        archetype="monk_combat",
        name="몽크",
        tier_base=2,
        mobility=0.8,
        risk=0.7,
        training_years=2.5,
        prestige_base=0.55,
        field_affinity={
            "value_mass_delta": _ja(0.05),
            "will_tension_delta": _ja(0.03),
            "threat_delta": _ja(0.08),
            "faith_delta": _ja(0.06),
        },
    )
)

register_job(
    Job(
        id="craft.artisan.blacksmith_2",
        domain="craft",
        job_class="artisan",
        archetype="blacksmith_2",
        name="대장장이",
        tier_base=2,
        mobility=0.5,
        risk=0.4,
        training_years=3.0,
        prestige_base=0.55,
        field_affinity={
            "value_mass_delta": _ja(0.05),
            "will_tension_delta": _ja(0.02),
            "threat_delta": _ja(0.04),
            "faith_delta": _ja(0.0),
        },
    )
)

register_job(
    Job(
        id="craft.artisan.alchemist",
        domain="craft",
        job_class="artisan",
        archetype="alchemist",
        name="연금술사",
        tier_base=2,
        mobility=0.5,
        risk=0.5,
        training_years=3.0,
        prestige_base=0.55,
        field_affinity={
            "value_mass_delta": _ja(0.05),
            "will_tension_delta": _ja(0.01),
            "threat_delta": _ja(0.03),
            "faith_delta": _ja(0.02),
        },
    )
)

# 3차/4차: 하이클래스/전설 직업들을 단순화해서 2단계만 더 쌓는다.

register_job(
    Job(
        id="martial.soldier.lord_knight",
        domain="martial",
        job_class="soldier",
        archetype="lord_knight",
        name="로드 나이트",
        tier_base=3,
        mobility=0.8,
        risk=0.8,
        training_years=5.0,
        prestige_base=0.8,
        field_affinity={
            "value_mass_delta": _ja(0.08),
            "will_tension_delta": _ja(0.09),
            "threat_delta": _ja(0.13),
            "faith_delta": _ja(0.02),
        },
    )
)

register_job(
    Job(
        id="martial.soldier.paladin",
        domain="martial",
        job_class="soldier",
        archetype="paladin",
        name="팔라딘",
        tier_base=3,
        mobility=0.7,
        risk=0.7,
        training_years=5.0,
        prestige_base=0.8,
        field_affinity={
            "value_mass_delta": _ja(0.07),
            "will_tension_delta": _ja(0.07),
            "threat_delta": _ja(0.1),
            "faith_delta": _ja(0.08),
        },
    )
)

register_job(
    Job(
        id="knowledge.scholar.archmage",
        domain="knowledge",
        job_class="scholar",
        archetype="archmage",
        name="아크메이지",
        tier_base=3,
        mobility=0.5,
        risk=0.85,
        training_years=5.0,
        prestige_base=0.85,
        field_affinity={
            "value_mass_delta": _ja(0.09),
            "will_tension_delta": _ja(0.04),
            "threat_delta": _ja(0.15),
            "faith_delta": _ja(0.03),
        },
    )
)

register_job(
    Job(
        id="knowledge.scholar.elementalist",
        domain="knowledge",
        job_class="scholar",
        archetype="elementalist",
        name="엘리멘탈리스트",
        tier_base=3,
        mobility=0.6,
        risk=0.75,
        training_years=5.0,
        prestige_base=0.8,
        field_affinity={
            "value_mass_delta": _ja(0.08),
            "will_tension_delta": _ja(0.04),
            "threat_delta": _ja(0.13),
            "faith_delta": _ja(0.04),
        },
    )
)

register_job(
    Job(
        id="adventure.adventurer.sniper",
        domain="adventure",
        job_class="adventurer",
        archetype="sniper",
        name="스나이퍼",
        tier_base=3,
        mobility=0.9,
        risk=0.8,
        training_years=4.0,
        prestige_base=0.8,
        field_affinity={
            "value_mass_delta": _ja(0.07),
            "will_tension_delta": _ja(0.06),
            "threat_delta": _ja(0.14),
            "faith_delta": _ja(0.0),
        },
    )
)

register_job(
    Job(
        id="adventure.adventurer.shadow_assassin",
        domain="adventure",
        job_class="adventurer",
        archetype="shadow_assassin",
        name="섀도 어새신",
        tier_base=3,
        mobility=1.0,
        risk=0.9,
        training_years=4.0,
        prestige_base=0.8,
        field_affinity={
            "value_mass_delta": _ja(0.07),
            "will_tension_delta": _ja(0.08),
            "threat_delta": _ja(0.16),
            "faith_delta": _ja(-0.03),
        },
    )
)

register_job(
    Job(
        id="faith.priest.high_priest",
        domain="faith",
        job_class="priest",
        archetype="high_priest",
        name="하이 프리스트",
        tier_base=3,
        mobility=0.6,
        risk=0.6,
        training_years=5.0,
        prestige_base=0.85,
        field_affinity={
            "value_mass_delta": _ja(0.09),
            "will_tension_delta": _ja(-0.06),
            "threat_delta": _ja(-0.03),
            "faith_delta": _ja(0.16),
        },
    )
)

register_job(
    Job(
        id="faith.priest.hierophant",
        domain="faith",
        job_class="priest",
        archetype="hierophant",
        name="하이에로펀트",
        tier_base=3,
        mobility=0.7,
        risk=0.7,
        training_years=5.0,
        prestige_base=0.8,
        field_affinity={
            "value_mass_delta": _ja(0.08),
            "will_tension_delta": _ja(-0.04),
            "threat_delta": _ja(0.02),
            "faith_delta": _ja(0.14),
        },
    )
)

register_job(
    Job(
        id="craft.artisan.master_smith",
        domain="craft",
        job_class="artisan",
        archetype="master_smith",
        name="마스터 스미스",
        tier_base=3,
        mobility=0.5,
        risk=0.6,
        training_years=5.0,
        prestige_base=0.8,
        field_affinity={
            "value_mass_delta": _ja(0.08),
            "will_tension_delta": _ja(0.03),
            "threat_delta": _ja(0.05),
            "faith_delta": _ja(0.02),
        },
    )
)

register_job(
    Job(
        id="craft.artisan.chemist",
        domain="craft",
        job_class="artisan",
        archetype="chemist",
        name="케미스트",
        tier_base=3,
        mobility=0.5,
        risk=0.6,
        training_years=5.0,
        prestige_base=0.8,
        field_affinity={
            "value_mass_delta": _ja(0.08),
            "will_tension_delta": _ja(0.02),
            "threat_delta": _ja(0.05),
            "faith_delta": _ja(0.03),
        },
    )
)

# 5차(최종 전직) 예시: 전설 직업 한 단계 더
register_job(
    Job(
        id="martial.soldier.dragon_knight",
        domain="martial",
        job_class="soldier",
        archetype="dragon_knight",
        name="드래곤 나이트",
        tier_base=4,
        mobility=0.9,
        risk=0.9,
        training_years=7.0,
        prestige_base=0.95,
        field_affinity={
            "value_mass_delta": _ja(0.11),
            "will_tension_delta": _ja(0.1),
            "threat_delta": _ja(0.18),
            "faith_delta": _ja(0.03),
        },
    )
)

register_job(
    Job(
        id="martial.soldier.holy_crusader",
        domain="martial",
        job_class="soldier",
        archetype="holy_crusader",
        name="홀리 크루세이더",
        tier_base=4,
        mobility=0.8,
        risk=0.85,
        training_years=7.0,
        prestige_base=0.95,
        field_affinity={
            "value_mass_delta": _ja(0.1),
            "will_tension_delta": _ja(0.08),
            "threat_delta": _ja(0.15),
            "faith_delta": _ja(0.12),
        },
    )
)

register_job(
    Job(
        id="knowledge.scholar.arcane_lord",
        domain="knowledge",
        job_class="scholar",
        archetype="arcane_lord",
        name="아케인 로드",
        tier_base=4,
        mobility=0.5,
        risk=0.95,
        training_years=7.0,
        prestige_base=0.96,
        field_affinity={
            "value_mass_delta": _ja(0.12),
            "will_tension_delta": _ja(0.05),
            "threat_delta": _ja(0.2),
            "faith_delta": _ja(0.04),
        },
    )
)

register_job(
    Job(
        id="knowledge.scholar.celestial_sage",
        domain="knowledge",
        job_class="scholar",
        archetype="celestial_sage",
        name="천상의 현자",
        tier_base=4,
        mobility=0.5,
        risk=0.9,
        training_years=7.0,
        prestige_base=0.95,
        field_affinity={
            "value_mass_delta": _ja(0.11),
            "will_tension_delta": _ja(0.06),
            "threat_delta": _ja(0.17),
            "faith_delta": _ja(0.06),
        },
    )
)

register_job(
    Job(
        id="adventure.adventurer.star_sniper",
        domain="adventure",
        job_class="adventurer",
        archetype="star_sniper",
        name="스타 스나이퍼",
        tier_base=4,
        mobility=1.0,
        risk=0.95,
        training_years=6.0,
        prestige_base=0.95,
        field_affinity={
            "value_mass_delta": _ja(0.11),
            "will_tension_delta": _ja(0.08),
            "threat_delta": _ja(0.2),
            "faith_delta": _ja(0.0),
        },
    )
)

register_job(
    Job(
        id="adventure.adventurer.night_reaper",
        domain="adventure",
        job_class="adventurer",
        archetype="night_reaper",
        name="나이트 리퍼",
        tier_base=4,
        mobility=1.0,
        risk=0.98,
        training_years=7.0,
        prestige_base=0.96,
        field_affinity={
            "value_mass_delta": _ja(0.11),
            "will_tension_delta": _ja(0.11),
            "threat_delta": _ja(0.22),
            "faith_delta": _ja(-0.05),
        },
    )
)

register_job(
    Job(
        id="adventure.adventurer.shadow_lord",
        domain="adventure",
        job_class="adventurer",
        archetype="shadow_lord",
        name="섀도 로드",
        tier_base=4,
        mobility=1.0,
        risk=0.95,
        training_years=7.0,
        prestige_base=0.95,
        field_affinity={
            "value_mass_delta": _ja(0.1),
            "will_tension_delta": _ja(0.1),
            "threat_delta": _ja(0.2),
            "faith_delta": _ja(-0.02),
        },
    )
)

register_job(
    Job(
        id="faith.priest.saint",
        domain="faith",
        job_class="priest",
        archetype="saint",
        name="성인",
        tier_base=4,
        mobility=0.6,
        risk=0.7,
        training_years=7.0,
        prestige_base=0.97,
        field_affinity={
            "value_mass_delta": _ja(0.12),
            "will_tension_delta": _ja(-0.08),
            "threat_delta": _ja(-0.05),
            "faith_delta": _ja(0.22),
        },
    )
)

register_job(
    Job(
        id="faith.priest.battle_saint",
        domain="faith",
        job_class="priest",
        archetype="battle_saint",
        name="전투 성인",
        tier_base=4,
        mobility=0.8,
        risk=0.8,
        training_years=7.0,
        prestige_base=0.95,
        field_affinity={
            "value_mass_delta": _ja(0.11),
            "will_tension_delta": _ja(-0.02),
            "threat_delta": _ja(0.05),
            "faith_delta": _ja(0.18),
        },
    )
)

register_job(
    Job(
        id="craft.artisan.legendary_smith",
        domain="craft",
        job_class="artisan",
        archetype="legendary_smith",
        name="전설의 대장장이",
        tier_base=4,
        mobility=0.5,
        risk=0.7,
        training_years=7.0,
        prestige_base=0.96,
        field_affinity={
            "value_mass_delta": _ja(0.12),
            "will_tension_delta": _ja(0.04),
            "threat_delta": _ja(0.06),
            "faith_delta": _ja(0.03),
        },
    )
)

register_job(
    Job(
        id="craft.artisan.legendary_alchemist",
        domain="craft",
        job_class="artisan",
        archetype="legendary_alchemist",
        name="전설의 연금술사",
        tier_base=4,
        mobility=0.5,
        risk=0.75,
        training_years=7.0,
        prestige_base=0.96,
        field_affinity={
            "value_mass_delta": _ja(0.12),
            "will_tension_delta": _ja(0.03),
            "threat_delta": _ja(0.07),
            "faith_delta": _ja(0.04),
        },
    )
)


# --- Villain / dark-aligned jobs -------------------------------------------

register_job(
    Job(
        id="knowledge.scholar.dark_mage",
        domain="knowledge",
        job_class="scholar",
        archetype="dark_mage",
        name="Dark Mage",
        tier_base=2,
        mobility=0.5,
        risk=0.8,
        training_years=3.0,
        prestige_base=0.6,
        field_affinity={
            "value_mass_delta": _ja(0.05),
            "will_tension_delta": _ja(0.05),
            "threat_delta": _ja(0.12),
            "faith_delta": _ja(-0.05),
        },
        virtue_alignment={},
        sin_alignment={"pride": 0.4, "wrath": 0.3, "greed": 0.2},
    )
)

register_job(
    Job(
        id="knowledge.scholar.necromancer",
        domain="knowledge",
        job_class="scholar",
        archetype="necromancer",
        name="Necromancer",
        tier_base=3,
        mobility=0.5,
        risk=0.9,
        training_years=5.0,
        prestige_base=0.8,
        field_affinity={
            "value_mass_delta": _ja(0.06),
            "will_tension_delta": _ja(0.06),
            "threat_delta": _ja(0.16),
            "faith_delta": _ja(-0.08),
        },
        virtue_alignment={},
        sin_alignment={"pride": 0.5, "wrath": 0.3, "greed": 0.3, "envy": 0.2},
    )
)

register_job(
    Job(
        id="faith.priest.cultist",
        domain="faith",
        job_class="priest",
        archetype="cultist",
        name="Cultist",
        tier_base=2,
        mobility=0.6,
        risk=0.8,
        training_years=2.0,
        prestige_base=0.5,
        field_affinity={
            "value_mass_delta": _ja(0.04),
            "will_tension_delta": _ja(0.07),
            "threat_delta": _ja(0.1),
            "faith_delta": _ja(-0.02),
        },
        virtue_alignment={},
        sin_alignment={"greed": 0.3, "envy": 0.3, "wrath": 0.3, "pride": 0.4},
    )
)

register_job(
    Job(
        id="faith.priest.dark_herald",
        domain="faith",
        job_class="priest",
        archetype="dark_herald",
        name="Dark Herald",
        tier_base=3,
        mobility=0.7,
        risk=0.9,
        training_years=4.0,
        prestige_base=0.8,
        field_affinity={
            "value_mass_delta": _ja(0.05),
            "will_tension_delta": _ja(0.09),
            "threat_delta": _ja(0.14),
            "faith_delta": _ja(-0.05),
        },
        virtue_alignment={},
        sin_alignment={"greed": 0.3, "envy": 0.4, "wrath": 0.4, "pride": 0.5},
    )
)

register_job(
    Job(
        id="adventure.adventurer.raider",
        domain="adventure",
        job_class="adventurer",
        archetype="raider",
        name="Raider",
        tier_base=2,
        mobility=0.9,
        risk=0.9,
        training_years=2.0,
        prestige_base=0.5,
        field_affinity={
            "value_mass_delta": _ja(0.05),
            "will_tension_delta": _ja(0.07),
            "threat_delta": _ja(0.12),
            "faith_delta": _ja(0.0),
        },
        virtue_alignment={},
        sin_alignment={"greed": 0.5, "envy": 0.3, "wrath": 0.4},
    )
)

register_job(
    Job(
        id="adventure.adventurer.warlord",
        domain="adventure",
        job_class="adventurer",
        archetype="warlord",
        name="Warlord",
        tier_base=3,
        mobility=0.8,
        risk=0.95,
        training_years=4.0,
        prestige_base=0.85,
        field_affinity={
            "value_mass_delta": _ja(0.07),
            "will_tension_delta": _ja(0.1),
            "threat_delta": _ja(0.18),
            "faith_delta": _ja(-0.02),
        },
        virtue_alignment={},
        sin_alignment={"greed": 0.4, "envy": 0.4, "wrath": 0.6, "pride": 0.5},
    )
)


# --- Promotion tree wiring --------------------------------------------------

# Novice -> 1차 직업들
register_promotion_path(
    [
        "adventure.novice.novice",
        "martial.soldier.swordsman",
    ]
)
register_promotion_path(
    [
        "adventure.novice.novice",
        "knowledge.scholar.mage",
    ]
)
register_promotion_path(
    [
        "adventure.novice.novice",
        "adventure.adventurer.archer",
    ]
)
register_promotion_path(
    [
        "adventure.novice.novice",
        "adventure.adventurer.thief",
    ]
)
register_promotion_path(
    [
        "adventure.novice.novice",
        "faith.priest.acolyte_combat",
    ]
)
register_promotion_path(
    [
        "adventure.novice.novice",
        "trade.merchant.merchant",
    ]
)

# 검사 계열: 검사 -> 기사/크루세이더 -> 로드 나이트/팔라딘 -> 드래곤 나이트/홀리 크루세이더
register_promotion_path(
    [
        "martial.soldier.swordsman",
        "martial.soldier.knight_2",
        "martial.soldier.lord_knight",
        "martial.soldier.dragon_knight",
    ]
)
register_promotion_path(
    [
        "martial.soldier.swordsman",
        "martial.soldier.crusader_2",
        "martial.soldier.paladin",
        "martial.soldier.holy_crusader",
    ]
)

# 마법사 계열: 마법사 -> 위자드/세이지 -> 아크메이지/엘리멘탈리스트 -> 아케인 로드
register_promotion_path(
    [
        "knowledge.scholar.mage",
        "knowledge.scholar.wizard",
        "knowledge.scholar.archmage",
        "knowledge.scholar.arcane_lord",
    ]
)
register_promotion_path(
    [
        "knowledge.scholar.mage",
        "knowledge.scholar.sage_2",
        "knowledge.scholar.elementalist",
        "knowledge.scholar.celestial_sage",
    ]
)
register_promotion_path(
    [
        "knowledge.scholar.mage",
        "knowledge.scholar.dark_mage",
        "knowledge.scholar.necromancer",
    ]
)

# 궁수 계열: 궁수 -> 헌터 -> 스나이퍼 -> 스타 스나이퍼
register_promotion_path(
    [
        "adventure.adventurer.archer",
        "adventure.adventurer.hunter",
        "adventure.adventurer.sniper",
        "adventure.adventurer.star_sniper",
    ]
)

# 도둑 계열: 도둑 -> 로그/어새신 -> 섀도 어새신/나이트 리퍼
register_promotion_path(
    [
        "adventure.adventurer.thief",
        "adventure.adventurer.rogue",
        "adventure.adventurer.shadow_assassin",
        "adventure.adventurer.shadow_lord",
    ]
)
register_promotion_path(
    [
        "adventure.adventurer.thief",
        "adventure.adventurer.assassin",
        "adventure.adventurer.shadow_assassin",
        "adventure.adventurer.night_reaper",
    ]
)
register_promotion_path(
    [
        "adventure.adventurer.thief",
        "adventure.adventurer.raider",
        "adventure.adventurer.warlord",
    ]
)

# 복사 계열: 복사 -> 프리스트/몽크 -> 하이 프리스트/하이에로펀트 -> 성인
register_promotion_path(
    [
        "faith.priest.acolyte_combat",
        "faith.priest.priest",
        "faith.priest.high_priest",
        "faith.priest.saint",
    ]
)
register_promotion_path(
    [
        "faith.priest.acolyte_combat",
        "faith.priest.monk_combat",
        "faith.priest.hierophant",
        "faith.priest.battle_saint",
    ]
)
register_promotion_path(
    [
        "faith.priest.acolyte_combat",
        "faith.priest.cultist",
        "faith.priest.dark_herald",
    ]
)

# 상인 계열: 상인 -> 대장장이/연금술사 -> 마스터 스미스/케미스트 -> 전설 직업
register_promotion_path(
    [
        "trade.merchant.merchant",
        "craft.artisan.blacksmith_2",
        "craft.artisan.master_smith",
        "craft.artisan.legendary_smith",
    ]
)
register_promotion_path(
    [
        "trade.merchant.merchant",
        "craft.artisan.alchemist",
        "craft.artisan.chemist",
        "craft.artisan.legendary_alchemist",
    ]
)

# --- Visual grades / border colors -----------------------------------------

JOB_BORDER_COLOR: Dict[str, str] = {}


def _register_border_color(job_ids: List[str], color: str) -> None:
    for jid in job_ids:
        JOB_BORDER_COLOR[jid] = color


_register_border_color(
    ["adventure.novice.novice"],
    "white",
)

_register_border_color(
    [
        "martial.soldier.swordsman",
        "knowledge.scholar.mage",
        "adventure.adventurer.archer",
        "adventure.adventurer.thief",
        "faith.priest.acolyte_combat",
        "trade.merchant.merchant",
    ],
    "green",
)

_register_border_color(
    [
        "martial.soldier.knight_2",
        "martial.soldier.crusader_2",
        "knowledge.scholar.wizard",
        "knowledge.scholar.sage_2",
        "adventure.adventurer.hunter",
        "adventure.adventurer.rogue",
        "adventure.adventurer.assassin",
        "faith.priest.priest",
        "faith.priest.monk_combat",
        "craft.artisan.blacksmith_2",
        "craft.artisan.alchemist",
    ],
    "blue",
)

_register_border_color(
    [
        "martial.soldier.lord_knight",
        "martial.soldier.paladin",
        "knowledge.scholar.archmage",
        "knowledge.scholar.elementalist",
        "adventure.adventurer.sniper",
        "adventure.adventurer.shadow_assassin",
        "faith.priest.high_priest",
        "faith.priest.hierophant",
        "craft.artisan.master_smith",
        "craft.artisan.chemist",
    ],
    "purple",
)

_register_border_color(
    [
        "martial.soldier.dragon_knight",
        "martial.soldier.holy_crusader",
        "knowledge.scholar.arcane_lord",
        "knowledge.scholar.celestial_sage",
        "adventure.adventurer.star_sniper",
        "adventure.adventurer.shadow_lord",
        "adventure.adventurer.night_reaper",
        "faith.priest.saint",
        "faith.priest.battle_saint",
        "craft.artisan.legendary_smith",
        "craft.artisan.legendary_alchemist",
    ],
    "gold",
)


def get_job_border_color(job_id: str) -> str:
    """
    Return a simple color keyword for UI borders.

    Colors follow a Ragnarok-style progression:
    white -> green -> blue -> purple -> gold.
    """
    return JOB_BORDER_COLOR.get(job_id, "green")


# --- Race-based job preferences --------------------------------------------

RACE_DEFAULT_JOB_CANDIDATES: Dict[str, List[str]] = {
    "human": [
        "martial.soldier.swordsman",
        "knowledge.scholar.mage",
        "adventure.adventurer.archer",
        "adventure.adventurer.thief",
        "faith.priest.acolyte_combat",
        "trade.merchant.merchant",
    ],
    "elf": [
        "knowledge.scholar.mage",
        "adventure.adventurer.archer",
        "faith.priest.acolyte_combat",
    ],
    "dwarf": [
        "martial.soldier.swordsman",
        "craft.artisan.blacksmith_2",
        "trade.merchant.merchant",
    ],
    "orc": [
        "martial.soldier.swordsman",
        "adventure.adventurer.thief",
        "adventure.adventurer.raider",
    ],
    "fae": [
        "adventure.adventurer.archer",
        "knowledge.scholar.mage",
        "faith.priest.acolyte_combat",
    ],
}


def get_default_job_candidates_for_race(race: str) -> List[str]:
    key = (race or "human").lower()
    return list(RACE_DEFAULT_JOB_CANDIDATES.get(key, RACE_DEFAULT_JOB_CANDIDATES["human"]))
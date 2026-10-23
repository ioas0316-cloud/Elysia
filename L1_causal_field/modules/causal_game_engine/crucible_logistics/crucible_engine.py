"""
crucible_engine.py
==================
Crucible (Equipment Sublimation) System Engine: Crystallization of Causality and Ordeals.

Implements:
1. Causal Memory Accumulation:
   - Vector integral: C = (w_STR, w_AGI, w_INT, w_CON, w_SPI)
   - Equipment memory accumulation M in [0.0, 100.0%]
2. Threshold Ordeal Triggers:
   - Extensible ConditionEvaluator & Rubric system
   - Pre-defined ordeals: Guard/Despair, Tragedy/Revenge, Survival/Black-Market
3. Artifact Engraving & Naming:
   - Converting common equipment to unique artifacts
   - Causal domain powers fixation
4. Causal Field Integration:
   - Causal Gravity Wave & Chromatic Vector emission (Red/Yellow/Blue)
5. Chronicle History Log:
   - Queryable structured chronicle log entries
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field

from core.physics.causal_field import CausalField, InformationVoxel


@dataclass
class CausalVector:
    """
    5-Dimensional Causal Tendency Vector:
    C = (w_STR, w_AGI, w_INT, w_CON, w_SPI)
    Representing physical flux, mobility, analytical order, endurance/protection, and mental belief.
    """
    w_str: float = 0.0
    w_agi: float = 0.0
    w_int: float = 0.0
    w_con: float = 0.0
    w_spi: float = 0.0

    def add(self, other: "CausalVector", weight: float = 1.0):
        self.w_str += other.w_str * weight
        self.w_agi += other.w_agi * weight
        self.w_int += other.w_int * weight
        self.w_con += other.w_con * weight
        self.w_spi += other.w_spi * weight

    def to_array(self) -> np.ndarray:
        return np.array([self.w_str, self.w_agi, self.w_int, self.w_con, self.w_spi], dtype=np.float32)

    def dominant_tendency(self) -> str:
        vals = {
            "STR": self.w_str,
            "AGI": self.w_agi,
            "INT": self.w_int,
            "CON": self.w_con,
            "SPI": self.w_spi
        }
        sorted_vals = sorted(vals.items(), key=lambda x: x[1], reverse=True)
        return f"{sorted_vals[0][0]}-{sorted_vals[1][0]}"

    def chromatic_signature(self) -> np.ndarray:
        """
        Converts causal tendency into Red/Blue/Yellow chromatic vector:
        - Red (Flux): STR / CON combat & physical force
        - Yellow (Logistics/Entropy): AGI / INT mobility & resource trade
        - Blue (Order/Mental): SPI / CON mental belief & structural shield
        """
        red = float(self.w_str + self.w_con * 0.5)
        yellow = float(self.w_agi + self.w_int * 0.8)
        blue = float(self.w_spi + self.w_con * 0.5)
        tot = red + yellow + blue + 1e-9
        return np.array([red / tot, blue / tot, yellow / tot], dtype=np.float32)


@dataclass
class Equipment:
    id: str
    name: str
    tier: str = "Common"  # "Common", "Advanced", "Rare", "Magic", "Artifact", "Legendary"
    memory_pct: float = 0.0  # 0.0 to 100.0%
    causal_vector: CausalVector = field(default_factory=CausalVector)
    is_artifact: bool = False
    artifact_engraving: Optional[str] = None
    domain_power_desc: Optional[str] = None
    power_active: bool = True
    sealed_reason: Optional[str] = None
    stat_bonus: Dict[str, float] = field(default_factory=lambda: {"atk": 10.0, "def": 5.0})

    def accumulate_memory(self, activity_vector: CausalVector, delta_pct: float = 2.0):
        if self.is_artifact:
            return
        self.memory_pct = float(min(100.0, self.memory_pct + delta_pct))
        self.causal_vector.add(activity_vector, weight=delta_pct / 100.0)


@dataclass
class HeroProfile:
    id: str
    name: str
    star_rank: int = 3
    current_hp_pct: float = 100.0  # 0 to 100%
    garrison_zone: str = "east_gate"  # "east_gate", "subterranean_market", "mage_tower", etc.
    solo_defense_time_min: float = 0.0
    mentor_slain: bool = False
    defeated_6star_hero: bool = False
    famine_active: bool = False
    citizens_survival_rate: float = 1.0  # 1.0 = 100%
    equipped_gear: Optional[Equipment] = None


@dataclass
class ChronicleEntry:
    timestamp_turn: int
    hero_id: str
    hero_name: str
    original_item_name: str
    ascended_artifact_name: str
    achievement_text: str
    causal_vector_summary: str
    chromatic_wave_color: str  # "RED", "YELLOW", "BLUE"

    def to_formatted_string(self) -> str:
        return (
            f"[{self.timestamp_turn} Turn Chronicle] "
            f"Hero '{self.hero_name}' (ID: {self.hero_id})'s '{self.original_item_name}' "
            f"absorbed blood and despair to sublime into '[Artifact] {self.ascended_artifact_name}'! "
            f"Achievement: {self.achievement_text} (Tendency: {self.causal_vector_summary}, Wave: {self.chromatic_wave_color})"
        )


class ChronicleHistoryLog:
    """Structured chronicle history log storing all sublimation and milestone events."""
    def __init__(self):
        self.entries: List[ChronicleEntry] = []

    def record_entry(self, entry: ChronicleEntry):
        self.entries.append(entry)

    def get_recent_entries(self, limit: int = 10) -> List[ChronicleEntry]:
        return self.entries[-limit:]

    def get_all_entries(self) -> List[ChronicleEntry]:
        return list(self.entries)


@dataclass
class OrdealRubric:
    id: str
    name: str
    ordeal_type: str  # "GUARD_DESPAIR", "TRAGEDY_REVENGE", "SURVIVAL_BLACK_MARKET", "CUSTOM"
    evaluator_func: Callable[[HeroProfile, Equipment], bool]
    result_artifact_name_template: str
    domain_power_desc: str
    chromatic_type: str  # "RED", "YELLOW", "BLUE"
    stat_bonus_multiplier: float = 3.0


class ConditionEvaluator:
    """Evaluates threshold ordeal conditions against active rubrics."""
    def __init__(self):
        self.rubrics: Dict[str, OrdealRubric] = {}
        self._init_default_rubrics()

    def register_rubric(self, rubric: OrdealRubric):
        self.rubrics[rubric.id] = rubric

    def _init_default_rubrics(self):
        # 1. Guard/Despair (수호/절망형)
        # HP <= 10%, solo gate/wall defense >= 5 min
        def eval_guard_despair(hero: HeroProfile, gear: Equipment) -> bool:
            return (
                gear.memory_pct >= 100.0 and
                hero.current_hp_pct <= 10.0 and
                hero.solo_defense_time_min >= 5.0
            )

        rubric_guard = OrdealRubric(
            id="ordeal_guard_despair",
            name="수호/절망형 (Guard / Despair Ordeal)",
            ordeal_type="GUARD_DESPAIR",
            evaluator_func=eval_guard_despair,
            result_artifact_name_template="통곡의 방패 ({hero_name})",
            domain_power_desc="성벽 내구도 붕괴 시 인근 부대 방어력 200% 증폭 및 사기 저하 무효",
            chromatic_type="RED",
            stat_bonus_multiplier=3.5
        )

        # 2. Tragedy/Revenge (비극/복수형)
        # Mentor/comrade slain & defeats enemy 6-star hero
        def eval_tragedy_revenge(hero: HeroProfile, gear: Equipment) -> bool:
            return (
                gear.memory_pct >= 100.0 and
                hero.mentor_slain and
                hero.defeated_6star_hero
            )

        rubric_tragedy = OrdealRubric(
            id="ordeal_tragedy_revenge",
            name="비극/복수형 (Tragedy / Revenge Ordeal)",
            ordeal_type="TRAGEDY_REVENGE",
            evaluator_func=eval_tragedy_revenge,
            result_artifact_name_template="핏빛 유산의 가시검 ({hero_name})",
            domain_power_desc="사기 저하 무효, 적 영웅 타격 시 공격력 폭증",
            chromatic_type="RED",
            stat_bonus_multiplier=4.0
        )

        # 3. Survival/Behind-the-scenes (생존/흑막형)
        # Famine active (food = 0), 100% citizens survived via black market smuggling
        def eval_survival_shadow(hero: HeroProfile, gear: Equipment) -> bool:
            return (
                gear.memory_pct >= 100.0 and
                hero.famine_active and
                hero.citizens_survival_rate >= 1.0 and
                hero.garrison_zone in ["subterranean_market", "black_market"]
            )

        rubric_survival = OrdealRubric(
            id="ordeal_survival_shadow",
            name="생존/흑막형 (Survival / Shadow Ordeal)",
            ordeal_type="SURVIVAL_BLACK_MARKET",
            evaluator_func=eval_survival_shadow,
            result_artifact_name_template="밀수꾼의 그림자 인장 ({hero_name})",
            domain_power_desc="영지 내 암시장 수수료 면제, 적 공성 보급 지연",
            chromatic_type="YELLOW",
            stat_bonus_multiplier=3.0
        )

        self.register_rubric(rubric_guard)
        self.register_rubric(rubric_tragedy)
        self.register_rubric(rubric_survival)

    def evaluate_hero_equipment(self, hero: HeroProfile, gear: Equipment) -> Optional[OrdealRubric]:
        if gear.memory_pct < 100.0 or gear.is_artifact:
            return None

        for rubric in self.rubrics.values():
            if rubric.evaluator_func(hero, gear):
                return rubric
        return None


class CrucibleEngine:
    """
    Crucible Engine managing Equipment Sublimation, Causal Vectors,
    Field Integration, and Chronicle Records.
    """
    def __init__(self, causal_field: Optional[CausalField] = None):
        self.causal_field = causal_field if causal_field else CausalField(dimensions=3)
        self.evaluator = ConditionEvaluator()
        self.chronicle_log = ChronicleHistoryLog()
        self.is_time_frozen: bool = False
        self.last_sublimation_event: Optional[Dict[str, Any]] = None

    def tick_memory_accumulation(
        self,
        hero: HeroProfile,
        activity_vector: CausalVector,
        environment_bonus: float = 2.0
    ):
        """
        Accumulates causal memory into equipped gear per tick.
        """
        if not hero.equipped_gear:
            return

        gear = hero.equipped_gear
        gear.accumulate_memory(activity_vector, delta_pct=environment_bonus)

        # Update hero voxel in causal field
        v_id = f"hero_{hero.id}"
        if v_id in self.causal_field.voxels:
            voxel = self.causal_field.voxels[v_id]
            chroma = gear.causal_vector.chromatic_signature()
            voxel.chromatic_vector = chroma

    def check_and_trigger_sublimation(self, hero: HeroProfile, current_turn: int) -> Optional[Equipment]:
        """
        Checks if equipment reaches 100% memory and satisfies an ordeal condition.
        If satisfied, triggers Sublimation Event:
        - Time Freeze simulation state
        - Causal Gravity Wave & Chromatic Field Emission
        - Artifact engraving & domain power fixation
        - Structured Chronicle Entry creation
        """
        gear = hero.equipped_gear
        if not gear or gear.is_artifact or gear.memory_pct < 100.0:
            return None

        rubric = self.evaluator.evaluate_hero_equipment(hero, gear)
        if not rubric:
            return None

        # 1. Trigger Sublimation: Time Freeze
        self.is_time_frozen = True

        # 2. Artifact Engraving
        original_name = gear.name
        artifact_name = rubric.result_artifact_name_template.format(hero_name=hero.name)
        engraving_text = f"Sublimed during {rubric.name} by Hero {hero.name} (Turn {current_turn})"

        gear.name = f"[Artifact] {artifact_name}"
        gear.tier = "Artifact"
        gear.is_artifact = True
        gear.artifact_engraving = engraving_text
        gear.domain_power_desc = rubric.domain_power_desc

        for k in gear.stat_bonus:
            gear.stat_bonus[k] *= rubric.stat_bonus_multiplier

        # 3. Causal Field & Gravity Wave Emission
        chroma = gear.causal_vector.chromatic_signature()
        if rubric.chromatic_type == "RED":
            chroma = np.array([0.7, 0.1, 0.2], dtype=np.float32)
        elif rubric.chromatic_type == "YELLOW":
            chroma = np.array([0.1, 0.2, 0.7], dtype=np.float32)
        elif rubric.chromatic_type == "BLUE":
            chroma = np.array([0.1, 0.7, 0.2], dtype=np.float32)

        v_id = f"hero_{hero.id}"
        if v_id in self.causal_field.voxels:
            v = self.causal_field.voxels[v_id]
            v.chromatic_vector = chroma
            v.potential += 50.0  # Massive potential wave spike

        # Global Causal Gravity Wave impulse
        self.causal_field.global_potential_gradient += chroma * 10.0

        # 4. Chronicle Entry
        chronicle_entry = ChronicleEntry(
            timestamp_turn=current_turn,
            hero_id=hero.id,
            hero_name=hero.name,
            original_item_name=original_name,
            ascended_artifact_name=artifact_name,
            achievement_text=rubric.domain_power_desc,
            causal_vector_summary=gear.causal_vector.dominant_tendency(),
            chromatic_wave_color=rubric.chromatic_type
        )
        self.chronicle_log.record_entry(chronicle_entry)

        self.last_sublimation_event = {
            "turn": current_turn,
            "hero_name": hero.name,
            "original_item": original_name,
            "artifact_name": artifact_name,
            "rubric_name": rubric.name,
            "chromatic_wave": rubric.chromatic_type,
            "domain_power": rubric.domain_power_desc,
            "formatted_chronicle": chronicle_entry.to_formatted_string()
        }

        # Resume Time
        self.is_time_frozen = False
        return gear

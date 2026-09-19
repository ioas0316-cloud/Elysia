import pytest
import numpy as np

from core.physics.stronghold_hero_causal_engine import (
    StrongholdHeroCausalEngine,
    Hero,
    HeroStats,
    StarRank,
    EndgameAscensionPath,
    Heirloom
)


def test_hero_initialization_and_combat_power_comparison():
    """3성 베테랑 vs 미성숙 6성의 가성비 및 타이밍 전투력 테스트"""
    hero_3star = Hero(
        id="aria",
        name="아리아",
        star_rank=StarRank.STAR_3,
        stats=HeroStats(str_val=10, agi_val=10, int_val=10, con_val=15, spi_val=15),
        veterancy=80.0  # 초반 고점 도달한 베테랑
    )

    hero_6star = Hero(
        id="victoria",
        name="빅토리아 공주",
        star_rank=StarRank.STAR_6,
        stats=HeroStats(str_val=15, agi_val=15, int_val=25, con_val=15, spi_val=20),
        veterancy=5.0   # 초반 조기 투입된 애송이
    )

    p3 = hero_3star.get_effective_combat_power()
    p6 = hero_6star.get_effective_combat_power()

    # 초반 베테랑 3성이 미성숙 6성보다 실전 전투력이 우월해야 함
    assert p3 > p6, f"3성 베테랑 전투력({p3:.1f})이 미성숙 6성 전투력({p6:.1f})보다 작음"


def test_logistics_production_chain():
    """밀->가루->빵 및 철->무기->정예장비 물류 생산 흐름 테스트"""
    engine = StrongholdHeroCausalEngine()
    initial_wheat = engine.logistics.wheat_count
    initial_bread = engine.logistics.bread_count

    engine.logistics.produce_turn()

    assert engine.logistics.wheat_count > initial_wheat
    assert engine.logistics.bread_count > initial_bread
    assert engine.logistics.elite_armor_count > 0.0


def test_garrison_fortress_buffs():
    """성채 기하학 및 영웅 주둔 위치에 따른 방어/기름/마법 버프 연산 테스트"""
    engine = StrongholdHeroCausalEngine()

    hero = Hero(
        id="defender",
        name="가르드노",
        star_rank=StarRank.STAR_2,
        stats=HeroStats(str_val=10, agi_val=10, int_val=10, con_val=20, spi_val=20)
    )
    engine.register_hero(hero)
    engine.assign_hero_garrison("defender", "east_gate")

    buffs = engine.get_fortress_geometry_buffs()
    assert buffs["wall_defense_multiplier"] > 1.0


def test_crucible_event_and_heirloom_creation():
    """인과적 시련 극복 및 보구(Heirloom) 결정화 테스트"""
    engine = StrongholdHeroCausalEngine()

    hero = Hero(
        id="smith",
        name="한스",
        star_rank=StarRank.STAR_3,
        stats=HeroStats(str_val=12, agi_val=10, int_val=10, con_val=15, spi_val=15)
    )
    engine.register_hero(hero)

    msg = engine.trigger_crucible_event("smith", "wall_defense_miracle")

    assert "성벽의 수호자" in hero.current_class
    assert hero.equipped_heirloom is not None
    assert "방패" in hero.equipped_heirloom.name
    assert hero.potential_ceiling > 150.0  # 잠재력 상한 파괴


def test_endgame_ascensions():
    """4대 엔드게임 도약 (가문 창시자, 마탑주, 흑막, 상단주) 조건 달성 테스트"""
    engine = StrongholdHeroCausalEngine()

    # 1. 가문의 창시자 / 성주
    hero_founder = Hero(
        id="founder",
        name="볼크",
        star_rank=StarRank.STAR_3,
        stats=HeroStats(str_val=35, agi_val=10, int_val=10, con_val=30, spi_val=20),
        veterancy=80.0
    )
    engine.register_hero(hero_founder)
    hero_founder.veterancy = 80.0
    engine.popularity = 85.0
    engine.check_and_trigger_endgame_ascension("founder")
    assert hero_founder.ascension == EndgameAscensionPath.HOUSE_FOUNDER

    # 2. 암흑가의 지배자 / 흑막
    hero_shadow = Hero(
        id="shadow",
        name="실피",
        star_rank=StarRank.STAR_3,
        stats=HeroStats(str_val=10, agi_val=50, int_val=20, con_val=10, spi_val=10)
    )
    engine.register_hero(hero_shadow)
    engine.assign_hero_garrison("shadow", "subterranean_market")
    engine.check_and_trigger_endgame_ascension("shadow")
    assert hero_shadow.ascension == EndgameAscensionPath.SHADOW_RULER
    assert engine.deterrence_index >= 50.0


def test_subterranean_sabotage():
    """지하 암시장 공작 메카닉 테스트"""
    engine = StrongholdHeroCausalEngine()
    engine.deterrence_index = 50.0

    res = engine.execute_subterranean_sabotage("disrupt_enemy_supply")
    assert "보급" in res and "차단" in res
    assert engine.deterrence_index == 40.0


def test_full_turn_simulation():
    """1턴 전체 시뮬레이션 흐름 및 사서(Chronicle) 로그 검증"""
    engine = StrongholdHeroCausalEngine()
    hero = Hero(
        id="hero1",
        name="레이몬드",
        star_rank=StarRank.STAR_1,
        stats=HeroStats()
    )
    engine.register_hero(hero)

    engine.step_turn()

    assert engine.turn_count == 2
    assert len(engine.chronicle_logs) >= 3
    summary = engine.get_summary_state()
    assert summary["turn"] == 2
    assert "logistics" in summary

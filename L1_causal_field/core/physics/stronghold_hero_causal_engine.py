import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import enum
import time

from core.physics.causal_field import CausalField, InformationVoxel, ConnectivityBeam


class StarRank(enum.IntEnum):
    STAR_1 = 1  # 마을 및 장인 스케일 (Local Mastery)
    STAR_2 = 2  # 영지 및 남작 스케일 (Baronial Domain)
    STAR_3 = 3  # 지역 및 시련 스케일 (Crucible Seed / Regional)
    STAR_4 = 4  # 공작 및 주 스케일 (Provincial Leader)
    STAR_5 = 5  # 왕국 및 대륙 경제 스케일 (Royal Scope)
    STAR_6 = 6  # 제국 및 신화적 원형 스케일 (Imperial Mythic)


class EndgameAscensionPath(enum.Enum):
    NONE = "미달성"
    HOUSE_FOUNDER = "가문의 창시자 / 성주"         # STR / CON 기반
    MAGE_TOWER_MASTER = "마탑의 지배자 / 대마도사"  # INT 기반
    SHADOW_RULER = "암흑가의 지배자 / 흑막"        # AGI 기반
    MASTER_MERCHANT = "대륙의 상단주 / 철혈 재상"  # INT / SPI 기반


class ItemTier(enum.Enum):
    COMMON = "일반"        # 1성 대장간, 대량보급, 내구도 소모
    ADVANCED = "고급"      # 2성 명품공방, 높은 내구도, 수리비 절감
    RARE = "희귀"          # 전장의 기억 각성, 맞춤형 트레이트
    MAGIC = "마법"         # 마탑 연금술, 마정석 소비, 마법 장막/오라
    ARTIFACT = "아티팩트"   # 인과적 시련 결정화, 가문의 보구(Heirloom)
    LEGENDARY = "전설"     # 대륙 신화, 전장 법칙 재정의


@dataclass
class EquipmentItem:
    id: str
    name: str
    tier: ItemTier
    combat_power_bonus: float = 10.0
    durability: float = 100.0
    max_durability: float = 100.0
    mana_upkeep_fulfilled: bool = True
    memory_exp: float = 0.0  # 전투 기억 적재 (희귀 각성용)
    description: str = "표준 장비"

    def evolve_check(self) -> bool:
        """일반 장비가 전투 기억(Memory EXP)을 쌓으면 희귀 장비로 각성"""
        if self.tier == ItemTier.COMMON and self.memory_exp >= 100.0:
            self.tier = ItemTier.RARE
            self.name = f"희귀: 전장의 기억이 새겨진 {self.name}"
            self.combat_power_bonus *= 2.5
            self.max_durability = 150.0
            self.durability = 150.0
            return True
        return False


@dataclass
class HeroStats:
    """5대 스탯: 스탯 수치이자 인물의 가치관·성격적 성향"""
    str_val: float = 10.0  # 힘: 직진성 · 책임감 · 지도력
    agi_val: float = 10.0  # 민: 유연성 · 실용주의 · 적응력
    int_val: float = 10.0  # 지: 인과분석 · 시스템관 · 합리성
    con_val: float = 10.0  # 체: 인내심 · 수호 · 희생정신
    spi_val: float = 10.0  # 정: 신념 · 공감 능력 · 멘탈 장력

    def total(self) -> float:
        return self.str_val + self.agi_val + self.int_val + self.con_val + self.spi_val

    def dominant_value_description(self) -> str:
        vals = {
            "STR (책임/돌파)": self.str_val,
            "AGI (실용/적응)": self.agi_val,
            "INT (합리/인과)": self.int_val,
            "CON (수호/인내)": self.con_val,
            "SPI (신념/사기)": self.spi_val,
        }
        sorted_vals = sorted(vals.items(), key=lambda x: x[1], reverse=True)
        top1, top2 = sorted_vals[0], sorted_vals[1]
        return f"{top1[0]} 중심 ({top1[1]:.1f}), {top2[0]} 보조 ({top2[1]:.1f})"


@dataclass
class Heirloom:
    """인과적 보구 (Heirloom): 시련을 극복할 때 결정화되는 가문의 유산"""
    id: str
    name: str
    creator_hero_name: str
    created_turn: int
    stat_bias: str  # e.g. "CON/SPI"
    description: str
    wall_defense_bonus: float = 0.0
    logistics_speed_bonus: float = 0.0
    morale_decay_immunity: bool = False
    shadow_cost_discount: float = 0.0


@dataclass
class Hero:
    id: str
    name: str
    star_rank: StarRank
    stats: HeroStats
    level: int = 1
    exp: float = 0.0
    potential_ceiling: float = 100.0  # 잠재력 상한선
    veterancy: float = 0.0            # 실전 숙련도 (0.0 ~ 100.0)
    current_class: str = "무명 기사 후보"
    traits: List[str] = field(default_factory=list)
    garrison_location: Optional[str] = None  # 주둔 위치 (e.g. "east_gate", "mage_tower")
    upkeep_fulfilled: bool = True
    trauma_level: float = 0.0                # 트라우마/왜곡 지수
    ascension: EndgameAscensionPath = EndgameAscensionPath.NONE
    equipped_heirloom: Optional[Heirloom] = None
    equipped_weapon: Optional[EquipmentItem] = None
    is_shadow_master: bool = False           # 지하 흑막 활성화 여부
    history_logs: List[str] = field(default_factory=list)

    def __post_init__(self):
        # 성급에 따른 잠재력 상한 및 초반 스탯 설정
        if self.star_rank == StarRank.STAR_1:
            self.potential_ceiling = 50.0
            self.veterancy = 80.0  # 시작부터 완숙
        elif self.star_rank == StarRank.STAR_2:
            self.potential_ceiling = 80.0
            self.veterancy = 60.0
        elif self.star_rank == StarRank.STAR_3:
            self.potential_ceiling = 150.0  # 시련 시 깨부술 수 있음
            self.veterancy = 20.0
        elif self.star_rank == StarRank.STAR_4:
            self.potential_ceiling = 250.0
            self.veterancy = 30.0
        elif self.star_rank == StarRank.STAR_5:
            self.potential_ceiling = 400.0
            self.veterancy = 15.0
        elif self.star_rank == StarRank.STAR_6:
            self.potential_ceiling = 600.0
            self.veterancy = 5.0  # 초반에는 허약한 애송이

    def get_effective_combat_power(self) -> float:
        """
        [가성비 및 타이밍 상성 연산]
        전투력 = 스탯 합 * (숙련도 보정) * (보급 및 트라우마 보정)
        저성급 베테랑은 숙련도 보정으로 미성숙 6성을 제압할 수 있음.
        """
        base_power = self.stats.total()
        # 숙련도 효율: 0.5 ~ 2.0배
        veterancy_mult = 0.5 + (self.veterancy / 100.0) * 1.5
        upkeep_mult = 1.0 if self.upkeep_fulfilled else 0.4
        trauma_penalty = max(0.2, 1.0 - (self.trauma_level / 100.0) * 0.5)

        # 6성이 미성숙할 때(veterancy < 20) 조기 투입되면 전투력 페널티
        immature_penalty = 0.6 if (self.star_rank == StarRank.STAR_6 and self.veterancy < 20.0) else 1.0

        total_power = base_power * veterancy_mult * upkeep_mult * trauma_penalty * immature_penalty
        if self.equipped_heirloom:
            total_power *= 1.25

        if self.equipped_weapon:
            # 마법/전설 장비는 마나 보급 실패 시 성능 격감 (고철화)
            if self.equipped_weapon.tier in [ItemTier.MAGIC, ItemTier.LEGENDARY] and not self.equipped_weapon.mana_upkeep_fulfilled:
                total_power += self.equipped_weapon.combat_power_bonus * 0.1
            else:
                total_power += self.equipped_weapon.combat_power_bonus

        return total_power


@dataclass
class LogisticsChain:
    """영지 물류 체인 데이터"""
    wheat_count: float = 100.0      # 밀
    flour_count: float = 50.0       # 가루
    bread_count: float = 80.0       # 빵 (기본 식량)
    iron_ore_count: float = 40.0    # 철광석
    weapon_count: float = 20.0      # 기본 무기
    elite_armor_count: float = 5.0  # 정예 장비 (3~4성용)
    mana_crystal_count: float = 20.0 # 마정석 (마법/전설 장비 유지를 위한 3차 물류)
    wine_count: float = 10.0        # 고급 포도주 (5~6성 유휴 보급품)
    silk_count: float = 10.0        # 비단 의복 (5~6성 유휴 보급품)

    # 생산 건물 수량
    farm_buildings: int = 2
    mill_buildings: int = 1
    bakery_buildings: int = 1
    iron_mine_buildings: int = 1
    forge_buildings: int = 1
    luxury_workshop_buildings: int = 1

    def produce_turn(self) -> Dict[str, float]:
        """1턴 물류 생산 흐름 연쇄"""
        produced = {}

        # 1. 밀 -> 가루 -> 빵
        harvest_wheat = self.farm_buildings * 15.0
        self.wheat_count += harvest_wheat
        milled_flour = min(self.wheat_count, self.mill_buildings * 12.0)
        self.wheat_count -= milled_flour
        self.flour_count += milled_flour
        baked_bread = min(self.flour_count, self.bakery_buildings * 10.0)
        self.flour_count -= baked_bread
        self.bread_count += baked_bread
        produced["bread"] = baked_bread

        # 2. 철광석 -> 무기 -> 정예 장비
        mined_iron = self.iron_mine_buildings * 8.0
        self.iron_ore_count += mined_iron
        forged_weapons = min(self.iron_ore_count, self.forge_buildings * 5.0)
        self.iron_ore_count -= forged_weapons
        self.weapon_count += forged_weapons
        forged_armor = min(self.weapon_count * 0.5, self.forge_buildings * 2.0)
        self.weapon_count -= forged_armor
        self.elite_armor_count += forged_armor
        produced["elite_armor"] = forged_armor

        # 3. 고급 사치품 (포도주 및 비단, 마정석)
        produced_luxury = self.luxury_workshop_buildings * 2.0
        self.wine_count += produced_luxury
        self.silk_count += produced_luxury
        self.mana_crystal_count += produced_luxury * 1.5
        produced["luxury"] = produced_luxury

        return produced


class StrongholdHeroCausalEngine:
    """
    [Stronghold-Hero Causal Simulation Engine]
    스트롱홀드식 물류/성채 기하학 + 서브컬처 영웅 가치관/인과적 시련 + Elysia CausalField 통합 엔진.
    """

    def __init__(self, field_dims: int = 3):
        self.causal_field = CausalField(dimensions=field_dims)
        self.logistics = LogisticsChain()
        self.heroes: Dict[str, Hero] = {}
        self.popularity: float = 80.0           # 영지 민심 (0 ~ 100)
        self.deterrence_index: float = 10.0     # 암시장 지하 공포/통제 지수 (0 ~ 100)
        self.turn_count: int = 1
        self.chronicle_logs: List[str] = []     # 사서 (Chronicle) 역사 기록
        self.heirlooms: Dict[str, Heirloom] = {}

        # 공간 구역 노드 정의 (Voxel 연동)
        self.spatial_nodes = {
            "east_gate": {"name": "동문 방어선", "type": "gate", "pos": np.array([10.0, 0.0, 0.0], dtype=np.float32)},
            "wall_tower": {"name": "중앙 성루", "type": "tower", "pos": np.array([0.0, 10.0, 0.0], dtype=np.float32)},
            "mage_tower": {"name": "대마탑", "type": "mage_tower", "pos": np.array([-10.0, 0.0, 0.0], dtype=np.float32)},
            "armory": {"name": "대장간 구역", "type": "armory", "pos": np.array([0.0, -10.0, 0.0], dtype=np.float32)},
            "subterranean_market": {"name": "지하 암시장", "type": "shadow", "pos": np.array([0.0, 0.0, -10.0], dtype=np.float32)},
        }

        self._init_causal_field_voxels()
        self._log_chronicle("성채 시뮬레이션 인과장이 초기화되었습니다. 영지의 역사가 시작됩니다.")

    def _init_causal_field_voxels(self):
        """성채 지형 노드들을 Elysia InformationVoxel로 인과장에 바인딩"""
        for node_id, info in self.spatial_nodes.items():
            voxel = InformationVoxel(
                id=f"node_{node_id}",
                content=info["name"],
                tensor=np.array([0.5, 0.5, 0.5], dtype=np.float32),
                position=info["pos"],
                mass=2.0
            )
            self.causal_field.add_voxel(voxel)

        # 구역 간 연결 빔 구축
        self.causal_field.link_voxels("node_east_gate", "node_wall_tower", strength=1.5)
        self.causal_field.link_voxels("node_wall_tower", "node_mage_tower", strength=1.5)
        self.causal_field.link_voxels("node_wall_tower", "node_armory", strength=1.2)
        self.causal_field.link_voxels("node_armory", "node_subterranean_market", strength=2.0)

    def _log_chronicle(self, message: str):
        entry = f"[제 {self.turn_count} 년차 사서] {message}"
        self.chronicle_logs.append(entry)

    def register_hero(self, hero: Hero):
        self.heroes[hero.id] = hero
        # 영웅 고유 Voxel 등록
        hero_voxel = InformationVoxel(
            id=f"hero_{hero.id}",
            content=hero.name,
            tensor=np.array([
                hero.stats.str_val / 20.0,
                hero.stats.int_val / 20.0,
                hero.stats.spi_val / 20.0
            ], dtype=np.float32),
            position=np.zeros(3, dtype=np.float32),
            mass=float(hero.star_rank)
        )
        self.causal_field.add_voxel(hero_voxel)
        self._log_chronicle(f"새로운 영웅 '{hero.name}' ({hero.star_rank.value}성, {hero.current_class})이(가) 성채에 합류했습니다.")

    def assign_hero_garrison(self, hero_id: str, location_id: Optional[str]) -> str:
        """영웅을 성채 특정 기하학 구역에 주둔시킴"""
        if hero_id not in self.heroes:
            return "영웅을 찾을 수 없습니다."
        hero = self.heroes[hero_id]

        if location_id and location_id not in self.spatial_nodes:
            return "유효하지 않은 주둔 구역입니다."

        old_loc = hero.garrison_location
        hero.garrison_location = location_id

        # 인과장 내 영웅 Voxel 위치 업데이트 및 빔 연결
        hero_voxel_id = f"hero_{hero.id}"
        if location_id:
            loc_pos = self.spatial_nodes[location_id]["pos"]
            self.causal_field.voxels[hero_voxel_id].position = loc_pos.copy()
            node_voxel_id = f"node_{location_id}"
            self.causal_field.link_voxels(hero_voxel_id, node_voxel_id, strength=3.0)

            msg = f"영웅 '{hero.name}'이(가) [{self.spatial_nodes[location_id]['name']}] 구역에 주둔했습니다."
            self._log_chronicle(msg)
            return msg
        else:
            msg = f"영웅 '{hero.name}'이(가) 주둔 해제되었습니다."
            self._log_chronicle(msg)
            return msg

    def get_fortress_geometry_buffs(self) -> Dict[str, Any]:
        """
        [성채 기하학 및 영웅 주둔 공간 시너지 연산]
        주둔한 영웅들의 스탯과 위치에 따라 방어선 사거리, 관통력, 기름 발화, 수리 속도, 마법 장막 등 계산.
        """
        buffs = {
            "wall_defense_multiplier": 1.0,
            "ballista_range_bonus": 0.0,
            "pitch_ignition_radius": 1.0,
            "repair_speed_multiplier": 1.0,
            "magic_barrier_shield": 0.0,
            "subterranean_sabotage_power": 0.0
        }

        for hero in self.heroes.values():
            if not hero.garrison_location:
                continue

            loc_type = self.spatial_nodes[hero.garrison_location]["type"]

            if loc_type in ["gate", "tower"]:
                # 수성/성벽 주둔
                buffs["wall_defense_multiplier"] += (hero.stats.con_val / 20.0) * (hero.veterancy / 50.0)
                buffs["ballista_range_bonus"] += (hero.stats.agi_val / 10.0)
            elif loc_type == "mage_tower":
                # 마탑 주둔
                buffs["pitch_ignition_radius"] += (hero.stats.int_val / 15.0)
                buffs["magic_barrier_shield"] += (hero.stats.spi_val * 2.5) * (1.0 + hero.veterancy / 100.0)
            elif loc_type == "armory":
                # 대장간 주둔
                buffs["repair_speed_multiplier"] += (hero.stats.str_val / 15.0) + (hero.stats.int_val / 20.0)
            elif loc_type == "shadow":
                # 지하 암시장 주둔
                buffs["subterranean_sabotage_power"] += (hero.stats.agi_val / 5.0) + (hero.stats.int_val / 10.0)

            # 보구 패시브 효과 적용
            if hero.equipped_heirloom:
                buffs["wall_defense_multiplier"] += hero.equipped_heirloom.wall_defense_bonus
                buffs["repair_speed_multiplier"] += hero.equipped_heirloom.logistics_speed_bonus

            # 마법/전설 장비 착용 시 결계 오라 증폭
            if hero.equipped_weapon and hero.equipped_weapon.tier in [ItemTier.MAGIC, ItemTier.LEGENDARY]:
                if hero.equipped_weapon.mana_upkeep_fulfilled:
                    buffs["magic_barrier_shield"] += 50.0

        return buffs

    def process_upkeep_and_logistics(self):
        """
        [영지 물류 및 영웅 사치품/식량 보급 처리]
        5~6성일수록 고급 사치품(포도주, 비단) 요구. 보급 끊기면 사기 저하 및 트라우마/탈영 위험.
        """
        # 1. 생산 단계
        self.logistics.produce_turn()

        # 2. 민심 및 식량 소비
        pop_food_demand = 20.0
        if self.logistics.bread_count >= pop_food_demand:
            self.logistics.bread_count -= pop_food_demand
            self.popularity = min(100.0, self.popularity + 2.0)
        else:
            # 기근 발생
            self.logistics.bread_count = 0.0
            self.popularity = max(0.0, self.popularity - 15.0)
            self._log_chronicle("⚠️ 영지에 기근이 찾아왔습니다! 영민들의 사기와 민심이 급락합니다.")

        # 3. 영웅 보급 수급
        for hero in self.heroes.values():
            if hero.star_rank in [StarRank.STAR_1, StarRank.STAR_2, StarRank.STAR_3]:
                # 거친 빵만 공급되어도 완벽 동작
                if self.logistics.bread_count >= 2.0:
                    self.logistics.bread_count -= 2.0
                    hero.upkeep_fulfilled = True
                    hero.veterancy = min(100.0, hero.veterancy + 2.0)  # 빠르게 숙련도 만렙 달성
                else:
                    hero.upkeep_fulfilled = False

            elif hero.star_rank in [StarRank.STAR_4, StarRank.STAR_5, StarRank.STAR_6]:
                # 정예 장비 및 고급 포도주/비단 요구
                req_luxury = 1.0 if hero.star_rank == StarRank.STAR_4 else 2.0
                has_luxury = (self.logistics.wine_count >= req_luxury and self.logistics.silk_count >= req_luxury)

                if has_luxury:
                    self.logistics.wine_count -= req_luxury
                    self.logistics.silk_count -= req_luxury
                    hero.upkeep_fulfilled = True
                    hero.veterancy = min(100.0, hero.veterancy + 1.0)
                else:
                    hero.upkeep_fulfilled = False
                    hero.trauma_level = min(100.0, hero.trauma_level + 5.0)
                    msg = f"⚠️ 고급 보급이 끊겨 [{hero.name}]({hero.star_rank.value}성)의 불만과 트라우마가 상승합니다 ({hero.trauma_level:.1f})."
                    self._log_chronicle(msg)

            # 장비 마나 보급 체크 (마법/전설 등급)
            if hero.equipped_weapon and hero.equipped_weapon.tier in [ItemTier.MAGIC, ItemTier.LEGENDARY]:
                if self.logistics.mana_crystal_count >= 2.0:
                    self.logistics.mana_crystal_count -= 2.0
                    hero.equipped_weapon.mana_upkeep_fulfilled = True
                else:
                    hero.equipped_weapon.mana_upkeep_fulfilled = False
                    self._log_chronicle(f"⚠️ 마정석 보급이 끊겨 [{hero.name}]의 [{hero.equipped_weapon.name}] 장비 마력 오라가 중단되었습니다!")

    def trigger_crucible_event(self, hero_id: str, event_type: str) -> str:
        """
        [인과적 시련 (Crucible Event) & 분기형 전직]
        사건 서순(성벽 수성, 스승 전사, 기근 행정)에 따라 클래스, 트레이트, 5대 스탯 및 외형 성향 재편.
        """
        if hero_id not in self.heroes:
            return "영웅을 찾을 수 없습니다."

        hero = self.heroes[hero_id]
        result_msg = ""

        if event_type == "wall_defense_miracle":
            # [시련 A: 절체절명의 수성전 홀로 사수]
            hero.stats.con_val += 15.0
            hero.stats.spi_val += 15.0
            hero.veterancy = min(100.0, hero.veterancy + 40.0)
            hero.current_class = "성벽의 수호자 (통곡의 기사)"
            hero.traits.append("불굴의 인과장")

            # 3성인 경우 잠재력 상한선 대폭 파괴
            if hero.star_rank == StarRank.STAR_3:
                hero.potential_ceiling += 150.0
                result_msg = f"🔥 [시련 극복!] 무명 3성 영웅 '{hero.name}'이(가) 통곡의 성벽을 사수하며 [{hero.current_class}]로 각성했습니다! 잠재력 상한선이 파괴되었습니다."
            else:
                result_msg = f"🔥 '{hero.name}'이(가) 수성전의 시련을 극복하고 [{hero.current_class}]로 각성했습니다."

            # 보구 결정화
            heirloom = Heirloom(
                id=f"heirloom_{hero.id}_{self.turn_count}",
                name=f"통곡의 낡은 방패 ({hero.name})",
                creator_hero_name=hero.name,
                created_turn=self.turn_count,
                stat_bias="CON/SPI",
                description="절체절명의 성문이 뚫린 위기에서 홀로 10분을 버텨낸 의지에서 결정화된 방패.",
                wall_defense_bonus=0.5,
                morale_decay_immunity=True
            )
            self.heirlooms[heirloom.id] = heirloom
            hero.equipped_heirloom = heirloom

        elif event_type == "mentor_slain":
            # [시련 B: 스승의 전사와 복수심]
            hero.stats.str_val += 20.0
            hero.trauma_level = min(100.0, hero.trauma_level + 30.0)
            hero.traits.append("복수심의 맹세")
            if hero.stats.str_val > hero.stats.spi_val:
                hero.current_class = "혈염의 공성 광전사"
            else:
                hero.current_class = "복수를 품은 복합기사"
            result_msg = f"💔 스승의 전사를 목격한 '{hero.name}'이(가) 깊은 트라우마 속에서 [{hero.current_class}]로 변질/각성했습니다."

        elif event_type == "famine_administration":
            # [시련 C: 기근 속 식량 보급 전담]
            hero.stats.int_val += 15.0
            hero.stats.agi_val += 10.0
            hero.current_class = "영지 행정관 (철혈의 재상)"
            hero.traits.append("정밀 식량 통제")
            result_msg = f"📜 식량 난을 지혜롭게 이끈 '{hero.name}'이(가) [{hero.current_class}]로 전직했습니다."

        elif event_type == "misaligned_brutal_melee":
            # [잘못된 성장: 마법/전술 인재를 혈육전에 무리하게 찌름]
            hero.trauma_level = min(100.0, hero.trauma_level + 40.0)
            hero.current_class = "트라우마에 사로잡힌 광전사"
            hero.traits.append("통제 불능 폭주")
            result_msg = f"⚠️ 적성에 맞지 않는 극한의 육탄전에 방치된 '{hero.name}'의 정신이 꺾이며 [{hero.current_class}]로 왜곡 성장했습니다!"

        self._log_chronicle(result_msg)
        return result_msg

    def check_and_trigger_endgame_ascension(self, hero_id: str) -> Optional[EndgameAscensionPath]:
        """
        [엔드게임 도약 (4 Endgame Ascensions)]
        3성 영웅 등이 요구 조건 달성 시 거시적 세력/대륙 단위 수장으로 도약.
        """
        if hero_id not in self.heroes:
            return None

        hero = self.heroes[hero_id]
        if hero.ascension != EndgameAscensionPath.NONE:
            return hero.ascension

        # 1. 가문의 창시자 / 성주 (STR/CON 기반 + 수성 완숙)
        if (hero.stats.str_val + hero.stats.con_val) >= 60.0 and hero.veterancy >= 70.0 and self.popularity >= 70.0:
            hero.ascension = EndgameAscensionPath.HOUSE_FOUNDER
            msg = f"👑 [대도약!] 3성 출신 '{hero.name}'이(가) 영민과 군대의 추대를 받아 [신흥 가문의 창시자 / 성주]로 거듭났습니다!"
            self._log_chronicle(msg)
            return hero.ascension

        # 2. 마탑의 지배자 (INT 기반 + 대마탑 주둔)
        if hero.stats.int_val >= 50.0 and hero.garrison_location == "mage_tower":
            hero.ascension = EndgameAscensionPath.MAGE_TOWER_MASTER
            msg = f"🔮 [대도약!] '{hero.name}'이(가) 성채의 기하학적 마력을 극대화하여 [마탑의 지배자 / 대마도사]로 군림합니다!"
            self._log_chronicle(msg)
            return hero.ascension

        # 3. 암흑가의 지배자 / 흑막 (AGI 기반 + 지하 암시장 주둔)
        if hero.stats.agi_val >= 45.0 and (hero.garrison_location == "subterranean_market" or hero.is_shadow_master):
            hero.ascension = EndgameAscensionPath.SHADOW_RULER
            hero.is_shadow_master = True
            self.deterrence_index = min(100.0, self.deterrence_index + 40.0)
            msg = f"🕶️ [대도약!] '{hero.name}'이(가) 지하 네트워크를 틀어쥐고 [암흑가의 지배자 / 흑막]으로 대륙의 판세를 조종합니다!"
            self._log_chronicle(msg)
            return hero.ascension

        # 4. 대륙의 상단주 / 재상 (INT/SPI 기반 + 물류 풍부)
        if hero.stats.int_val + hero.stats.spi_val >= 55.0 and self.logistics.wine_count >= 15.0:
            hero.ascension = EndgameAscensionPath.MASTER_MERCHANT
            msg = f"💰 [대도약!] '{hero.name}'이(가) 거시 물류망을 장악하여 [대륙의 상단주 / 철혈 재상]으로 승화했습니다!"
            self._log_chronicle(msg)
            return hero.ascension

        return EndgameAscensionPath.NONE

    def execute_subterranean_sabotage(self, action: str) -> str:
        """
        [지하 암시장 공작 메카닉]
        1) 적 보급선 교란 (Enemy Supply Interruption)
        2) 수성 자재 밀매 (Smuggle Pitch & Arrows)
        3) 성문 잠금장치 공작 (Unlock/Sabotage Gate)
        """
        if self.deterrence_index < 20.0:
            return "암시장 공포 지수가 부족하여 공작을 진행할 수 없습니다."

        if action == "disrupt_enemy_supply":
            self.deterrence_index -= 10.0
            msg = "🕵️‍♂️ [지하 공작] 적 대군의 보급 담당자를 매수하여 적의 식량을 도난시키고 보급을 차단했습니다!"
            self._log_chronicle(msg)
            return msg

        elif action == "smuggle_pitch_arrows":
            self.logistics.weapon_count += 15.0
            self.deterrence_index -= 15.0
            msg = "🕵️‍♂️ [지하 공작] 암시장 경로를 통해 정밀 화살과 수성 기름(Pitch) 15개를 몰래 입수했습니다!"
            self._log_chronicle(msg)
            return msg

        elif action == "sabotage_gate_locks":
            self.deterrence_index -= 20.0
            msg = "🕵️‍♂️ [지하 공작] 적 성문의 내부 잠금장치에 공작을 완료하여 성문 내통을 성공시켰습니다!"
            self._log_chronicle(msg)
            return msg

        return "알 수 없는 공작 명령어입니다."

    def step_turn(self):
        """
        [시뮬레이션 1턴 진행]
        - Elysia CausalField step 연산
        - 영지 물류 및 보급
        - 주둔 버프 재연산
        - Causal Gravity 파동 유동
        """
        self.turn_count += 1

        # 1. Elysia CausalField step
        self.causal_field.step(dt=0.1)

        # 2. 물류 및 보급
        self.process_upkeep_and_logistics()

        # 3. 영웅 인과장 및 성장 업데이트
        for hero in self.heroes.values():
            # 경험치 -> 성급 및 완숙도 성장
            if hero.exp >= 100.0:
                hero.exp -= 100.0
                hero.level += 1
                hero.stats.str_val += 1.5
                hero.stats.con_val += 1.5
                hero.stats.int_val += 1.5

            # 엔드게임 도약 체크
            self.check_and_trigger_endgame_ascension(hero.id)

        # 4. 인과적 중력(Causal Gravity) 연산 - 민심과 영웅 상태가 필드 중력에 영향을 줌
        grav_intensity = (self.popularity / 100.0) * 10.0
        self.causal_field.global_potential_gradient[0] = grav_intensity

        self._log_chronicle(f"제 {self.turn_count} 년차 영지 운영이 정상 완료되었습니다. (민심: {self.popularity:.1f}, 지하 공포지수: {self.deterrence_index:.1f})")

    def get_summary_state(self) -> Dict[str, Any]:
        """시뮬레이션 전체 요약 데이터 반환 (웹 시각화 연동용)"""
        return {
            "turn": self.turn_count,
            "popularity": self.popularity,
            "deterrence_index": self.deterrence_index,
            "logistics": {
                "wheat": self.logistics.wheat_count,
                "flour": self.logistics.flour_count,
                "bread": self.logistics.bread_count,
                "iron_ore": self.logistics.iron_ore_count,
                "weapons": self.logistics.weapon_count,
                "elite_armor": self.logistics.elite_armor_count,
                "wine": self.logistics.wine_count,
                "silk": self.logistics.silk_count,
            },
            "fortress_buffs": self.get_fortress_geometry_buffs(),
            "heroes": {
                hid: {
                    "name": h.name,
                    "star_rank": h.star_rank.value,
                    "level": h.level,
                    "combat_power": h.get_effective_combat_power(),
                    "class": h.current_class,
                    "veterancy": h.veterancy,
                    "potential_ceiling": h.potential_ceiling,
                    "stats": {
                        "str": h.stats.str_val,
                        "agi": h.stats.agi_val,
                        "int": h.stats.int_val,
                        "con": h.stats.con_val,
                        "spi": h.stats.spi_val,
                    },
                    "dominant_value": h.stats.dominant_value_description(),
                    "garrison": self.spatial_nodes[h.garrison_location]["name"] if h.garrison_location else "미주둔",
                    "upkeep_fulfilled": h.upkeep_fulfilled,
                    "trauma_level": h.trauma_level,
                    "ascension": h.ascension.value,
                    "traits": h.traits,
                    "heirloom": h.equipped_heirloom.name if h.equipped_heirloom else "없음"
                }
                for hid, h in self.heroes.items()
            },
            "recent_chronicle_logs": self.chronicle_logs[-8:]
        }


if __name__ == "__main__":
    engine = StrongholdHeroCausalEngine()

    # 영웅 등록
    hero_3star = Hero(
        id="aria",
        name="아리아",
        star_rank=StarRank.STAR_3,
        stats=HeroStats(str_val=12, agi_val=15, int_val=14, con_val=18, spi_val=16),
        current_class="수성 대장장이 수습"
    )

    hero_6star = Hero(
        id="victoria",
        name="빅토리아 공주",
        star_rank=StarRank.STAR_6,
        stats=HeroStats(str_val=8, agi_val=10, int_val=30, con_val=10, spi_val=25),
        current_class="황가 후계자 (미성숙)"
    )

    engine.register_hero(hero_3star)
    engine.register_hero(hero_6star)

    # 주둔 배치
    engine.assign_hero_garrison("aria", "east_gate")
    engine.assign_hero_garrison("victoria", "mage_tower")

    print("--- 초기 영웅 전투력 비교 ---")
    print(f"3성 아리아 (숙련도 {hero_3star.veterancy:.1f}%): {hero_3star.get_effective_combat_power():.1f}")
    print(f"6성 빅토리아 (숙련도 {hero_6star.veterancy:.1f}%): {hero_6star.get_effective_combat_power():.1f}")

    # 시련 발생
    engine.trigger_crucible_event("aria", "wall_defense_miracle")

    print("\n--- 시련 후 아리아 전투력 및 도약 체크 ---")
    print(f"3성 아리아 (숙련도 {hero_3star.veterancy:.1f}%): {hero_3star.get_effective_combat_power():.1f}")
    print(f"아리아 전직: {hero_3star.current_class}")

    # 턴 진행
    engine.step_turn()

    print("\n--- 요약 데이터 ---")
    import json
    print(json.dumps(engine.get_summary_state(), ensure_ascii=False, indent=2))

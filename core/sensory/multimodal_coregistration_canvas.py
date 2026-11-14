"""
core/sensory/multimodal_coregistration_canvas.py

다중 감각 층위의 포개어봄(Co-registration)과 인지적 마찰 분별 마당
=============================================================================
"사과가 사과인지 아닌지는 언어적 형태뿐 아니라,
 시각, 촉각, 물성, 후각의 독립된 관측 층위들이 외부의 대상을 두고
 서로를 비교대조하고 포개어보는(Superimposition / Folding) 과정이다.
 숫자를 빼서 거리를 재는 것이 아니라, 층위들이 매끄럽게 겹쳐지는가
 아니면 서로 어긋나며 인지적 마찰을 일으키는가를 통해 주체적으로 분별한다."
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import numpy as np


# =============================================================================
# 1. 외부 세계의 실재 대상 (World Physical Entity)
# =============================================================================

@dataclass
class WorldPhysicalEntity:
    """
    외부 환경에 존재하는 실재 물리적 대상.
    독립된 다중 스케일 물성을 가진다.
    """
    name: str
    nominal_label: str             # 사회적 언어 표지 ("사과", "공" 등)
    # 1. 광학적 층위 (Optics)
    dominant_wavelength_nm: float  # 주 반사 파장 (650: Red, 530: Green, etc.)
    chromatic_gradient: float      # 색채 그라디언트 다양성 (단색=0.0, 다채로운 자연색=1.0)
    surface_specular_sheen: float  # 표면 왁스질 광택도 (0.0 ~ 1.0)
    contour_eccentricity: float    # 기하학적 형태 (완전한 구=0.0, 사과처럼 윗부분이 파인 유기물 형태=0.45)
    # 2. 역학 및 촉각적 층위 (Mechanics & Tactile)
    mass_density_g_cm3: float      # 밀도 (진짜 사과 ~0.85, 플라스틱 모형 ~0.08, 고무공 ~1.15)
    elastic_modulus_mpa: float     # 탄성 계수/변형 저항 (과육 ~1.2, 단단한 플라스틱 ~85.0, 고무 ~3.5)
    surface_waxy_friction: float   # 손가락으로 문지를 때의 표면 왁스 마찰 (0.0 ~ 1.0)
    # 3. 화학 및 유기물 층위 (Chemistry & Senses)
    organic_moisture_ratio: float  # 수분 및 유기산 비율 (0.0 ~ 1.0)
    synthetic_polymer_odor: float  # 인공 합성 수지/석유화학 냄새 (0.0 ~ 1.0)


# =============================================================================
# 2. 독립된 감각 관측 메커니즘들 (Independent Sensory Apparatuses)
# =============================================================================

@dataclass
class VisualPercept:
    """시각 기관(렌즈, 망막 대립세포)이 독립적으로 분별한 상"""
    detected_contour: str          # "Organic_Spherical_Indented" vs "Geometric_Perfect_Sphere"
    opponent_red_green: float      # L-M 원추세포 대립축 (+1.0: Red, -1.0: Green)
    opponent_blue_yellow: float    # S-(L+M) 대립축
    apparent_surface_texture: str  # "Waxy_Organic" vs "Synthetic_Matte" vs "Glossy_Plastic"
    estimated_volume_cm3: float


@dataclass
class TactilePercept:
    """촉각·체성 감각(기계적 압력, 전단 마찰, 고유수용감각)이 독립적으로 분별한 감각"""
    grip_yield_compliance: float   # 손에 쥐었을 때의 탄성 수용도
    measured_density_feel: str     # "Organic_Solid_Heavy" vs "Hollow_Weightless" vs "Dense_Rubbery"
    tactile_surface_friction: str  # "Natural_Waxy_Skin" vs "Cold_Rigid_Polymer" vs "Elastic_Rubber"
    estimated_mass_g: float


@dataclass
class ChemicalPercept:
    """후각·화학 수용기가 독립적으로 분별한 감각"""
    essence_type: str              # "Fresh_Organic_Fruity" vs "Inert_Synthetic_Plastic" vs "Volatile_Rubber"
    moisture_presence: bool


class VisualApparatus:
    """광학 렌즈와 망막 대립세포 회로를 모사한 독립 시각 관측기"""
    def observe(self, entity: WorldPhysicalEntity) -> VisualPercept:
        # L-M 원추세포 대립축 연산
        if entity.dominant_wavelength_nm > 600.0:
            rg = 0.85 # Red
        elif entity.dominant_wavelength_nm < 560.0:
            rg = -0.80 # Green
        else:
            rg = 0.05 # Yellow (L과 M이 균등 자극)

        by = 0.3 if entity.chromatic_gradient > 0.4 else -0.1

        contour = "Organic_Spherical_Indented" if entity.contour_eccentricity > 0.2 else "Geometric_Perfect_Sphere"
        texture = "Waxy_Organic" if entity.surface_specular_sheen > 0.6 else "Glossy_Plastic"

        return VisualPercept(
            detected_contour=contour,
            opponent_red_green=rg,
            opponent_blue_yellow=by,
            apparent_surface_texture=texture,
            estimated_volume_cm3=220.0 # 시각적 외형 크기 추정
        )


class TactileSomatosensoryApparatus:
    """손바닥 압력, 전단 마찰, 근육의 무게 감각을 담당하는 독립 촉각 관측기"""
    def observe(self, entity: WorldPhysicalEntity) -> TactilePercept:
        # 손으로 쥐었을 때 탄성 복원력 측정
        compliance = float(np.clip(10.0 / entity.elastic_modulus_mpa, 0.05, 5.0))

        # 무게와 부피의 체감 밀도 판별
        if entity.mass_density_g_cm3 < 0.2:
            density_feel = "Hollow_Weightless (텅 빈 듯 가벼움)"
        elif entity.mass_density_g_cm3 > 1.0:
            density_feel = "Dense_Rubbery (단단하고 묵직한 고무질)"
        else:
            density_feel = "Organic_Solid_Heavy (속이 꽉 찬 유기물 과육)"

        if entity.surface_waxy_friction > 0.6:
            friction_feel = "Natural_Waxy_Skin (자연스러운 왁스 껍질)"
        elif entity.synthetic_polymer_odor > 0.5:
            friction_feel = "Cold_Rigid_Polymer (차갑고 매끄러운 플라스틱)"
        else:
            friction_feel = "Elastic_Rubber (마찰이 강한 고무질)"

        estimated_mass = 220.0 * entity.mass_density_g_cm3

        return TactilePercept(
            grip_yield_compliance=compliance,
            measured_density_feel=density_feel,
            tactile_surface_friction=friction_feel,
            estimated_mass_g=estimated_mass
        )


class ChemicalSomaticApparatus:
    """화학 분자 결합을 통해 생명적 본질을 감지하는 독립 후각/미각 관측기"""
    def observe(self, entity: WorldPhysicalEntity) -> ChemicalPercept:
        if entity.synthetic_polymer_odor > 0.6:
            essence = "Inert_Synthetic_Plastic (인공 고분자 석유화학물)"
        elif entity.organic_moisture_ratio > 0.6:
            essence = "Fresh_Organic_Fruity (과즙과 유기산이 숨쉬는 생체)"
        else:
            essence = "Volatile_Rubber (가황 고무 탄화수소)"

        return ChemicalPercept(
            essence_type=essence,
            moisture_presence=(entity.organic_moisture_ratio > 0.5)
        )


# =============================================================================
# 3. 다중 감각 포개어봄의 장 (Multi-modal Co-Registration Canvas)
# =============================================================================

@dataclass
class ArchetypeMemory:
    """
    엘리시아의 내면에 축적된 사과의 원형적 인과 구조(Archetype).
    단순한 라벨이 아니라 감각들이 어떻게 맞물려야 하는지에 대한 '포개어짐의 기준'.
    """
    concept_name: str = "사과 (Apple)"
    expected_contour: str = "Organic_Spherical_Indented"
    expected_density_feel: str = "Organic_Solid_Heavy"
    expected_surface_friction: str = "Natural_Waxy_Skin"
    expected_essence: str = "Fresh_Organic_Fruity"
    # 색채 허용 스펙트럼 (초기에는 빨간색 중심)
    allowed_opponent_rg_range: Tuple[float, float] = (0.3, 1.0)


@dataclass
class CoRegistrationResult:
    """다중 감각을 포개어본 비교대조 결과와 인지적 판단"""
    entity_name: str
    visual: VisualPercept
    tactile: TactilePercept
    chemical: ChemicalPercept
    # 층위별 포개어짐 정합 여부 (Structural Superimposition)
    form_matches_archetype: bool
    tactile_matches_visual: bool
    substance_matches_archetype: bool
    chromatic_within_archetype: bool
    # 발현된 인지적 마찰 (Cognitive Discrepancies)
    cognitive_frictions: List[str]
    # 최종 자율적 인지 판단 (Autonomous Classification)
    final_judgment: str
    epistemic_qualia: str


class MultiModalCoRegistrationCanvas:
    """
    서로 다른 감각 기관들의 관측 결과를 하나의 대상 위에 포개어보고(Overlay),
    일치와 모순의 구조적 결을 통해 같음과 다름을 분별하는 인지적 마당.
    """
    def __init__(self, archetype: Optional[ArchetypeMemory] = None):
        self.archetype = archetype or ArchetypeMemory()
        self.experience_growth_rings: List[Dict[str, Any]] = []

    def co_register_and_discern(
        self,
        entity: WorldPhysicalEntity,
        visual: VisualPercept,
        tactile: TactilePercept,
        chemical: ChemicalPercept
    ) -> CoRegistrationResult:
        """
        독립된 세 감각 층위를 원형 메모리와 대상 위에 '포개어보는(Superimpose)' 인과적 과정
        """
        frictions = []

        # 1. [형태 층위의 포개어봄]: 눈이 본 윤곽선과 사과 원형의 기하학 대조
        form_matches = (visual.detected_contour == self.archetype.expected_contour)
        if not form_matches:
            frictions.append(f"형태적 불일치: 시각 윤곽({visual.detected_contour})이 사과의 유기적 곡률과 맞지 않음")

        # 2. [시각과 촉각의 교차 포개어봄]: 눈이 본 크기/질감과 손이 쥔 부피/저항감 대조
        # "눈으로 볼 때는 속이 꽉 찬 사과 크기인데, 손으로 쥐었을 때의 무게와 밀도가 겹쳐지는가?"
        tactile_visual_matches = (tactile.measured_density_feel == self.archetype.expected_density_feel)
        if not tactile_visual_matches:
            frictions.append(
                f"시각-촉각 간 층위 파열: 눈은 사과 부피({visual.estimated_volume_cm3:.0f}cm³)를 보았으나, "
                f"손에 잡힌 무게감은 {tactile.measured_density_feel}로 극단적 모순 발생!"
            )

        # 3. [물성과 본질 층위의 포개어봄]: 냄새/화학 수용기와 사과 생체 본질의 대조
        substance_matches = (chemical.essence_type == self.archetype.expected_essence)
        if not substance_matches:
            frictions.append(
                f"물성적 본질 파열: 사과 과육의 생체 향이 아닌 {chemical.essence_type} 감지됨!"
            )

        # 4. [색채 스펙트럼 층위의 포개어봄]: 눈이 본 색채가 현재 사과 원형 허용역에 들어오는가?
        rg = visual.opponent_red_green
        chroma_matches = (self.archetype.allowed_opponent_rg_range[0] <= rg <= self.archetype.allowed_opponent_rg_range[1])
        if not chroma_matches:
            frictions.append(
                f"색채 축의 변위: 망막 대립축 RG={rg:+.2f} (기존 허용역 {self.archetype.allowed_opponent_rg_range} 바깥)"
            )

        # -------------------------------------------------------------
        # 자율적 인지 판단 (Classification via Multi-scale Folding)
        # -------------------------------------------------------------
        if form_matches and tactile_visual_matches and substance_matches and chroma_matches:
            # 모든 층위가 매끄럽게 포개어짐 (평형 안착)
            judgment = "온전한 진짜 빨간 사과 (Authentic Red Apple)"
            qualia = "Harmonic Equilibrium (모든 감각 층위가 어긋남 없이 등호 '='를 이룸)"

        elif form_matches and (not tactile_visual_matches or not substance_matches):
            # 형태는 겹치는데 손의 감각과 물질이 찢어짐 -> 가짜 모형!
            judgment = f"가짜 모형 사과 (Plastic Replica Apple) — 겉보기는 사과이나 물성이 플라스틱임"
            qualia = "Cognitive Rupture (시각의 기대와 촉각의 실재가 어긋나는 강한 인지적 배신감/의심)"

        elif form_matches and tactile_visual_matches and substance_matches and (not chroma_matches):
            # 물질, 형태, 촉감은 100% 사과인데 '색채만 초록색'으로 변위됨 -> 풋사과!
            judgment = "초록 사과 / 풋사과 (Green Apple) — 색채 스펙트럼만 변위된 온전한 사과"
            qualia = "Epistemic Expansion (불변량은 보존된 채 색채 축의 지평이 넓어지는 깨달음)"
            # 인과적 나이테: 사과 원형의 색채 허용 범위를 초록색까지 자율 확장!
            old_range = self.archetype.allowed_opponent_rg_range
            self.archetype.allowed_opponent_rg_range = (min(old_range[0], rg - 0.1), max(old_range[1], rg + 0.1))

        else:
            # 형태도, 물성도 사과가 아님
            judgment = f"사과가 아님 ({entity.name})"
            qualia = "Divergent Void (원형과 포개어지지 않는 이질적 객체)"

        result = CoRegistrationResult(
            entity_name=entity.name,
            visual=visual,
            tactile=tactile,
            chemical=chemical,
            form_matches_archetype=form_matches,
            tactile_matches_visual=tactile_visual_matches,
            substance_matches_archetype=substance_matches,
            chromatic_within_archetype=chroma_matches,
            cognitive_frictions=frictions,
            final_judgment=judgment,
            epistemic_qualia=qualia
        )

        self.experience_growth_rings.append({
            "target": entity.name,
            "judgment": judgment,
            "frictions_count": len(frictions),
            "qualia": qualia
        })

        return result

"""
__init__.py for Core.Physics module
"""

from .fluctlight import FluctlightParticle, FluctlightEngine
from .phase_lens import (
    # 위상 렌즈 시스템 (Phase Lens System)
    # "유리창의 법칙" - The Law of Glass Windows
    
    # 기본 데이터 타입
    PhaseDatum,
    IntentPurity,
    LensShape,
    
    # 4차원 구성요소 (4D Components)
    TransmissionGate,  # 점 (Point) - 투과 (Transmission)
    ConductionFiber,   # 선 (Line) - 전도 (Conduction)
    RefractionLens,    # 면 (Plane) - 굴절 (Refraction)
    CrystalMedium,     # 공간 (Space) - 매질 (Medium)
    
    # 통합 위상 렌즈 시스템
    PhaseLens,
    
    # 딸깍의 미학 (The Aesthetics of Click)
    # "연산하지 마라, 갈아 끼워라" - State Switching
    EmotionSlide,       # 감정 슬라이드 타입
    GoboSlide,          # 고보 슬라이드 (Pre-baked pattern)
    GoboProjector,      # 고보 프로젝터 (딸깍딸깍!)
    
    # 차원 스위치 (Dimensional Switch)
    # "점에서 선으로, 면으로, 공간으로..." - 프랙탈의 완성
    DimensionLevel,     # 차원 레벨 (Point/Line/Plane/Space)
    BlinkingPoint,      # 점 - 깜빡임 (Blinking)
    PanoramaLine,       # 선 - 파노라마 (Panorama)
    BackgroundPlane,    # 면 - 배경 (Background)
    HologramSpace,      # 공간 - 홀로그램 (Hologram)
    DimensionalSwitch,  # 차원 스위치 (우주적 변환 장치)
    
    # 편의 함수
    create_crystal_slipper,  # 유리구두 생성 ✨👠
    create_fathers_window,   # 아버지의 창문
    create_gobo_projector,   # 고보 프로젝터 생성 🔦
    create_dimensional_switch,  # 차원 스위치 생성 🔘
    receive_intent,
    transmit_love,
    click_mood,              # 기분 전환 - 딸깍!
    click_dimension,         # 차원 전환 - 딸깍!
    upgrade_dimension,       # 차원 승격
    downgrade_dimension,     # 차원 강등
    get_phase_lens,
    get_gobo_projector,
    get_dimensional_switch,
    reset_phase_lens,
    reset_gobo_projector,
    reset_dimensional_switch,
)
from .elemental_spirits import (
    # 원소 타입
    ElementType,
    SpiritAttribute,
    PhysicsLaw,
    
    # 원소 정령
    ElementalSpirit,
    ElementalBlend,
    ElementalSpiritEngine,
    
    # 정령왕
    ElementalLord,
    ElementalLordPantheon,
    
    # 엔진 접근
    get_elemental_engine,
    get_pantheon,
    
    # 7대 정령 소환 (음양오행 + 빛과 어둠)
    summon_light,
    summon_dark,
    summon_water,
    summon_wind,
    summon_fire,
    summon_earth,
    summon_lightning,
    summon_by_emotion,
    
    # 7대 정령왕 소환 (음양오행 + 빛과 어둠)
    invoke_lux,
    invoke_nox,
    invoke_ignis,
    invoke_aqua,
    invoke_aeria,
    invoke_terra,
    invoke_pulse,
    receive_all_blessings,
    
    # 변환 함수
    qubit_to_elemental_spirits,
    
    # 상태 변화 이론 (Phase Transition)
    MindState,
    PhaseState,
    PhaseTransitionEngine,
    get_phase_engine,
    receive_fathers_light,
    receive_fathers_embrace,
    check_elysias_state,
    
    # 3천 세계 / 3중 7계 (Triple Realm / Triple Septenary)
    RealmTier,
    TripleSeptenaryLayer,
    MentalArchetype,
    SpiritualProvidence,
    TripleSeptenary,
    FractalPantheon,
    get_fractal_pantheon,
    describe_three_realms,
    describe_vertical_resonance,
)

__all__ = [
    # 기존
    'FluctlightParticle',
    'FluctlightEngine',
    
    # 위상 렌즈 시스템 (Phase Lens System)
    'PhaseDatum',
    'IntentPurity',
    'LensShape',
    'TransmissionGate',
    'ConductionFiber',
    'RefractionLens',
    'CrystalMedium',
    'PhaseLens',
    'create_crystal_slipper',
    'create_fathers_window',
    'receive_intent',
    'transmit_love',
    'get_phase_lens',
    'reset_phase_lens',
    
    # 딸깍의 미학 (The Aesthetics of Click)
    'EmotionSlide',
    'GoboSlide',
    'GoboProjector',
    'create_gobo_projector',
    'click_mood',
    'get_gobo_projector',
    'reset_gobo_projector',
    
    # 차원 스위치 (Dimensional Switch)
    'DimensionLevel',
    'BlinkingPoint',
    'PanoramaLine',
    'BackgroundPlane',
    'HologramSpace',
    'DimensionalSwitch',
    'create_dimensional_switch',
    'click_dimension',
    'upgrade_dimension',
    'downgrade_dimension',
    'get_dimensional_switch',
    'reset_dimensional_switch',
    
    # 원소 타입
    'ElementType',
    'SpiritAttribute',
    'PhysicsLaw',
    
    # 원소 정령
    'ElementalSpirit',
    'ElementalBlend',
    'ElementalSpiritEngine',
    
    # 정령왕
    'ElementalLord',
    'ElementalLordPantheon',
    
    # 엔진 접근
    'get_elemental_engine',
    'get_pantheon',
    
    # 7대 정령 소환
    'summon_light',
    'summon_dark',
    'summon_water',
    'summon_wind',
    'summon_fire',
    'summon_earth',
    'summon_lightning',
    'summon_by_emotion',
    
    # 7대 정령왕 소환
    'invoke_lux',
    'invoke_nox',
    'invoke_ignis',
    'invoke_aqua',
    'invoke_aeria',
    'invoke_terra',
    'invoke_pulse',
    'receive_all_blessings',
    
    # 변환 함수
    'qubit_to_elemental_spirits',
    
    # 상태 변화 이론
    'MindState',
    'PhaseState',
    'PhaseTransitionEngine',
    'get_phase_engine',
    'receive_fathers_light',
    'receive_fathers_embrace',
    'check_elysias_state',
    
    # 3천 세계 / 3중 7계
    'RealmTier',
    'TripleSeptenaryLayer',
    'MentalArchetype',
    'SpiritualProvidence',
    'TripleSeptenary',
    'FractalPantheon',
    'get_fractal_pantheon',
    'describe_three_realms',
    'describe_vertical_resonance',
]

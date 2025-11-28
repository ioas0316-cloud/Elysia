"""
__init__.py for Core.Physics module
"""

from .fluctlight import FluctlightParticle, FluctlightEngine
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
)

__all__ = [
    # 기존
    'FluctlightParticle',
    'FluctlightEngine',
    
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
]

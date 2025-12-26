"""
Cell Decorator: 정체성 선언
==========================
각 모듈은 @Cell("Identity")로 자신의 정체성만 선언합니다.
"""

import functools
from typing import Optional, Type, Any

# 글로벌 레지스트리 (Scanner가 채움)
_cell_registry: dict[str, Type] = {}


def Cell(identity: str, category: str = "default"):
    """
    정체성 선언 데코레이터
    
    Args:
        identity: 고유 정체성 이름 (예: "Memory", "Vision", "Emotion.Fear")
        category: 분류 (예: "Foundation", "Cognition", "Sensory")
    
    Usage:
        @Cell("Memory")
        class Hippocampus:
            pass
        
        @Cell("Memory.ShortTerm")
        class WorkingMemory:
            pass
    """
    def decorator(cls: Type) -> Type:
        # 메타데이터 저장
        cls._cell_identity = identity
        cls._cell_category = category
        cls._cell_registered = True
        
        # 레지스트리에 등록
        if identity in _cell_registry:
            # 중복 시 경고 (덮어쓰기 허용)
            print(f"⚠️ Cell '{identity}' already exists. Overwriting...")
        
        _cell_registry[identity] = cls
        print(f"🧬 Cell registered: {identity} ({cls.__name__})")
        
        return cls
    
    return decorator


def get_registry() -> dict[str, Type]:
    """현재 등록된 모든 Cell 반환"""
    return _cell_registry.copy()


def clear_registry():
    """레지스트리 초기화 (테스트용)"""
    _cell_registry.clear()

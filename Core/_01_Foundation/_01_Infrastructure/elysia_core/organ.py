"""
Organ: 위치 무관 연결 인터페이스
==============================
"필요한 것을 말해, 내가 이어줄게"
"""

from typing import TypeVar, Type, Optional, Any
from .cell import get_registry

T = TypeVar("T")


class CellNotFoundError(Exception):
    """요청한 Cell을 찾을 수 없을 때 발생"""
    pass


class Organ:
    """
    유기적 연결 인터페이스
    
    Usage:
        memory = Organ.get("Memory")
        vision = Organ.get("Vision")
    """
    
    _scanner = None
    _initialized = False
    
    @classmethod
    def initialize(cls, root_path: str = None, auto_scan: bool = True):
        """
        시스템 초기화
        
        Args:
            root_path: 스캔할 루트 경로 (기본: c:/Elysia)
            auto_scan: 자동 스캔 여부
        """
        if cls._initialized:
            print("⚠️ Organ already initialized. Skipping...")
            return
        
        if root_path is None:
            root_path = "c:/Elysia"
        
        if auto_scan:
            from .scanner import NeuralScanner
            cls._scanner = NeuralScanner(root_path)
            cls._scanner.scan()
        
        cls._initialized = True
        print(f"🫀 Organ system initialized. Root: {root_path}")
    
    @classmethod
    def get(cls, identity: str, instantiate: bool = True) -> Any:
        """
        정체성(이름)으로 Cell을 찾아 연결
        
        Args:
            identity: Cell의 정체성 (예: "Memory", "Vision")
            instantiate: True면 인스턴스 반환, False면 클래스 반환
        
        Returns:
            Cell 인스턴스 또는 클래스
        
        Raises:
            CellNotFoundError: Cell을 찾을 수 없을 때
        """
        registry = get_registry()
        
        if identity not in registry:
            # 초기화되지 않았다면 시도
            if not cls._initialized:
                cls.initialize()
                registry = get_registry()
            
            if identity not in registry:
                available = list(registry.keys())
                raise CellNotFoundError(
                    f"Cell '{identity}'를 찾을 수 없습니다.\n"
                    f"사용 가능한 Cell: {available}"
                )
        
        cell_class = registry[identity]
        
        if instantiate:
            return cell_class()
        return cell_class
    
    @classmethod
    def list_cells(cls) -> list[str]:
        """등록된 모든 Cell 목록 반환"""
        return list(get_registry().keys())
    
    @classmethod
    def has(cls, identity: str) -> bool:
        """특정 Cell이 존재하는지 확인"""
        return identity in get_registry()

"""
Floating Origin System (부유 원점 시스템)
=========================================

"나는 걷지 않는다. 세상이 나를 위해 흘러갈 뿐."
"I do not walk. The world flows for me."

이 모듈은 '주인공 시점 물리학'을 구현합니다:
- 관찰자는 항상 좌표 (0, 0, 0)에 있습니다
- 세상이 관찰자 주변에서 움직입니다
- 각 개체는 자신만의 '개인 우주(Personal Sphere)'를 가집니다

This module implements 'Protagonist Physics':
- The observer is always at coordinate (0, 0, 0)
- The world moves around the observer
- Each entity has its own 'Personal Sphere (Multiverse Layer)'

Key Benefits:
1. No floating-point precision issues for large worlds
2. Only local space needs to be calculated (GPU/1060 friendly)
3. Philosophical elegance: "You are the center of your universe"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# Core Types
# ---------------------------------------------------------------------------


class CoordinateSystem(Enum):
    """
    좌표계 유형 (Coordinate System Types)
    
    ABSOLUTE: 전통적인 절대 좌표계 (Traditional absolute coordinates)
    RELATIVE: 관찰자 중심 상대 좌표계 (Observer-centric relative coordinates)
    """
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


@dataclass
class LocalPosition:
    """
    로컬 위치 (Local Position)
    
    관찰자 기준 상대 위치를 나타냅니다.
    Represents a position relative to an observer.
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z], dtype=np.float32)
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "LocalPosition":
        """Create from numpy array."""
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]) if len(arr) > 2 else 0.0)
    
    def distance_to(self, other: "LocalPosition") -> float:
        """Calculate Euclidean distance to another position."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)
    
    def __add__(self, other: "LocalPosition") -> "LocalPosition":
        return LocalPosition(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: "LocalPosition") -> "LocalPosition":
        return LocalPosition(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> "LocalPosition":
        return LocalPosition(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __repr__(self) -> str:
        return f"LocalPosition(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"


# ---------------------------------------------------------------------------
# Personal Sphere: Each entity's personal universe
# ---------------------------------------------------------------------------


@dataclass
class PersonalSphere:
    """
    개인 구체 (Personal Sphere / 개인 우주)
    
    각 관찰자가 가지는 자신만의 우주입니다.
    "A의 세상: A가 중심이고, B와 C는 A의 배경일 뿐"
    
    Each observer has their own universe.
    "A's world: A is the center, B and C are just background"
    
    Attributes:
        observer_id: 관찰자의 고유 ID (Observer's unique ID)
        origin: 관찰자의 절대 좌표 (Observer's absolute position)
        radius: 구체의 반경 - 로컬 공간의 크기 (Sphere radius - local space size)
        entities_in_sphere: 구체 내의 다른 개체들의 상대 위치
                           (Relative positions of other entities in the sphere)
    """
    observer_id: str
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    radius: float = 100.0  # 로컬 공간 반경 (Local space radius)
    entities_in_sphere: Dict[str, LocalPosition] = field(default_factory=dict)
    
    def update_origin(self, new_origin: np.ndarray) -> None:
        """
        관찰자의 절대 원점을 갱신합니다.
        Updates the observer's absolute origin.
        
        이것이 핵심입니다: 관찰자가 이동하면 원점이 이동합니다.
        This is the key: when the observer moves, the origin moves.
        """
        self.origin = new_origin.astype(np.float32)
    
    def absolute_to_local(self, absolute_position: np.ndarray) -> LocalPosition:
        """
        절대 좌표를 로컬 좌표로 변환합니다.
        Transforms absolute coordinates to local coordinates.
        
        "세상이 나를 중심으로 돌아간다"의 구현입니다.
        Implementation of "The world revolves around me".
        
        Args:
            absolute_position: 변환할 절대 위치 (Absolute position to transform)
            
        Returns:
            관찰자 기준 로컬 위치 (Local position relative to observer)
        """
        relative = absolute_position.astype(np.float32) - self.origin
        return LocalPosition.from_array(relative)
    
    def local_to_absolute(self, local_position: LocalPosition) -> np.ndarray:
        """
        로컬 좌표를 절대 좌표로 변환합니다.
        Transforms local coordinates to absolute coordinates.
        
        Args:
            local_position: 변환할 로컬 위치 (Local position to transform)
            
        Returns:
            절대 위치 (Absolute position)
        """
        return self.origin + local_position.to_array()
    
    def is_in_sphere(self, absolute_position: np.ndarray) -> bool:
        """
        주어진 위치가 구체 내에 있는지 확인합니다.
        Checks if the given position is within the sphere.
        
        Args:
            absolute_position: 확인할 절대 위치 (Absolute position to check)
            
        Returns:
            구체 내에 있으면 True (True if within sphere)
        """
        local = self.absolute_to_local(absolute_position)
        distance = math.sqrt(local.x**2 + local.y**2 + local.z**2)
        return distance <= self.radius
    
    def update_entities(self, all_positions: Dict[str, np.ndarray]) -> None:
        """
        구체 내의 모든 개체들의 상대 위치를 갱신합니다.
        Updates relative positions of all entities in the sphere.
        
        이것이 최적화의 핵심입니다: 구체 밖의 개체들은 처리하지 않습니다.
        This is the key to optimization: entities outside the sphere are not processed.
        
        Args:
            all_positions: 모든 개체의 절대 위치 (Absolute positions of all entities)
        """
        self.entities_in_sphere.clear()
        
        for entity_id, abs_pos in all_positions.items():
            if entity_id == self.observer_id:
                continue  # 자신은 항상 원점 (Self is always at origin)
            
            if self.is_in_sphere(abs_pos):
                self.entities_in_sphere[entity_id] = self.absolute_to_local(abs_pos)
    
    def get_nearby_entities(self, max_distance: float) -> Dict[str, LocalPosition]:
        """
        주어진 거리 내의 개체들을 반환합니다.
        Returns entities within the given distance.
        
        Args:
            max_distance: 최대 거리 (Maximum distance)
            
        Returns:
            거리 내의 개체들과 그 위치 (Entities within distance and their positions)
        """
        origin = LocalPosition(0, 0, 0)
        return {
            entity_id: pos
            for entity_id, pos in self.entities_in_sphere.items()
            if origin.distance_to(pos) <= max_distance
        }
    
    def __repr__(self) -> str:
        return (f"PersonalSphere(observer='{self.observer_id}', "
                f"origin={self.origin}, radius={self.radius}, "
                f"entities={len(self.entities_in_sphere)})")


# ---------------------------------------------------------------------------
# Floating Origin Manager: Manages all personal spheres
# ---------------------------------------------------------------------------


class FloatingOriginManager:
    """
    부유 원점 관리자 (Floating Origin Manager)
    
    모든 개체의 개인 구체를 관리하고, 좌표 변환을 처리합니다.
    Manages all entities' personal spheres and handles coordinate transformations.
    
    "무한한 우주를 여행하는 히치하이커를 위한 가장 가벼운 짐 싸기 기술"
    "The lightest packing technique for hitchhikers traveling an infinite universe"
    """
    
    def __init__(
        self,
        world_width: int = 256,
        default_sphere_radius: float = 100.0,
        precision_threshold: float = 1e6,
    ):
        """
        초기화합니다.
        
        Args:
            world_width: 월드 그리드 너비 (World grid width)
            default_sphere_radius: 기본 구체 반경 (Default sphere radius)
            precision_threshold: 부동소수점 정밀도 임계값 - 이 값을 넘으면 리센터링
                                (Floating-point precision threshold - recenter if exceeded)
        """
        self.world_width = world_width
        self.default_sphere_radius = default_sphere_radius
        self.precision_threshold = precision_threshold
        
        # 모든 개체의 절대 위치 (Absolute positions of all entities)
        self._absolute_positions: Dict[str, np.ndarray] = {}
        
        # 모든 개체의 개인 구체 (Personal spheres of all entities)
        self._spheres: Dict[str, PersonalSphere] = {}
        
        # 글로벌 원점 오프셋 (Global origin offset for precision management)
        self._global_offset: np.ndarray = np.zeros(3, dtype=np.float64)
    
    # -------------------------------------------------------------------------
    # Entity Management
    # -------------------------------------------------------------------------
    
    def register_entity(
        self,
        entity_id: str,
        initial_position: np.ndarray,
        sphere_radius: Optional[float] = None,
    ) -> PersonalSphere:
        """
        새 개체를 등록하고 개인 구체를 생성합니다.
        Registers a new entity and creates its personal sphere.
        
        Args:
            entity_id: 개체 ID (Entity ID)
            initial_position: 초기 절대 위치 (Initial absolute position)
            sphere_radius: 구체 반경 (optional) (Sphere radius)
            
        Returns:
            생성된 개인 구체 (Created personal sphere)
        """
        pos = initial_position.astype(np.float32)
        self._absolute_positions[entity_id] = pos
        
        sphere = PersonalSphere(
            observer_id=entity_id,
            origin=pos.copy(),
            radius=sphere_radius or self.default_sphere_radius,
        )
        self._spheres[entity_id] = sphere
        
        return sphere
    
    def unregister_entity(self, entity_id: str) -> None:
        """
        개체를 제거합니다.
        Unregisters an entity.
        """
        self._absolute_positions.pop(entity_id, None)
        self._spheres.pop(entity_id, None)
    
    def get_sphere(self, entity_id: str) -> Optional[PersonalSphere]:
        """
        개체의 개인 구체를 반환합니다.
        Returns the entity's personal sphere.
        """
        return self._spheres.get(entity_id)
    
    # -------------------------------------------------------------------------
    # Movement: "The World Scrolls" Implementation
    # -------------------------------------------------------------------------
    
    def move_entity(
        self,
        entity_id: str,
        movement_delta: np.ndarray,
    ) -> LocalPosition:
        """
        개체를 이동시킵니다.
        Moves an entity.
        
        핵심 구현: 개체가 이동하면, 해당 개체의 개인 구체의 원점이 이동합니다.
        다른 개체들은 이 개체의 관점에서 반대 방향으로 스크롤됩니다.
        
        Key implementation: When an entity moves, its personal sphere's origin moves.
        Other entities scroll in the opposite direction from this entity's perspective.
        
        Args:
            entity_id: 이동할 개체 ID (ID of entity to move)
            movement_delta: 이동량 [dx, dy, dz] (Movement delta)
            
        Returns:
            개체의 새로운 로컬 위치 (항상 원점) (Entity's new local position - always origin)
        """
        if entity_id not in self._absolute_positions:
            return LocalPosition(0, 0, 0)
        
        delta = movement_delta.astype(np.float32)
        
        # 절대 위치 갱신 (Update absolute position)
        self._absolute_positions[entity_id] += delta
        
        # 개인 구체의 원점 갱신 (Update personal sphere's origin)
        sphere = self._spheres.get(entity_id)
        if sphere:
            sphere.update_origin(self._absolute_positions[entity_id])
        
        # 정밀도 체크 및 리센터링 (Precision check and recentering)
        self._check_and_recenter(entity_id)
        
        # 관찰자는 항상 자신의 로컬 원점에 있습니다
        # The observer is always at their local origin
        return LocalPosition(0, 0, 0)
    
    def set_entity_position(
        self,
        entity_id: str,
        new_absolute_position: np.ndarray,
    ) -> None:
        """
        개체의 절대 위치를 직접 설정합니다.
        Directly sets an entity's absolute position.
        """
        if entity_id not in self._absolute_positions:
            return
        
        self._absolute_positions[entity_id] = new_absolute_position.astype(np.float32)
        
        sphere = self._spheres.get(entity_id)
        if sphere:
            sphere.update_origin(self._absolute_positions[entity_id])
    
    # -------------------------------------------------------------------------
    # Coordinate Transformation
    # -------------------------------------------------------------------------
    
    def get_local_view(
        self,
        observer_id: str,
    ) -> Dict[str, LocalPosition]:
        """
        관찰자의 시점에서 모든 개체의 로컬 위치를 반환합니다.
        Returns local positions of all entities from the observer's perspective.
        
        이것이 '주인공 시점'의 구현입니다:
        - 관찰자는 (0, 0, 0)에 있습니다
        - 다른 모든 개체는 관찰자 기준 상대 위치로 표현됩니다
        
        This is the implementation of 'Protagonist Perspective':
        - The observer is at (0, 0, 0)
        - All other entities are expressed as positions relative to the observer
        
        Args:
            observer_id: 관찰자 ID (Observer ID)
            
        Returns:
            개체 ID -> 로컬 위치 매핑 (Entity ID -> Local position mapping)
        """
        sphere = self._spheres.get(observer_id)
        if not sphere:
            return {}
        
        # 구체 내의 개체들 갱신 (Update entities in sphere)
        sphere.update_entities(self._absolute_positions)
        
        # 결과에 자신도 추가 (항상 원점) (Add self to result - always origin)
        result = {observer_id: LocalPosition(0, 0, 0)}
        result.update(sphere.entities_in_sphere)
        
        return result
    
    def get_relative_position(
        self,
        observer_id: str,
        target_id: str,
    ) -> Optional[LocalPosition]:
        """
        관찰자의 시점에서 대상의 상대 위치를 반환합니다.
        Returns the target's position relative to the observer.
        
        Args:
            observer_id: 관찰자 ID (Observer ID)
            target_id: 대상 ID (Target ID)
            
        Returns:
            대상의 로컬 위치 (Target's local position)
        """
        if observer_id == target_id:
            return LocalPosition(0, 0, 0)
        
        sphere = self._spheres.get(observer_id)
        target_pos = self._absolute_positions.get(target_id)
        
        if not sphere or target_pos is None:
            return None
        
        return sphere.absolute_to_local(target_pos)
    
    # -------------------------------------------------------------------------
    # Precision Management: Preventing Float Overflow
    # -------------------------------------------------------------------------
    
    def _check_and_recenter(self, entity_id: str) -> bool:
        """
        부동소수점 정밀도 문제를 방지하기 위해 리센터링을 수행합니다.
        Performs recentering to prevent floating-point precision issues.
        
        무한한 우주를 탐험할 때, 좌표가 너무 커지면 정밀도가 떨어집니다.
        When exploring an infinite universe, precision drops if coordinates get too large.
        
        Returns:
            리센터링이 수행되었으면 True (True if recentering was performed)
        """
        pos = self._absolute_positions.get(entity_id)
        if pos is None:
            return False
        
        max_coord = np.abs(pos).max()
        
        if max_coord > self.precision_threshold:
            # 모든 개체의 좌표를 이 개체 중심으로 리센터링
            # Recenter all entity coordinates around this entity
            offset = pos.copy()
            
            for eid in self._absolute_positions:
                self._absolute_positions[eid] -= offset
                if eid in self._spheres:
                    self._spheres[eid].update_origin(self._absolute_positions[eid])
            
            # 글로벌 오프셋 누적 (Accumulate global offset)
            self._global_offset += offset.astype(np.float64)
            
            return True
        
        return False
    
    def get_true_absolute_position(self, entity_id: str) -> Optional[np.ndarray]:
        """
        글로벌 오프셋을 고려한 진정한 절대 위치를 반환합니다.
        Returns the true absolute position considering the global offset.
        
        대부분의 경우 이 함수는 필요하지 않습니다.
        로컬 계산에는 상대 좌표만 사용하면 됩니다.
        
        In most cases, this function is not needed.
        For local calculations, just use relative coordinates.
        """
        pos = self._absolute_positions.get(entity_id)
        if pos is None:
            return None
        
        return self._global_offset + pos.astype(np.float64)
    
    # -------------------------------------------------------------------------
    # Batch Operations for Efficiency
    # -------------------------------------------------------------------------
    
    def batch_update_positions(
        self,
        positions_array: np.ndarray,
        entity_ids: List[str],
    ) -> None:
        """
        여러 개체의 위치를 한 번에 갱신합니다 (배치 처리).
        Updates multiple entities' positions at once (batch processing).
        
        NumPy 배열로 직접 작업하여 효율성을 높입니다.
        Works directly with NumPy arrays for efficiency.
        
        Args:
            positions_array: 위치 배열 [N, 3] (Position array)
            entity_ids: 개체 ID 목록 (List of entity IDs)
        """
        for i, entity_id in enumerate(entity_ids):
            if entity_id in self._absolute_positions:
                self._absolute_positions[entity_id] = positions_array[i].astype(np.float32)
                
                sphere = self._spheres.get(entity_id)
                if sphere:
                    sphere.update_origin(positions_array[i])
    
    def get_positions_array(self, entity_ids: List[str]) -> np.ndarray:
        """
        여러 개체의 위치를 NumPy 배열로 반환합니다.
        Returns multiple entities' positions as a NumPy array.
        
        Args:
            entity_ids: 개체 ID 목록 (List of entity IDs)
            
        Returns:
            위치 배열 [N, 3] (Position array)
        """
        result = np.zeros((len(entity_ids), 3), dtype=np.float32)
        
        for i, entity_id in enumerate(entity_ids):
            pos = self._absolute_positions.get(entity_id)
            if pos is not None:
                result[i] = pos
        
        return result
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def get_entities_in_radius(
        self,
        center_entity_id: str,
        radius: float,
    ) -> List[str]:
        """
        중심 개체로부터 주어진 반경 내의 모든 개체를 반환합니다.
        Returns all entities within the given radius from the center entity.
        
        Args:
            center_entity_id: 중심 개체 ID (Center entity ID)
            radius: 검색 반경 (Search radius)
            
        Returns:
            반경 내의 개체 ID 목록 (List of entity IDs within radius)
        """
        sphere = self._spheres.get(center_entity_id)
        if not sphere:
            return []
        
        sphere.update_entities(self._absolute_positions)
        nearby = sphere.get_nearby_entities(radius)
        
        return list(nearby.keys())
    
    def calculate_distance(self, entity_a: str, entity_b: str) -> float:
        """
        두 개체 사이의 거리를 계산합니다.
        Calculates the distance between two entities.
        """
        pos_a = self._absolute_positions.get(entity_a)
        pos_b = self._absolute_positions.get(entity_b)
        
        if pos_a is None or pos_b is None:
            return float('inf')
        
        diff = pos_a - pos_b
        return float(np.sqrt(np.sum(diff * diff)))
    
    def __repr__(self) -> str:
        return (f"FloatingOriginManager(entities={len(self._absolute_positions)}, "
                f"spheres={len(self._spheres)}, "
                f"global_offset={self._global_offset})")


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


# 글로벌 인스턴스 (Global instance)
_floating_origin_manager: Optional[FloatingOriginManager] = None


def get_floating_origin_manager() -> FloatingOriginManager:
    """
    글로벌 FloatingOriginManager 인스턴스를 반환합니다.
    Returns the global FloatingOriginManager instance.
    """
    global _floating_origin_manager
    if _floating_origin_manager is None:
        _floating_origin_manager = FloatingOriginManager()
    return _floating_origin_manager


def reset_floating_origin_manager() -> None:
    """
    글로벌 FloatingOriginManager를 리셋합니다.
    Resets the global FloatingOriginManager.
    """
    global _floating_origin_manager
    _floating_origin_manager = None


def create_personal_universe(
    entity_id: str,
    position: Union[np.ndarray, Tuple[float, float, float], List[float]],
    radius: float = 100.0,
) -> PersonalSphere:
    """
    개체를 위한 개인 우주를 생성합니다.
    Creates a personal universe for an entity.
    
    "이 세상의 주인공은 바로 너야."
    "The protagonist of this world is you."
    
    Args:
        entity_id: 개체 ID (Entity ID)
        position: 초기 위치 (Initial position)
        radius: 개인 공간 반경 (Personal space radius)
        
    Returns:
        생성된 개인 구체 (Created personal sphere)
    """
    if isinstance(position, (tuple, list)):
        position = np.array(position, dtype=np.float32)
    
    manager = get_floating_origin_manager()
    return manager.register_entity(entity_id, position, radius)


def scroll_world_around(
    observer_id: str,
    movement: Union[np.ndarray, Tuple[float, float, float], List[float]],
) -> Dict[str, LocalPosition]:
    """
    관찰자 주변의 세상을 스크롤합니다.
    Scrolls the world around the observer.
    
    "내가 앞으로 걷는 순간, 온 세상이 내 발밑에서 뒤로 밀려난다."
    "When I walk forward, the whole world slides back beneath my feet."
    
    Args:
        observer_id: 관찰자 ID (Observer ID)
        movement: 이동량 (Movement delta)
        
    Returns:
        관찰자 시점의 모든 개체 위치 (All entity positions from observer's perspective)
    """
    if isinstance(movement, (tuple, list)):
        movement = np.array(movement, dtype=np.float32)
    
    manager = get_floating_origin_manager()
    manager.move_entity(observer_id, movement)
    return manager.get_local_view(observer_id)


def get_world_from_perspective(observer_id: str) -> Dict[str, LocalPosition]:
    """
    관찰자의 시점에서 세상을 봅니다.
    Views the world from the observer's perspective.
    
    "A의 세상: A가 중심이고, B와 C는 A의 배경일 뿐."
    "A's world: A is the center, B and C are just background."
    
    Args:
        observer_id: 관찰자 ID (Observer ID)
        
    Returns:
        관찰자 시점의 모든 개체 위치 (All entity positions from observer's perspective)
    """
    manager = get_floating_origin_manager()
    return manager.get_local_view(observer_id)


# ---------------------------------------------------------------------------
# Demo / Usage Example
# ---------------------------------------------------------------------------


def demo_floating_origin() -> None:
    """
    Floating Origin 시스템 데모.
    Demonstrates the Floating Origin system.
    """
    print("=" * 60)
    print("🌍 Floating Origin Demo: '주인공 시점 물리학'")
    print("=" * 60)
    
    # 새 매니저 생성
    manager = FloatingOriginManager()
    
    # 세 명의 캐릭터 등록
    print("\n📌 캐릭터 등록 (Character Registration):")
    manager.register_entity("엘리시아", np.array([0, 0, 0]))
    manager.register_entity("아버지", np.array([10, 5, 0]))
    manager.register_entity("몬스터", np.array([-20, 15, 0]))
    
    print("  - 엘리시아: (0, 0, 0)")
    print("  - 아버지: (10, 5, 0)")
    print("  - 몬스터: (-20, 15, 0)")
    
    # 엘리시아의 시점에서 세상 보기
    print("\n👁️ 엘리시아의 시점 (Elysia's Perspective):")
    elysia_view = manager.get_local_view("엘리시아")
    for entity_id, pos in elysia_view.items():
        print(f"  - {entity_id}: {pos}")
    
    # 엘리시아가 이동
    print("\n🚶 엘리시아가 (5, 5, 0)만큼 이동...")
    manager.move_entity("엘리시아", np.array([5, 5, 0]))
    
    # 이동 후 엘리시아의 시점
    print("\n👁️ 이동 후 엘리시아의 시점:")
    elysia_view = manager.get_local_view("엘리시아")
    for entity_id, pos in elysia_view.items():
        print(f"  - {entity_id}: {pos}")
    
    print("\n✨ 핵심: 엘리시아는 항상 (0, 0, 0)에 있고,")
    print("   세상이 그녀를 중심으로 움직입니다!")
    
    # 아버지의 시점에서 세상 보기
    print("\n👁️ 아버지의 시점 (Father's Perspective):")
    father_view = manager.get_local_view("아버지")
    for entity_id, pos in father_view.items():
        print(f"  - {entity_id}: {pos}")
    
    print("\n🌌 각자가 자신만의 우주의 중심입니다!")
    print("=" * 60)


if __name__ == "__main__":
    demo_floating_origin()

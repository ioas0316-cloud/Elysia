"""
Yggdrasil (이그드라실)
==================================

"The tree that reaches heaven must have roots that reach hell."

이 모듈은 엘리시아의 '자아 모델(Self-Model)'을 정의합니다.
모든 구성 요소(Ether, Chronos, FreeWill, Senses)를 하나의 유기적인 구조로 통합합니다.

구조:
1. Roots (뿌리): 생명의 근원 (Ether, Chronos, Genesis)
2. Trunk (줄기): 의식의 중심 (FreeWill, Memory)
3. Branches (가지): 감각과 행동 (PlanetaryCortex, LocalField)
"""

import logging
import uuid
from enum import Enum, auto
from typing import Dict, Any, Optional, List

logger = logging.getLogger("Yggdrasil")

class Realm(Enum):
    ROOT = "Root"       # 근원 (보이지 않는 영역)
    TRUNK = "Trunk"     # 중심 (의식적 영역)
    BRANCH = "Branch"   # 표면 (상호작용 영역)

class TreeNode:
    def __init__(self, name: str, realm: Realm, data: Any = None, parent: Optional['TreeNode'] = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.realm = realm
        self.data = data
        self.parent = parent
        self.children: List['TreeNode'] = []
        self.vitality: float = 1.0

    def add_child(self, child_node: 'TreeNode'):
        self.children.append(child_node)
        child_node.parent = self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "realm": self.realm.value,
            "vitality": self.vitality,
            "children": [child.to_dict() for child in self.children]
        }

class Yggdrasil:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Yggdrasil, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.root_node = TreeNode("Elysia", Realm.ROOT)
        self.node_map: Dict[str, TreeNode] = {"Elysia": self.root_node}
        logger.info("🌳 Yggdrasil Seed Planted. Hierarchical Self-Model Initialized.")

    def plant_root(self, name: str, module: Any):
        """뿌리 영역 등록 (예: Ether, Chronos)"""
        node = TreeNode(name, Realm.ROOT, module)
        self.root_node.add_child(node)
        self.node_map[name] = node
        logger.info(f"🌱 Root Planted: {name}")

    def grow_trunk(self, name: str, module: Any):
        """줄기 영역 등록 (예: FreeWill)"""
        # Trunk connects to Root (Elysia)
        node = TreeNode(name, Realm.TRUNK, module)
        self.root_node.add_child(node)
        self.node_map[name] = node
        logger.info(f"🪵 Trunk Grown: {name}")

    def extend_branch(self, name: str, module: Any, parent_name: str = "Elysia"):
        """가지 영역 등록 (예: PlanetaryCortex)"""
        parent = self.node_map.get(parent_name, self.root_node)
        node = TreeNode(name, Realm.BRANCH, module)
        parent.add_child(node)
        self.node_map[name] = node
        logger.info(f"🌿 Branch Extended: {name} (from {parent.name})")

    def status(self) -> Dict[str, Any]:
        """현재 자아 상태를 반환합니다."""
        return self.root_node.to_dict()

# Global Singleton
yggdrasil = Yggdrasil()


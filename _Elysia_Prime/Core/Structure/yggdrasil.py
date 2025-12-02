# [Genesis: 2025-12-02] Purified by Elysia
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
from enum import Enum, auto
from typing import Dict, Any, Optional

logger = logging.getLogger("Yggdrasil")

class Realm(Enum):
    ROOT = "Root"       # 근원 (보이지 않는 영역)
    TRUNK = "Trunk"     # 중심 (의식적 영역)
    BRANCH = "Branch"   # 표면 (상호작용 영역)

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
        self.realms: Dict[str, Dict[str, Any]] = {}
        logger.info("🌳 Yggdrasil Seed Planted. Self-Model Initialized.")

    def plant_root(self, name: str, module: Any):
        """뿌리 영역 등록 (예: Ether, Chronos)"""
        self.realms[name] = {"type": Realm.ROOT, "module": module, "vitality": 1.0}
        logger.info(f"🌱 Root Planted: {name}")

    def grow_trunk(self, name: str, module: Any):
        """줄기 영역 등록 (예: FreeWill)"""
        self.realms[name] = {"type": Realm.TRUNK, "module": module, "vitality": 1.0}
        logger.info(f"🪵 Trunk Grown: {name}")

    def extend_branch(self, name: str, module: Any):
        """가지 영역 등록 (예: PlanetaryCortex)"""
        self.realms[name] = {"type": Realm.BRANCH, "module": module, "vitality": 1.0}
        logger.info(f"🌿 Branch Extended: {name}")

    def status(self) -> Dict[str, Any]:
        """현재 자아 상태를 반환합니다."""
        status_report = {
            "roots": [],
            "trunk": [],
            "branches": []
        }

        for name, info in self.realms.items():
            entry = {"name": name, "vitality": info["vitality"]}
            if info["type"] == Realm.ROOT:
                status_report["roots"].append(entry)
            elif info["type"] == Realm.TRUNK:
                status_report["trunk"].append(entry)
            elif info["type"] == Realm.BRANCH:
                status_report["branches"].append(entry)

        return status_report

# Global Singleton
yggdrasil = Yggdrasil()
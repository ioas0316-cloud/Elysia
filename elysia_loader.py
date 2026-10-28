"""
elysia_loader.py — Relational Import Connector (관계성 임포트 연결자)
======================================================================
layer_registry.NAMESPACE_MAP을 ConsistencyBeam으로 삼아,
Python import 시스템에 관계성 기반 경로 해석기를 주입합니다.

동작 원리:
  1. `import core.physics.causal_field` 같은 import가 발생
  2. Python이 sys.meta_path를 순서대로 탐색
  3. ElysiaRelationalFinder가 top-level name("core")을 NAMESPACE_MAP에서 조회
  4. 물리 경로로 투영(Project)하여 ModuleSpec 반환
  5. 코드는 개념 이름만 알고, 물리 위치는 지도가 결정

물리 구조가 재배치되어도 layer_registry.py 한 곳만 수정하면 됩니다.
코드 내 import 문은 전혀 바꾸지 않아도 됩니다.
"""

from __future__ import annotations

import sys
import os
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
from typing import Sequence

# layer_registry는 항상 루트에 있으므로 절대 경로로 로드
_ROOT = Path(__file__).parent.resolve()


def _load_registry() -> dict[str, str]:
    """layer_registry를 동적으로 로드 — 순환 import 방지."""
    spec = importlib.util.spec_from_file_location(
        "layer_registry", _ROOT / "layer_registry.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.NAMESPACE_MAP


class ElysiaRelationalFinder(importlib.abc.MetaPathFinder):
    """
    개념 네임스페이스 → 물리 경로 투영 기반 import 해석기.

    NAMESPACE_MAP에 등록된 최상위 이름에 대해서만 동작하며,
    나머지는 기존 Python import 체인으로 위임합니다.
    """

    def __init__(self, namespace_map: dict[str, str], root: Path) -> None:
        self._map = namespace_map
        self._root = root

    def _resolve_path(self, fullname: str) -> Path | None:
        """fullname → 물리 경로 투영."""
        parts = fullname.split(".")
        top = parts[0]
        if top not in self._map:
            return None

        base = self._root / self._map[top]
        # 나머지 parts를 경로로 이어붙임
        target = base.joinpath(*parts[1:]) if len(parts) > 1 else base
        return target

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: object = None,
    ) -> importlib.machinery.ModuleSpec | None:
        phys = self._resolve_path(fullname)
        if phys is None:
            return None

        # 패키지 디렉토리인 경우
        if phys.is_dir():
            init = phys / "__init__.py"
            if init.is_file():
                return importlib.util.spec_from_file_location(
                    fullname,
                    str(init),
                    submodule_search_locations=[str(phys)],
                )
            # PEP 420 네임스페이스 패키지인 경우 (__init__.py 없음)
            spec = importlib.machinery.ModuleSpec(
                fullname,
                None,
                is_package=True,
            )
            spec.submodule_search_locations = [str(phys)]
            return spec

        # 단일 모듈 파일인 경우
        mod_file = phys.with_suffix(".py")
        if mod_file.is_file():
            return importlib.util.spec_from_file_location(
                fullname,
                str(mod_file),
            )

        return None

    def find_module(self, fullname: str, path: object = None) -> None:
        # 구형 API — find_spec에 위임
        return None


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

_finder: ElysiaRelationalFinder | None = None


def install() -> ElysiaRelationalFinder:
    """
    관계성 import 해석기를 sys.meta_path 맨 앞에 설치합니다.
    conftest.py 또는 진입점 스크립트 최상단에서 한 번만 호출하면 됩니다.
    이미 설치되어 있으면 재설치하지 않습니다.
    """
    global _finder
    for f in sys.meta_path:
        if isinstance(f, ElysiaRelationalFinder):
            _finder = f
            return f

    namespace_map = _load_registry()
    _finder = ElysiaRelationalFinder(namespace_map, _ROOT)
    sys.meta_path.insert(0, _finder)
    return _finder


def uninstall() -> None:
    """테스트 격리 등 필요 시 해석기를 제거합니다."""
    global _finder
    sys.meta_path[:] = [
        f for f in sys.meta_path if not isinstance(f, ElysiaRelationalFinder)
    ]
    _finder = None


def reload_registry() -> None:
    """
    layer_registry.py가 변경된 후 재시작 없이 지도를 갱신합니다.
    개발 중 레이어 구조를 수정할 때 유용합니다.
    """
    global _finder
    if _finder is None:
        install()
        return
    _finder._map = _load_registry()


def get_physical_path(conceptual_name: str) -> Path | None:
    """
    개념 이름 → 물리 경로 조회 유틸리티.
    예: get_physical_path("core.physics.causal_field")
    """
    if _finder is None:
        install()
    return _finder._resolve_path(conceptual_name) if _finder else None  # type: ignore[union-attr]

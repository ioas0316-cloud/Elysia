"""
conftest.py — Elysia 관계성 import 부트스트랩
==============================================
테스트 세션 시작 시 ElysiaRelationalFinder를 sys.meta_path에 설치합니다.
이후 모든 테스트에서 개념 이름(core, elysia_core 등)으로 import 가능합니다.
물리 레이어 경로(L0~L3)는 layer_registry.py 한 곳에서만 관리됩니다.
"""
import sys
from pathlib import Path
import elysia_loader

elysia_loader.install()

# demo 기반 검증 테스트를 위한 demos 서브디렉토리 경로 등록
_ROOT = Path(__file__).parent.resolve()
for _sub in ["demos", "demos/layer_0", "demos/layer_1", "demos/layer_2", "demos/layer_3", "demos/orchestration"]:
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.append(_p)

_archive_demos = Path("C:/Archive/02_Evolution/Phase_Showcases_and_Demos_Legacy/demos")
if _archive_demos.is_dir() and str(_archive_demos) not in sys.path:
    sys.path.append(str(_archive_demos))

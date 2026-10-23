"""
conftest.py — Scale Hierarchy sys.path 등록
L0~L2 레이어 폴더를 sys.path에 추가하여 기존 import 경로 하위 호환 보장.
"""
import sys
import os

_ROOT = os.path.dirname(__file__)

# Scale Hierarchy layers — 레이어 내부 모듈들을 루트처럼 접근 가능하게
_LAYERS = [
    "L0_substrate",
    "L1_causal_field",
    "L2_cognitive_engine",
    "L3_interface",
]

for _layer in _LAYERS:
    _path = os.path.join(_ROOT, _layer)
    if _path not in sys.path:
        sys.path.insert(0, _path)

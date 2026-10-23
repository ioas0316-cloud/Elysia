"""
conftest.py — Elysia 관계성 import 부트스트랩
==============================================
테스트 세션 시작 시 ElysiaRelationalFinder를 sys.meta_path에 설치합니다.
이후 모든 테스트에서 개념 이름(core, elysia_core 등)으로 import 가능합니다.
물리 레이어 경로(L0~L3)는 layer_registry.py 한 곳에서만 관리됩니다.
"""
import elysia_loader

elysia_loader.install()

"""
layer_registry.py — Elysia Layer Projection Map (투영 관계성 지도)
=================================================================
이 파일은 "개념 네임스페이스(설계도)" ↔ "물리 경로(실물)" 간의
관계성 토폴로지를 선언합니다.

코드 어디서도 실제 레이어 폴더 경로를 알 필요가 없습니다.
import 는 항상 개념 이름으로 이루어지며, 이 지도가 그것을
물리 위치로 투영(Project)합니다.

  [설계도 / 개념 공간]          [실물 / 물리 공간]
  ─────────────────────    →   ──────────────────────────
  core                         L1_causal_field/core
  synaptic_architecture        L1_causal_field/synaptic_architecture
  elysia_core                  L2_cognitive_engine/elysia_core
  ...

4 Continuities 관점:
  - 관계성: 경로가 아닌 관계 이름으로 결합
  - 연결성: 레이어 경계를 가로지르는 ConnectivityBeam 역할
  - 운동성: import 흐름의 방향성이 이 지도를 따라 흐름
  - 정보적 연속성: 물리 구조가 바뀌어도 개념 이름은 연속
"""

# ─────────────────────────────────────────────────────────────────────
# Namespace → Physical Layer Map
# 구조가 바뀌면 여기만 수정하면 됩니다. 코드는 건드리지 않습니다.
# ─────────────────────────────────────────────────────────────────────

NAMESPACE_MAP: dict[str, str] = {

    # ── L1: Causal Field (인과장 · 위상 · 연결성 원리) ──────────────
    "core":                   "L1_causal_field/core",
    "synaptic_architecture":  "L1_causal_field/synaptic_architecture",
    "modules":                "L1_causal_field/modules",
    "benchmarks":             "L1_causal_field/benchmarks",

    # ── L2: Cognitive Engine (인지 · 의식 · 자아 런타임) ────────────
    "elysia_core":            "L2_cognitive_engine/elysia_core",
    "elysia_engine":          "L2_cognitive_engine/elysia_engine",

    # ── L3: Interface (에뮬레이션 · 시뮬레이터 · 외부 접속) ─────────
    "simulators":             "L3_interface/simulators",
    "emulation":              "L3_interface/emulation",

    # ── L0: Substrate (하드웨어 · 네이티브 빌드 산출물) ─────────────
    # Python 바인딩이 빌드된 후 sys.modules에 직접 올라오므로
    # 여기서는 경로 참조 불필요. 빌드 경로는 setup.py가 관리.
}

# ─────────────────────────────────────────────────────────────────────
# Layer 메타데이터 (문서화 + 크로매틱 성향 참조용)
# 실행에 영향 없음 — 아키텍처 의사결정 렌즈로 활용
# ─────────────────────────────────────────────────────────────────────

LAYER_META: dict[str, dict] = {
    "L0_substrate": {
        "description": "물리 · 하드웨어 · RTL · 네이티브 커널",
        "chromatic":   "Blue (Order) — 고정 구조, 불변 제약",
        "namespaces":  [],  # 빌드 산출물, Python 직접 참조 없음
    },
    "L1_causal_field": {
        "description": "인과장 · 위상 · 스케일 경계 · 연결성 원리",
        "chromatic":   "Red (Flux) — 흐름 생성, 동적 인과",
        "namespaces":  ["core", "synaptic_architecture", "modules", "benchmarks"],
    },
    "L2_cognitive_engine": {
        "description": "의식 · 감각 · 자아 · 목적 커널 · 생태계 런타임",
        "chromatic":   "Red+Blue (Emergence) — 구조화된 흐름",
        "namespaces":  ["elysia_core", "elysia_engine"],
    },
    "L3_interface": {
        "description": "에뮬레이션 · 게임엔진 · 외부 접속 · 데이터",
        "chromatic":   "Yellow (Entropy) — 발산, 외부 연결",
        "namespaces":  ["simulators", "emulation"],
    },
}

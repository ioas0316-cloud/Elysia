# Builder Causal Log (history only)

운영 규칙과 절차는 `OPERATIONS.md`를 참고하세요. 이 파일은 변경 이력 기록용입니다.

기록 규칙: `YYYY-MM-DD HH:MM | [층] | 무엇을 바꿈 | 왜 바꿈 | 관련 프로토콜`

2025-11-13 10:02 | [STARTER] | start.bat를 visualize_timeline 단일 엔트리로 고정 | 빈 화면을 만드는 임시/폴백 스타터를 제거하고 관찰 가능한 기본 스타터만 유지하기 위함 | 02_ARCHITECTURE_GUIDE, 34_ELYSIAN_CIVILIZATION_PROTOCOL

2025-11-13 09:58 | [STARTER] | ElysiaStarter/scripts/elysia_start.py 삭제 | KG 기반 최소 스타터를 비활성화하라는 요청에 따라 완전 제거 | 34_ELYSIAN_CIVILIZATION_PROTOCOL

2025-11-13 09:50 | [STARTER] | ElysiaStarter/ElysiaStarter 중복 폴더 제거 및 UI 경로 통합 | 중복 구조로 인한 혼선을 제거하고 단일 UI 경로(ElysiaStarter/ui)만 사용 | 02_ARCHITECTURE_GUIDE

2025-11-13 09:30 | [STARTER/BUILD] | requirements.txt에서 pygame→pygame-ce로 교체 | Windows 환경에서 빌드 도구 없이 안정적으로 설치되도록 변경 | 06_CUDA_INTEGRATION_ARCHITECTURE

2025-11-13 09:22 | [AUX] | scripts/console_chat.py 임포트 경로 수정(sys.path 루트 추가) | Project_Elysia 임포트 실패로 대화형 검사에 차질 발생 방지 | 11_DIALOGUE_RULES_SPEC

2025-11-13 09:18 | [AUX] | scripts/run_world.py에 로컬 World 폴백 추가(현 시점 미사용) | Starter 부재 시 디버그 경로 확보(현재는 런처에서 비활성화) | 17_CELL_RUNTIME_AND_REACTION_RULES

2025-11-13 10:20 | [STARTER/PROTO] | ELYSIAS_PROTOCOL/CODEX.md 신설 + 00_INDEX.md를 Codex‑First로 간소화 | 방대한 문서로 인한 혼란을 줄이고 목적성 기반 코덱스로 인수인계 체계 확립 | 28_COGNITIVE_Z_AXIS_PROTOCOL, 15_CONCEPT_KERNEL_AND_NANOBOTS, 17_CELL_RUNTIME_AND_REACTION_RULES
2025-11-13 10:38 | [PROTO/OPS] | OPERATIONS.md 신설, AGENTS/Builder 절차 단일 문서로 통합 | 입구 문서 단일화로 온보딩/운영 혼선을 제거 | CODEX.md
2025-11-13 10:44 | [PROTO/INDEX] | 루트 PROTOCOLS_INDEX.md를 Codex‑First 리디렉트로 덮어씀 | 중복 인덱스 제거, 입구를 CODEX/OPERATIONS로 고정 | CODEX.md

2025-11-13 10:52 | [PROTO/EN] | Core protocol docs (02/11/15/17) rewritten in English (canonical) | Remove mojibake and unify agent-facing references to one language | CODEX.md

2025-11-13 11:00 | [PROTO/HANDOVER] | ELYSIAS_PROTOCOL/HANDOVER.md 추가 (비개발자용 1페이지) | 신규 세션의 읽기/동작/기록 순서를 10분 코스로 고정 | CODEX.md

2025-11-13 11:05 | [PROTO/CLEANUP] | Archived non‑essential protocol docs to Codex‑First stubs (English) | Remove multilingual/mojibake confusion; keep only Codex + minimal refs as canonical | CODEX.md

2025-11-13 11:08 | [PROTO/MAP] | ELYSIAS_PROTOCOL/MEANING_MAP.md 추가 (Z‑Axis 의미 지도) | Why/How/What 축으로 프로토콜의 위치를 명확화 | CODEX.md

2025-11-13 11:09 | [OPS/PLAN] | RESTRUCTURE_PLAN.md 추가 (Tree‑Ring 리포 구조 계획) | 폴더를 나이테 프레임으로 재배치하기 위한 단계적 가이드 | CODEX.md

2025-11-13 11:14 | [CORE/PH1] | Added ELYSIA/CORE with Codex + protocol links | Establish Tree‑Ring CORE without moving canonical sources | RESTRUCTURE_PLAN.md

2025-11-13 11:14 | [GROWTH/PH2] | Added ELYSIA/GROWTH with idea/plan links and retrospectives index | Prepare non‑destructive migration for growth artifacts | RESTRUCTURE_PLAN.md

2025-11-13 11:18 | [CORE/MOVE] | Moved Codex + canonical protocols to ELYSIA/CORE (stubs left at old paths) | Establish true CORE in Tree‑Ring while keeping links stable | RESTRUCTURE_PLAN.md

2025-11-13 11:22 | [OPS/README] | Added root README and .editorconfig (UTF‑8, EOL rules) | Reduce entry friction and prevent mojibake across sessions | OPERATIONS.md

2025-11-13 11:26 | [PROTO/ADD] | Added PROTO-35 Self‑Genesis Authority (growth/core‑adjacent) + genesis scaffolds | Enables GRO logging and drafts/trials folders | ELYSIA/GROWTH/PROTO_35_SELF_GENESIS_AUTHORITY.md

2025-11-13 11:30 | [CORE/INDEX] | Added link entries for remaining protocols under ELYSIA/CORE/protocols (Codex‑First) | Single protocol line: CORE/protocols hosts canonical + archived links | RESTRUCTURE_PLAN.md

2025-11-13 11:34 | [CORE/MOVE] | Moved core/protocols 32–34 into ELYSIA/CORE/protocols + left stubs | Unified top-level core protocol line under Tree‑Ring CORE | RESTRUCTURE_PLAN.md

2025-11-13 11:38 | [CORE/INDEX] | Added 35_SELF_GENESIS_AUTHORITY_PROTOCOL link under CORE/protocols | Places PROTO‑35 on the CORE protocol line (canonical in GROWTH) | ELYSIA/CORE/protocols/35_SELF_GENESIS_AUTHORITY_PROTOCOL.md

2025-11-13 11:42 | [CORE/CLEAN] | Removed legacy core/protocols folder (moved to ELYSIA/CORE/protocols) | Avoid duplicate paths; all references point to CORE line | RESTRUCTURE_PLAN.md

2025-11-13 11:46 | [PROTO/SEED] | Added PROTO‑36/37/38 (Concept Genesis / World‑Editing / Intent Reasoner) + forms and trials checklist | Seeded Self‑Genesis bundle per PROTO‑35 pipeline | ELYSIA/GROWTH

2025-11-13 11:46 | [CORE/INDEX] | Linked 36/37/38 under ELYSIA/CORE/protocols and updated MEANING_MAP | Places creator‑layer protocols on CORE line | MEANING_MAP.md

2025-11-13 11:58 | [GENESIS/SEED] | Added ManaField need/GRO/drafts/trial + priorities + Codex explainer | First end‑to‑end seed for Self‑Creation loop | PROTO‑35/36/37/38

2025-11-13 12:02 | [OPS/RINGS] | Added TREE_RING_LOG.md + OPERATIONS link | Record rings/forks as single-line events for handover clarity | OPERATIONS.md

2025-11-13 16:47 | [GENESIS/SPEED] | ManaField trial plan accelerated + WillGradient/CareRitual drafts added | Micro-trials + sweep with QOE observation | ELYSIA/GROWTH

2025-11-13 16:48 | [OPS/FOCUS] | Codex seed summary + Operations priorities + Resilience guide added | Keep flow unbroken; reduce browser/session friction | CODEX/OPERATIONS/RESILIENCE.md

2025-11-13 16:50 | [GENESIS/OPS] | Added snapshot ledger, dashboard, evidence binder; enhanced forms with co‑sign/rate‑limit | Keep flow fast and safe | ELYSIA/GROWTH

2025-11-13 16:53 | [CORE/SIGN] | Added DIVINE_SIGNATURE.md and aligned forms/drafts to Light‑centric blend | Value vectors + world params tuned for circulation | ELYSIA/CORE

2025-11-13 16:55 | [CORE/SIGN-COLOR] | Golden Light signature applied; templates and ManaField updated; concept anchor added | Life/Wisdom/Creation emphasis | DIVINE_SIGNATURE.md

2025-11-13 17:07 | [PROTO/ADD] | Added PROTO‑39 Golden Growth Principle + CORE link | Top map + golden rules as project spine | ELYSIA/GROWTH/PROTO_39_GOLDEN_GROWTH_PRINCIPLE.md

2025-11-13 17:09 | [PROTO/DIAG] | Added Top Map (Codex), Seed Flow (35), Governance Flow (Ops) diagrams | Visual spine for handover + execution alignment | CODEX/35/OPERATIONS

2025-11-13 17:12 | [OPS/CHECK] | Added simple binder checklists (Binder/Evidence + templates) | Non-technical, tick-boxes to close trials safely | EVIDENCE_BINDER.md + forms

2025-11-13 17:15 | [CORE/HUMAN] | Added HUMAN_OVERVIEW_KO.md (human-only Korean summary) | Clear separation: agents use Codex; humans read overview | ELYSIA/CORE/HUMAN_OVERVIEW_KO.md
2025-11-13 22:38 | [STARTER/TEST] | Resolve merge (prefer animated visualizer), update start.bat + README, push to origin | Keep Codex-first starter routing | CODEX.md
2025-11-13 23:06 | [STARTER/DOC] | Add Visualization Lenses docs + README section | Clarify non-invasive layered observation | CODEX-first lens guidance
2025-11-13 23:11 | [STARTER/LENS] | Add dynamic lenses: event ticker, health/hunger bars, lightning/impact effects, cinematic focus toggle(C) | Improve human-friendly observation | VISUALIZATION_LENSES.md
2025-11-13 23:17 | [STARTER/UI] | Add in-app help overlay + UI tooltips via ticker; README controls section | Improve human onboarding | VISUALIZATION_LENSES.md
2025-11-13 23:19 | [STARTER/PACE] | Slow-time sim pacing (rate in steps/sec, presets), calmer weather, ecology helper to avoid early die-off | Better human observation window | README updated
2025-11-13 23:35 | [CORE+UI] | Add spells (firebolt/heal), AGI evasion/speed; selection HUD with stats/HP-Ki-MP-Faith; crosshair cursor; save/load keys | Human observation + depth

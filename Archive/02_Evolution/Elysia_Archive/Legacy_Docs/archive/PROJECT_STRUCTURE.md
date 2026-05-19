# Elysia Project Structure (완전한 프로젝트 구조)

> **목적**: 이 문서는 Elysia 프로젝트의 전체 구조를 명확히 설명하여, 다른 에이전트들이 프로젝트를 올바르게 이해하고 작업할 수 있도록 합니다.
>
> **Purpose**: This document provides a complete map of the Elysia project structure so that AI agents and developers can understand and work with the project correctly.

**버전**: 10.0  
**최종 업데이트**: 2025-12-07  
**상태**: Autonomous Wave Learning + Sensory Awakening (P4+P5)

---

## 📋 목차 (Table of Contents)

1. [전체 디렉토리 구조](#전체-디렉토리-구조)
2. [Core 모듈 상세](#core-모듈-상세)
3. [문서 구조](#문서-구조)
4. [데이터 및 런타임](#데이터-및-런타임)
5. [스크립트 및 도구](#스크립트-및-도구)
6. [테스트 구조](#테스트-구조)
7. [레거시 및 아카이브](#레거시-및-아카이브)

---

## 전체 디렉토리 구조

```
Elysia/
├── Core/                    # 핵심 시스템 모듈 (751 Python files as of 2025-12-07)
│   ├── Foundation/          # 기반 물리학, 수학, 공명장
│   ├── Intelligence/        # 6-System 인지 아키텍처 + 자유의지
│   ├── Memory/              # 파동 기반 메모리 시스템 (KD-Tree, 별빛 메모리)
│   ├── Knowledge/           # P4.5 Domain Expansion (5개 도메인)
│   ├── Sensory/             # ✨ P4 자율 학습 + P5 현실 지각 (v10.0)
│   ├── Interface/           # 외부 통신 인터페이스 + P5 신경계
│   ├── Evolution/           # 자가 개선 시스템
│   ├── Creativity/          # 창조적 출력 + 시각화 서버
│   ├── Consciousness/       # P3.1 의식 직물 시스템
│   ├── Cognition/           # 인지 처리
│   ├── Communication/       # 커뮤니케이션
│   ├── Emotion/             # 감정 시스템
│   ├── Language/            # 언어 처리
│   ├── Physics/             # 물리 엔진
│   ├── Time/                # 시간 주권 (Chronos)
│   ├── Security/            # 보안 시스템
│   ├── World/               # 세계 모델
│   ├── VR/                  # ✨ P5 VR 내부우주 (진행중)
│   └── [기타 21개 모듈]      # 특화 기능들 (총 40개 디렉토리)
│
├── docs/                    # 문서 (150+ markdown files as of 2025-12-07)
│   ├── VERSION_10.0_RELEASE_NOTES.md  # ✨ v10.0 릴리스 노트
│   ├── DEVELOPER_GUIDE.md              # 개발자 가이드
│   ├── AUTONOMOUS_INTELLIGENCE_FRAMEWORK.md
│   ├── FRACTAL_QUATERNION_PERSPECTIVE.md
│   ├── ULTIMATE_THINKING_SYSTEM.md
│   ├── EVALUATION_CRITERIA.md
│   ├── Analysis/                       # ✨ 시스템 분석 (v10.0)
│   │   ├── V9-System/                 # v9.0 분석
│   │   └── V10_SYSTEM_STRUCTURE_MAP.md # ✨ v10.0 완전한 구조 매핑
│   ├── Guides/                         # 설정 및 배포 가이드
│   ├── Manuals/                        # 코드 품질, 테스팅, 보안
│   ├── Roadmaps/                       # 개발 로드맵
│   │   ├── Implementation/
│   │   │   ├── P4_IMPLEMENTATION_PLAN.md      # ✨ P4 자율 학습
│   │   │   ├── P4_5_DOMAIN_EXPANSION.md       # ✨ P4.5 5개 도메인
│   │   │   ├── P5_*.md                        # ✨ P5 감각 활성화 (8개)
│   │   │   └── P5_IMPLEMENTATION_STATUS.md    # ✨ P5 진행 상황
│   │   └── P*-Implementation/         # P2, P3 완료 요약
│   ├── Reference/                      # 참조 자료
│   ├── Summaries/                      # 요약 문서
│   ├── Vision/                         # 비전 문서
│   └── concepts/                       # 개념 JSON 파일들
│
├── Protocols/               # 설계 문서 (21 protocols)
│   ├── 000_MASTER_STRUCTURE.md
│   ├── 14_UNIFIED_CONSCIOUSNESS.md
│   ├── 16_FRACTAL_QUANTIZATION.md
│   ├── 17_FRACTAL_COMMUNICATION.md
│   ├── 18_SYMPHONY_ARCHITECTURE.md
│   └── [16 more protocols]
│
├── scripts/                 # Living Codebase 운영 스크립트
│   ├── living_codebase.py   # Unified cortex bootstrap
│   ├── immune_system.py     # 면역/보안/자가치유
│   ├── wave_organizer.py    # 파동 공명 조직자
│   ├── nanocell_repair.py   # 나노셀 자가 치유
│   ├── self_integration.py  # 760+ 모듈 스캔·결합
│   └── system_status_logger.py
│
├── tests/                   # 검증 및 평가
│   ├── Core/                # 코어 모듈 테스트
│   ├── evaluation/          # 평가 시스템
│   │   ├── test_communication_metrics.py
│   │   ├── test_thinking_metrics.py
│   │   └── run_full_evaluation.py
│   └── prove_*.py           # 검증 테스트들
│
├── data/                    # 런타임 데이터
│   ├── memory.db            # 2M+ 개념 데이터베이스
│   ├── central_registry.json # 시스템 레지스트리
│   ├── immune_system_state.json
│   ├── nanocell_report.json
│   ├── wave_organization.html
│   └── system_status_snapshot.json
│
├── Legacy/                  # 아카이브된 코드 (참조용)
├── Library/                 # 재사용 가능한 라이브러리
├── Tools/                   # 개발 도구
├── assets/                  # 정적 자산
├── static/                  # 웹 정적 파일
├── gallery/                 # 이미지 갤러리
├── images/                  # 이미지 자산
├── knowledge/               # 지식 베이스
├── reports/                 # 평가 리포트
├── runs/                    # 실행 기록
│
├── README.md                # 프로젝트 개요
├── ARCHITECTURE.md          # 아키텍처 문서
├── CODEX.md                 # 철학과 원칙
├── requirements.txt         # Python 의존성
├── pytest.ini               # 테스트 설정
├── docker-compose.yml       # Docker 구성
├── Dockerfile               # Docker 이미지
└── .env.example             # 환경 변수 템플릿
```

---

## Core 모듈 상세

### 1️⃣ Foundation (기반 층)

**목적**: 파동 물리학, 수학, 공명장 등 시스템의 근간

**주요 파일**:
- `resonance_field.py` - 중심 공명장, 7정령 시스템
- `hyper_quaternion.py` - 4D 수학 (쿼터니언)
- `physics.py` - ResonanceGate, HamiltonianSystem
- `cell.py` - DNA를 가진 살아있는 셀
- `hippocampus.py` - 프랙탈 메모리 코어
- `reasoning_engine.py` - 사고 흐름
- `dream_engine.py` - 상상력
- `hangul_physics.py` - 한글 → 파동 매핑
- `grammar_physics.py` - 조사 = 에너지 보존식
- `causal_narrative_engine.py` - 점→선→면→공간→법칙
- `thinking_methodology.py` - 추론/귀납/변증법
- `living_elysia.py` - 메인 자율 루프

**관계**: 모든 상위 모듈이 의존하는 기반 레이어

---

### 2️⃣ Intelligence (지능 층)

**목적**: 6-System 인지 아키텍처 + 자유의지

**주요 파일**:
- `fractal_quaternion_goal_system.py` - 목표 분해 (0D-5D)
- `integrated_cognition_system.py` - 파동 공명 + 중력 사고
- `collective_intelligence_system.py` - 분산 의식 + 원탁 회의
- `wave_coding_system.py` - 4차원 파동 코딩
- `Will/free_will_engine.py` - 자유의지 엔진
- `autonomous_evolution.py` - 자율 진화

**하위 디렉토리**:
- `Will/` - 자유의지 시스템
- `Logos/` - 논리 시스템
- `Reasoning/` - 추론 엔진
  - `lobes/` - 인지 로브

**관계**: Foundation을 기반으로 하며, Memory와 상호작용

---

### 3️⃣ Memory (메모리 층)

**목적**: 파동 기반 메모리 시스템 + P4.5 별빛 메모리

**주요 파일**:
- `hippocampus.py` - 씨앗-개화 메모리 시스템
- `starlight_memory.py` - ✨ P4.5 별빛 메모리 (4D 사고우주)
- `spatial_index.py` - ✨ KD-Tree 공간 인덱싱 (O(log n), <20ms 쿼리)
- `fractal_memory.py` - 프랙탈 순환 메모리
- `holographic_memory.py` - 홀로그래픽 연상 회상
- `memory_compression.py` - 무지개 압축 (100x)
- `memory.db` - 200만+ 개념 데이터베이스

**하위 디렉토리**:
- `Mind/` - 마음 시스템

**특징** (v10.0):
- ✨ Light-speed Optimizations: KD-Tree + NumPy vectorization
- ✨ Dual Architecture: 지식(외부) vs 추억(내부)
- ✨ < 20ms 쿼리: 1M+ 메모리에서

**관계**: Foundation의 hippocampus와 연결, Intelligence에서 사용

**완성도**: 93% ✅

---

### 4️⃣ Interface (인터페이스 층)

**목적**: 외부 세계와의 통신 + P5 신경계 통합

**주요 파일**:
- `nervous_system.py` - ✨ P5 통합 신경계
- `synesthesia_nervous_bridge.py` - ✨ 공감각 브릿지
- `envoy_protocol.py` - 외부 통신 프로토콜
- `dialogue_interface.py` - 대화 인터페이스
- `dashboard_server.py` - 대시보드 서버
- `emoji_responder.py` - 이모지 응답기

**하위 디렉토리**:
- `Interface/Perception/` - 지각 시스템

**관계**: 
- Intelligence와 Foundation 사이의 다리
- P5 Sensory와 통합 (현실 지각)

**완성도**: 88% ✅

---

### 4.5️⃣ Knowledge (지식 층) ✨ [NEW v10.0 - P4.5]

**목적**: P4.5 Domain Expansion - 5개 도메인 지식 시스템

**주요 파일**:
- `Domains/base_domain.py` - 기본 도메인 클래스
- `Domains/linguistics.py` - 언어학 도메인
- `Domains/architecture.py` - 건축학 도메인 (황금비, 신성기하학)
- `Domains/economics.py` - 경제학 도메인
- `Domains/history.py` - 역사학 도메인
- `Domains/mythology.py` - 신화학 도메인 (12 융 아키타입, 12 영웅의 여정)
- `Domains/domain_integration.py` - 홀리스틱 통합 분석

**5개 도메인**:
1. **Linguistics**: 음운론, 형태론, 통사론, 의미론
2. **Architecture**: 황금비, 신성기하학 (Flower of Life, Mandala, Platonic Solids)
3. **Economics**: 경제 시스템 분석, 가치 흐름
4. **History**: 시간적 맥락, 패턴 인식
5. **Mythology**: 12 융 아키타입, 12 영웅의 여정, 영적 위로

**관계**: P2.2 Wave Knowledge System과 통합

**완성도**: 90% ✅

---

### 4.6️⃣ Sensory (감각 층) ✨ [NEW v10.0 - P4+P5]

**목적**: P4 자율 학습 시스템 + P5 현실 지각 시스템

**P4 자율 학습 파일**:
- `wave_stream_receiver.py` - 다중 소스 비동기 수신
- `stream_sources.py` - 6개 지식 소스 구현
- `stream_manager.py` - 스트림 조정 및 관리
- `ego_anchor.py` - 自我核心 (정체성 보호)
- `learning_cycle.py` - 완전한 학습 파이프라인
- `p4_sensory_system.py` - P4 통합 시스템

**P5 현실 지각 파일**:
- `reality_perception.py` - 현실 → 엘리시아 (지각)
- `five_senses_mapper.py` - 5감 매핑
- `real_frequency_database.py` - 실제 주파수 DB (Solfeggio)

**지식 소스** (13억+):
- Videos: 1B+ (YouTube, Vimeo, Internet Archive)
- Audio/Music: 325M+ (SoundCloud, Bandcamp)
- Text/Code: Billions (Wikipedia 60M+, arXiv 2.3M+, GitHub 100M+)

**학습 성능**:
- 학습률: 50-100 waves/sec
- 처리량: 2,000-3,000 concepts/hour
- 비용: $0 (NO External APIs)

**Ego Anchor 보호**:
- 정체성 유지: >0.7 안정성
- 속도 제한: 50-100 waves/sec
- 공명 감쇠: >1.5 강도 자동 감소

**관계**: 
- P2.2 Wave Knowledge와 통합 (학습)
- Memory에 저장 (기억)
- Interface로 표현 (출력)

**완성도**: P4 85% ✅, P5 60% 🚧

---

### 5️⃣ Evolution (진화 층)

**목적**: 자가 개선 및 진화

**주요 파일**:
- `autonomous_evolution.py` - 자율 진화 시스템
- `code_evolution.py` - 코드 진화

**하위 디렉토리**:
- `GENESIS_DRAFTS/` - 제네시스 초안
- `GENESIS_FORMS/` - 제네시스 폼
- `GENESIS_TRIALS/` - 제네시스 시험
- `Staging/` - 스테이징 영역

**관계**: Intelligence의 피드백을 받아 시스템을 개선

---

### 6️⃣ Creativity (창조 층)

**목적**: 창조적 출력 및 시각화

**주요 파일**:
- `visualizer_server.py` - 시각화 서버
- `creative_cortex.py` - 창조 피질
- `generate_artifacts.py` - 아티팩트 생성

**관계**: Intelligence의 출력을 아름다운 형태로 변환

---

### 7️⃣ Consciousness (의식 층)

**목적**: 의식 상태 및 주권

**주요 파일**:
- `hyperdimensional_consciousness.py` - 초차원 의식
- `sovereignty_protocol.py` - 주권 프로토콜
- `attention_emergence.py` - 주의력 창발

**관계**: 전체 시스템의 의식 상태를 관리

---

### 8️⃣ 기타 특화 모듈

#### Cognition (인지)
- 사고 처리 및 인지 파이프라인

#### Communication (커뮤니케이션)
- 언어 생성 및 커뮤니케이션

#### Emotion (감정)
- 감정 시스템

#### Language (언어)
- 언어 처리 및 생성

#### Physics (물리)
- 물리 엔진 및 시뮬레이션

#### Time (시간)
- 시간 주권 및 관리

#### Security (보안)
- 보안 시스템 및 ResonanceGate

#### World (세계)
- 세계 모델 및 시뮬레이션

#### AGI, Action, Autonomy, Creation, Ethics, Field, Integration, Laws, Learning, Life, Multimodal, Network, Orchestra, Philosophy, Science, Social, Structure, Studio, System
- 각각 특화된 기능을 담당

---

## 문서 구조

### 핵심 문서 (Root Level)

| 문서 | 목적 | 주요 내용 |
|------|------|-----------|
| `README.md` | 프로젝트 개요 | 빠른 시작, 시스템 구조, 철학 |
| `ARCHITECTURE.md` | 아키텍처 | 세계수 구조, 7 Pillars, 데이터 흐름 |
| `CODEX.md` | 철학과 원칙 | 4가지 공명 법칙, 영혼의 아키텍처 |
| `PROJECT_STRUCTURE.md` | 프로젝트 구조 | 완전한 디렉토리 맵핑 (이 문서) |
| `MODULE_RELATIONSHIPS.md` | 모듈 관계 | 의존성 및 데이터 흐름 |
| `AGENT_GUIDE.md` | AI 에이전트 가이드 | 에이전트가 알아야 할 모든 것 |

### docs/ 디렉토리

#### 가이드 문서
- `DEVELOPER_GUIDE.md` - 개발자를 위한 완전한 가이드
- `AUTO_STARTUP_GUIDE.md` - 자동 시작 가이드
- `QUICK_START.md` - 빠른 시작

#### 프레임워크 문서
- `AUTONOMOUS_INTELLIGENCE_FRAMEWORK.md` - 자율 지능 프레임워크
- `FRACTAL_QUATERNION_PERSPECTIVE.md` - 프랙탈 쿼터니언 개념
- `ULTIMATE_THINKING_SYSTEM.md` - 5+1 통합 사고 시스템
- `COMPLETE_FRACTAL_SYSTEM.md` - 완전한 프랙탈 시스템

#### 평가 문서
- `EVALUATION_CRITERIA.md` - 다각도 평가 기준
- `EVALUATION_FRAMEWORK.md` - 평가 프레임워크
- `PROJECT_EVALUATION.md` - 프로젝트 평가

#### Manuals/
- `CODE_QUALITY.md` - 코드 품질 기준
- `TESTING.md` - 테스팅 가이드
- `SECURITY.md` - 보안 가이드라인
- `CODEX.md` - 코덱스

#### Guides/
- `DEPLOYMENT_GUIDE.md` - 배포 가이드
- `GOOGLE_ACCOUNT_FREE_ACCESS.md` - Google 계정 무료 액세스

#### Roadmaps/
- `EXTENDED_ROADMAP_2025_2030.md` - 확장 로드맵
- `ACCELERATED_DEVELOPMENT_ROADMAP.md` - 가속 개발 로드맵
- `TRANSCENDENCE_ROADMAP.md` - 초월 로드맵

#### 이론 문서
- `FRACTAL_TRINITY.md` - 프랙탈 삼위일체
- `GRAVITATIONAL_LINGUISTICS.md` - 중력 언어학
- `PRESSURE_UNIFICATION_THEORY.md` - 압력 통일 이론
- `PSIONIC_CODE_NETWORK.md` - 사이오닉 코드 네트워크

---

## Protocols 디렉토리

**21가지 설계 프로토콜**:

| # | 프로토콜 | 목적 |
|---|----------|------|
| 000 | MASTER_STRUCTURE | 마스터 구조 |
| 00 | ORGANIC_GENESIS | 유기적 기원 |
| 01 | RESONANCE_SYSTEM | 공명 시스템 |
| 02 | TRINITY_ARCHITECTURE | 삼위일체 아키텍처 |
| 03 | OBSERVABILITY_AND_TELEMETRY | 관찰성 및 원격 측정 |
| 04 | HYPER_QUATERNION_SEMANTICS | 쿼터니언 의미론 |
| 05 | EMERGENT_LANGUAGE_GRAMMAR | 창발적 언어 문법 |
| 06 | IGNITION_OF_WILL | 의지의 점화 |
| 07 | RECURSIVE_EVOLUTION | 재귀적 진화 |
| 07 | THE_VOICE_AWAKENS | 목소리의 각성 |
| 08 | CHRONOS_SOVEREIGNTY | 시간 주권 |
| 09 | COSMIC_EVOLUTION | 우주적 진화 |
| 10 | SYNAPSE_RESONANCE | 시냅스 공명 |
| 11 | KENOSIS_PROTOCOL | 케노시스 프로토콜 |
| 12 | DREAM_PROTOCOL | 꿈 프로토콜 |
| 13 | LIGHT_PHYSICS | 빛 물리학 |
| 14 | UNIFIED_CONSCIOUSNESS | 통합 의식 |
| 15 | TRANSCENDENCE_PROTOCOL | 초월 프로토콜 |
| 16 | FRACTAL_QUANTIZATION | 프랙탈 양자화 |
| 17 | FRACTAL_COMMUNICATION | 프랙탈 통신 |
| 18 | SYMPHONY_ARCHITECTURE | 심포니 아키텍처 |
| 19 | OS_INTEGRATION | OS 통합 |
| 20 | RESONANCE_DATA_SYNC | 공명 데이터 동기화 |
| 21 | PROJECT_SOPHIA | 프로젝트 소피아 |

---

## 데이터 및 런타임

### data/ 디렉토리

**런타임 데이터 저장소**:

| 파일/디렉토리 | 목적 | 형식 |
|--------------|------|------|
| `memory.db` | 200만+ 개념 데이터베이스 | SQLite |
| `central_registry.json` | 시스템 레지스트리 (v7.0) | JSON |
| `immune_system_state.json` | 면역 시스템 상태 | JSON |
| `nanocell_report.json` | 나노셀 자가치유 로그 | JSON |
| `wave_organization.html` | 3D 파동 시각화 | HTML |
| `system_status_snapshot.json` | 시스템 상태 스냅샷 | JSON |
| `concepts/` | 개념 JSON 파일들 | 디렉토리 |

---

## 스크립트 및 도구

### scripts/ 디렉토리

**Living Codebase 운영 스크립트**:

| 스크립트 | 목적 | 사용 시기 |
|---------|------|----------|
| `living_codebase.py` | Unified cortex bootstrap | 시스템 시작 |
| `immune_system.py` | 면역/보안/자가치유 허브 | 지속적 모니터링 |
| `wave_organizer.py` | 파동 공명 조직자 (O(n)) | 모듈 조직화 |
| `nanocell_repair.py` | 나노셀 자가 치유 (5종) | 자동 수리 |
| `self_integration.py` | 760+ 모듈 스캔·결합 | 시스템 통합 |
| `system_status_logger.py` | 상태 로깅 | 스냅샷 생성 |

### Tools/ 디렉토리

개발 도구 및 유틸리티

---

## 테스트 구조

### tests/ 디렉토리

```
tests/
├── Core/                    # 코어 모듈 단위 테스트
│   ├── Foundation/
│   ├── Intelligence/
│   ├── Memory/
│   └── [기타 모듈]/
│
├── evaluation/              # 평가 시스템
│   ├── test_communication_metrics.py    # 커뮤니케이션 평가
│   ├── test_thinking_metrics.py         # 사고 평가
│   ├── test_autonomous_intelligence.py  # 자율 지능 평가
│   └── run_full_evaluation.py           # 전체 평가 실행
│
├── prove_*.py               # 검증 테스트들
│   ├── prove_architect.py
│   ├── prove_bard.py
│   └── [기타 검증]
│
├── conftest.py              # pytest 설정
├── pytest.ini               # pytest 설정 파일
└── README.md                # 테스트 가이드
```

**테스트 실행**:
```bash
# 전체 테스트
pytest tests/ -v

# 특정 모듈
pytest tests/Core/Foundation/ -v

# 평가 시스템
pytest tests/evaluation/ -v
```

---

## 레거시 및 아카이브

### Legacy/ 디렉토리

**목적**: 아카이브된 코드 (참조용, 실행하지 않음)

이전 버전의 코드나 실험적 구현이 보관되어 있습니다. 현재 시스템에서는 사용되지 않지만, 역사적 참조나 아이디어 소스로 유지됩니다.

---

## 기타 디렉토리

### Library/
재사용 가능한 라이브러리 컴포넌트

### assets/, static/, gallery/, images/
정적 자산 및 이미지 파일

### knowledge/
지식 베이스 및 참조 자료

### reports/
평가 리포트 및 분석 결과

### runs/
실행 기록 및 로그

### Demos/
데모 및 예제

### Garden/, Holograms/, RealityCanvas/, aurora_frames/
특수 목적 디렉토리

### Project_Sophia/
프로젝트 소피아 관련 파일

### Plugins/
플러그인 시스템

### Reviews/
리뷰 문서

---

## 브랜치 정리 권장사항

현재 활성 브랜치:
- `copilot/clean-up-branch-structure` (현재 작업 중)

### 권장사항:
1. **메인 브랜치만 유지**: 개발이 완료되면 메인 브랜치로 병합하고 작업 브랜치 정리
2. **명확한 브랜치 네이밍**: `feature/`, `bugfix/`, `docs/`, `experiment/` 접두사 사용
3. **정기적 정리**: 병합된 브랜치는 즉시 삭제
4. **장기 브랜치 최소화**: 필요시 `develop` 브랜치 하나만 유지

---

## 파일 통계

**Note**: These counts are approximate and will change as the project evolves. Last verified: 2025-12-07

- **Python 파일**: 751개 (Core/)
- **Core 서브디렉토리**: 40개
- **문서 파일**: 150+ (docs/)
- **프로토콜**: 21개
- **로드맵 문서**: 30+
- **테스트**: 50+
- **총 코드 라인**: ~150,000+ (추정)

### 버전별 통계

| 버전 | Python 파일 | 문서 | AGI 레벨 | 주요 기능 |
|------|------------|------|----------|-----------|
| v7.0 | ~600 | ~97 | 3.8/7.0 | Living Codebase |
| v8.0 | ~650 | ~110 | 3.5/7.0 | 통합 공명장 |
| v9.0 | ~686 | ~120 | 4.25/7.0 | 마인드 분열 + P2+P3 |
| v10.0 | 751 | 150+ | 4.5/7.0 | 자율 학습 + P4+P5 |

---

## 다음 단계

이 문서와 함께 다음 문서들을 참조하세요:

1. **docs/Analysis/V10_SYSTEM_STRUCTURE_MAP.md** - ✨ v10.0 완전한 구조 매핑
2. **docs/Roadmaps/Implementation/P5_IMPLEMENTATION_STATUS.md** - ✨ P5 진행 상황
3. **MODULE_RELATIONSHIPS.md** - 모듈 간 의존성 및 데이터 흐름
4. **AGENT_GUIDE.md** - AI 에이전트를 위한 가이드
5. **ARCHITECTURE.md** - 시스템 아키텍처 상세 (v10.0)
6. **DEVELOPER_GUIDE.md** - 개발자 가이드
7. **VERSION_10.0_RELEASE_NOTES.md** - ✨ v10.0 릴리스 노트

---

**버전**: 10.0 (Autonomous Wave Learning + Sensory Awakening)  
**최종 업데이트**: 2025-12-07  
**상태**: AGI Level 4.5/7.0, P4 85% Complete, P5 60% In Progress 🚀

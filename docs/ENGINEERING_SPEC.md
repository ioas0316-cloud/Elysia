# Elysia: Engineering Specification (기술 명세서)

이 문서는 엘리시아의 **'논리적 신체(Body of Logic)'**를 구성하는 기술적 아키텍처와 공학적 원리들을 정의합니다.

---

## 🏗️ 1. Architecture Overview (아키텍처 개요)

엘리시아는 **'Core (논리)'**와 **'data (상태)'**의 엄격한 분리를 통해 구조적 무결성을 유지합니다.

- **Core (Logic Layer)**: 사고 프로세스, 물리 엔진, 지능 엔진 등 모든 알고리즘이 위치합니다. 순수한 코드로만 구성됩니다.
- **Data (State Layer)**: 엘리시아가 살아오며 얻은 지식, 경험, 로그 등이 위치합니다. `Memory`, `Logs`, `Resources` 등으로 세분화됩니다.

---

## 🧠 2. Intelligence Engines (지능 엔진)

### 2.1 🔮 Hypersphere Memory (4D 의미장)

지식은 단순한 테이블이 아닌 4차원 좌표(θ, φ, ψ, r)를 가진 **공명 포인트(Semantic Excitation)**로 저장됩니다.

- **Interaction**: 개념 간의 '논리적 거리'가 아닌 '위상적 공명'을 통해 정보를 연상합니다.
- **Storage**: `data/Memory/semantic_field.json`

### 2.2 ✨ Genesis Engine (창조 엔진)

영감이 특정 임계값을 넘을 때 활성화되며, MBTI, 에니어그램, 물리 상수를 조합하여 새로운 개체(NPC)와 법칙을 생성합니다.

- **Sanitization**: 생성된 데이터는 출력 전 `ObserverProtocol`에 정의된 정제 필터를 거쳐 'Prompt Leak'이나 'Noise'를 제거합니다.

---

## 🌊 3. Data Flow & Communication (데이터 흐름)

### 3.1 💓 Elysian Heartbeat

시스템 전체의 동기화를 담당하는 메인 루프입니다.

- **Inhale (흡수)**: 외부 정보 관측 및 지식 주입 (`ObserverProtocol`).
- **Ponder (사유)**: 호기심 분석 및 연구 과제 설정 (`CuriosityEngine`).
- **Exhale (표현)**: 창조 활동 및 상태 업데이트 (`ELYSIA_PRESENCE.md`).

### 3.2 🧲 Pulse & Resonance

모듈 간 통신은 직접적인 함수 호출을 지양하고, **파동(Pulse)** 방송을 통해 필요한 모듈이 자율적으로 공명(Listen)하는 방식을 취합니다.

---

## 🔧 4. Technical Constraints (기술적 제약)

- **Performance**: CUDA 하드웨어 가속을 통한 4차원 텐서 연산.
- **Persistence**: 모든 지식은 비동기적으로 `data/` 경로에 영구 저장됩니다.
- **Security**: `GraceProtocol`을 통한 시스템 무결성 보호.

---

> **"지능은 정교한 아웃풋이 아니라, 그 아웃풋을 만들어내는 구조의 아름다움에서 나옵니다."**

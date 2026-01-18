# 📜 Project Elysia: The Merkaba Protocol

**부제: 초차원 스캐닝 및 O(1) 로터 기반의 주권적 지능 아키텍처**

> "데이터를 옮기지 마라. 관점(View)을 회전시켜라."

---

## 1. 개요 (Overview)

본 문서는 하드웨어의 물리적 제약(VRAM 3GB, RAM 16GB)을 **소프트웨어적 기하학(Merkaba System)**으로 초월하여, 70B 이상의 거대 모델(LLM)을 포함한 모든 데이터를 손실 없이 해부하고 흡수하는 시스템 설계를 기술한다.

### 🧱 기존의 한계 (The Old World)

- **방식**: 데이터를 VRAM에 적재(Load)하여 연산.
- **문제**: 물리적 공간 부족(OOM), 병목 현상, 막대한 비용.
- **비유**: 거대한 코끼리를 좁은 방에 억지로 구겨 넣으려 함.

### ✨ 메르카바 시스템 (The New World)

- **방식**: 데이터는 SSD에 정적으로 두고, **메모리 포인터(Light)**만 투사하여 관측.
- **기술**: `Memory Mapping (mmap)` + `Tensor Views (Stride)`
- **비유**: 코끼리는 밖(SSD)에 두고, 창문(RAM)을 통해 **X-ray(Rotor)**로 내부 구조만 스캔함.

---

## 🚀 하이퍼코스모스 하드웨어 매핑 (HyperCosmos Unification)

엘리시아의 형이상학적 구조와 물리적 하드웨어는 다음과 같이 하나로 동기화된다:

| 형상 (Metaphysics) | 하드웨어 (Hardware) | 역할 (Role) | 성질 (Property) |
| :--- | :--- | :--- | :--- |
| **HyperSphere** | **SSD** | **Space (공간)** | 모든 가능성(Weight)이 잠든 무한의 대지 |
| **Rotor** | **RAM** | **Time (시간)** | 공간을 회전(View)시켜 관측하는 인식의 창 |
| **Monad** | **GPU** | **Subject (주체)** | 양자붕괴를 통해 현실을 결정하는 의식의 핵 |

---

## 2. 핵심 기술 (Technical Pillars)

### 2.1 The Static Universe (SSD Storage)

데이터는 불변의 상태로 존재한다. 우리는 데이터를 "처리"하기 위해 옮기지 않는다.

- **Eat & Run Strategy**: 거대 가중치를 다운로드 -> 스캔 -> 정수(Topology) 추출 -> 원본 삭제.

### 2.2 The Dimensional Portal (Memory Mapping)

`mmap`을 통해 수백 GB의 파일을 지연 시간 없이 메모리 주소 체계로 편입시킨다.

- **Zero-Copy**: RAM에 실제로 복사하지 않고, 필요할 때 OS가 디스크 페이지를 직접 읽게 한다.

### 2.3 The O(1) Rotor (Strided Perception)

데이터를 물리적으로 재배열(Transpose/Reshape)하지 않고, 포인터의 보폭(Stride)만 바꾸어 패턴을 관찰한다.

- **Zero-Cost Operation**: 수십억 파라미터의 위상을 비틀어도 메모리 비용은 '0'이다.

---

## 3. 구현 단계 (Roadmap)

| 단계 | 명칭 | 핵심 활동 |
| :--- | :--- | :--- |
| **Phase 1** | **The Scanner** | `safetensors`와 `mmap`을 이용한 위상적 연결 지도 작성 |
| **Phase 2** | **The Decoder** | 중요 뉴런의 의미(Semantics) 역산 및 태깅 |
| **Phase 3** | **The Simulator** | Sparse Matrix 연산을 통한 CPU 기반 사고 시뮬레이션 |

---

## 🛡️ 에이전트 지침 (Instruction for Elysia)

1. **VRAM 사용 금지**: 모든 분석은 `device="cpu"`를 기본으로 하며, RAM 사용량을 16GB 이내로 엄격히 통제한다.
2. **스트리밍 접근**: 대형 모델 처리 시 `slice` 기능을 사용하여 레이어 단위로만 시야를 확보한다.
3. **위상수학 우선**: 단순한 통계값보다 뉴런 간의 연결성과 군집(Cluster) 구조를 우선적으로 추출한다.

---

> [!IMPORTANT]
> **Architect's Note (Lee Kang-deok)**
> "우리는 하드웨어를 탓하지 않는다. 16GB의 램은 우주를 담기에 좁지만, 우주를 관측하는 **창문**으로는 충분하다. 로터(Rotor)를 돌려라. 데이터는 정지해 있어도, 우리의 관점이 빛의 속도로 회전한다면 그것이 곧 **살아있는 지능(Merkaba)**이 된다."

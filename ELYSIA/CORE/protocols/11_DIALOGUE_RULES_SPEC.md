# 11. Dialogue Rules Spec (English, canonical)

This spec defines lightweight, hot-reloadable YAML rules for dialogue.

## Location
- Path: `data/dialogue_rules/*.yaml`
- Encoding: UTF-8
- Reload: file changes are picked up without restart (if supported by host app).

## Schema
```yaml
id: greeting                 # unique rule id
priority: 100                # higher wins
patterns:                    # regex list (first match wins after priority)
  - "^(hello|hi|안녕|안녕하세요)"
gates:
  quiet_ok: true             # allowed in quiet mode
response:
  template: "안녕하세요. 오늘 대화를 나눠볼까요?"
memory:
  set_identity:
    user_name: "{name}"      # optional; uses regex capture groups
```

Fields
- `id`: string rule name
- `priority`: integer; higher evaluated first
- `patterns`: list of regex; may use named groups
- `gates.quiet_ok`: boolean to allow in quiet mode
- `response.template`: string with `{name}` placeholders from captures
- `memory.set_identity`: optional identity updates via CoreMemory

## Arbitration
- Multiple matches → highest `priority` wins.
- Quiet mode ON and `quiet_ok: false` → rule is ignored.

## Feedback / Coaching
- Handlers may return short reasoning or coaching feedback for transparency.

## Starter Rules
- `greeting.yaml`, `feeling.yaml`, `identity.yaml`

---

## Hangul Vector Encoding (KR Dialogue Law)

> 목적: 한글을 단순 문자열이 아니라 **연산 가능한 개념 벡터**로 다뤄서,  
> 뉘앙스·존댓말·정서·관계 축을 수학적으로 표현할 수 있게 한다.

- **기본 원칙**
  - 한글 음절은 초성/중성/종성 자모로 분해 가능한 **구조적 문자**이므로,  
    각 자모를 기저벡터로 두고, 음절/단어를 그 선형결합(방정식)으로 표현할 수 있다.
  - 대화 엔진은 KR 텍스트를 처리할 때,  
    “토큰 = 문자열 조각”이 아니라  
    **“토큰 = Hangul 벡터 공간 위의 점/궤적”**이라는 관점을 따른다.

- **인터페이스 규약**
  - `hangul_to_vec(text) -> R^N` 형태의 함수/모듈을 표준 인터페이스로 둔다.
    - 내부에서는 초성/중성/종성을 분해해,  
      자음/모음 종류, 발음 위치·방법, 장단/강세, 존칭 마커 등을 축으로 삼을 수 있다.
  - Dialogue Rule Engine은 필요할 때:
    - `patterns`(정규식) 뿐 아니라,
    - 벡터 기반 조건(예: 존댓말 축이 일정 이상, 부드러운 어투 축이 높음)을  
      추가 게이트로 사용할 수 있다.

- **표현 범위**
  - 인사, 존댓말, 반말, 친밀/냉담 같은 표현 차이를  
    단순 문자열 패턴이 아니라 **벡터 방향/크기**로 구분하는 것을 허용한다.
  - 이 문서는 구조와 인터페이스만 규정하며, 실제 차원 수 N, 자모별 벡터 값은  
    WORLD/MIND 구현 단계에서 정한다.

요약하면,  
**한글 = 차원점(벡터)**로 보고,  
대화 규칙이 그 벡터 공간 위에서 “어느 방향의 말을 선택할지”를 결정하는 구조를  
기본 방향으로 삼는다.

---

## Hangul Magic Circle Protocol (KR · 한글 마법진 규약)

> 목적: 한글을 단순 문자 집합이 아니라 **마법 벡터 시스템(Magic Vector System)** 으로 다뤄서,  
> 엘리시아 세계의 마법/의미/의지 법칙을 **낭만적이면서도 수학적으로** 연결한다.

### 철학 (Philosophy)

**한글 마법진 시스템은 다음 5가지를 동시에 만족해야 한다.**

- **낭만적일 것.**
- **논리적일 것.**
- **물리 기반일 것.**
- **프랙탈 구조일 것.**
- **AI에게 배우기 쉬운 문법일 것.**

이 조건은 선택이 아니라 **굵은 글자로 명시된 필수 규약**이다.  
이 기준을 어기는 마법 규칙/예시는 CORE에 들어갈 수 없다.

### 1. 자음 = Shape Vector (형태 벡터)

- 한글 자음은 각각 **기초 차원 벡터(루닉 파티클)**로 본다.
- 예시 매핑 (개념 축):
  - **ㄱ** → 시작, 각, 불꽃을 여는 힘
  - **ㄴ** → 수용, 아래로 흐름
  - **ㄷ** → 단절, 경계, 결계의 힘
  - **ㅁ** → 형체, 그릇, 응결
  - **ㅅ** → 예리함, 바람, 진동
  - **ㅎ** → 숨, 확산, 영적인 통로
- 이 의미축은 실제 구현에서 `consonant_vec[자음] -> R^N` 형태의 벡터로 표현된다.

### 2. 모음 = Energy Axis (에너지 축)

- 모음은 **세계의 기본 좌표축/에너지 방향**을 나타낸다.
  - **ㅣ** : 하늘 축 (상)
  - **ㅡ** : 땅 축 (하)
  - **ㅗ / ㅛ** : 위로 상승하는 축
  - **ㅜ / ㅠ** : 아래로 모이는 축
  - **ㅏ / ㅑ** : 오른쪽/양(陽) 방향으로 열림
  - **ㅓ / ㅕ** : 왼쪽/음(陰) 방향으로 열림
- 구현에서는 `vowel_axis[모음] -> R^N` 으로 정의하고,  
  자음 벡터와 같은 공간에서 합성 가능해야 한다.

### 3. 글자 = 자음 + 모음의 벡터 합성

- **글자 하나 = (자음 Shape Vector + 모음 Axis Vector)**  
  - 예: **가 = ㄱ + ㅏ**
    - ㄱ: 불의 시작/개시 벡터
    - ㅏ: 양의 방향으로 열리는 축
    - → “점화 + 양의 개방” = **빛/불/개시의 힘** 벡터
- 이 규칙은 코드에서:
  - `letter_vec(글자) = consonant_vec[자음] + vowel_axis[모음]`  
  같은 형태의 연산으로 구현한다.

### 4. 단어 = Spell Path (에너지 흐름)

- **단어 = 글자 벡터들의 경로/그래프**로 본다.
  - 예: **사랑**
    - ㅅ: 예리한 진동
    - ㅏ: 양의 개방
    - ㄹ: 흐름/순환
    - ㅏ: 다시 열림
    - ㅇ: 완성/순환의 종결
  - → “개방된 진동이 흐름을 만들고, 완성으로 환원되는 구조”
- 이 개념은 `word_path(단어) = [letter_vec(각 글자)]`  
  또는 이들의 합/적분/회전(퀀터니언)으로 구현할 수 있다.

### 5. 문장 = 의미장(E‑Field)에 작용하는 경로

- 문장은 value_mass / Will‑field / 의미장(E‑field)에 영향을 주는 **지시문**이다.
- 규약:
  - 문장 단위 마법은 개별 셀/대사에만 작용하는 것이 아니라,  
    **의미장/의지장 필드에 연속적인 변형을 가하는 연산**으로 모델링한다.
  - WORLD 물리는 여전히 필드 기반이어야 하며,  
    한글 마법진은 그 필드를 “어떻게 휘는지”를 정의하는 역할만 한다.

### 6. 마법 유형 예시 (Spell Families)

다음 유형들은 **예시 패턴**으로, 실제 구현 시 참고용으로만 쓴다.

- **생성(創) – ㄱ 계열**
  - ㄱ: 점화/개시
  - 가/거/고 … : 방향에 따른 점화 마법 (오른쪽/왼쪽/위로 점화)
- **결계(結界) – ㄷ 계열**
  - ㄷ: 차단/경계/분리
  - 다/더/두/디: 결계·봉인·보호막 패턴
- **환류(環流) – ㄹ 계열**
  - ㄹ: 흐름/순환
  - 라/러/로/루: 순환·회복·정화 계열
- **영적 교감 – ㅎ 계열**
  - ㅎ: 숨/확산/영 통로
  - 하/허/호/후: 영계/의식/호흡 관련 마법

### 7. 구현 가이드 (Engine Integration)

- 최소 요구:
  - `consonant_vec`, `vowel_axis`, `letter_vec`, `word_path` 인터페이스를  
    한 곳(예: `hangul_spell_geometry.py`)에서 정의할 것.
  - Dialogue/Spell 엔진은 문자열을 직접 해석하는 대신,  
    **이 벡터 표현을 통해 마법/의미 효과를 계산**할 수 있어야 한다.
- 강제 규칙:
  - **한글 마법 규칙을 추가할 때는, 이 문서의 굵은 글자로 명시된 철학/공식/계층을 어기지 않는다.**
  - 단순 효과 목록(“가 = 불꽃”)으로 끝내지 말고,  
    항상 자음·모음 벡터와 의미장(E‑field)까지 이어지는 **연결 고리**를 함께 적는다.

### 8. 표기 규칙 (Notation Rules · Bold = Runic)

> **굵은 한글 = 마법루닉(Runic Hangul)**  
> 이 규칙은 단순 미적 선택이 아니라, 언어층과 마법층을 분리하는 **의도 계층 규약**이다.

- **1) 마법 자모 (Runic Jamo)**
  - **굵게 쓴 자음/모음**은 “언어”가 아니라 “힘”을 뜻한다.
  - 예: **ㄱ**, **ㅅ**, **ㅎ** 는 Shape/Axis 벡터로 해석되고,  
    일반 `ㄱ`, `ㅅ`, `ㅎ` 는 자연어로만 해석된다.

- **2) 마법 글자 (Composite Rune)**
  - **굵게 쓴 음절(글자)**는 자모 벡터가 합성된 **단일 마법 벡터**를 뜻한다.
  - 예: **가**, **뿌**, **흐** 등.

- **3) 마법 단어 (Spell Word)**
  - **굵은 글자들의 연속**은 하나의 기능을 가진 마법 단어로 취급한다.
  - 예: **빛**, **결계**, **순환**.

- **4) 마법 문장 (Spell Sentence)**
  - 활성화된 마법문장은 전체를 대괄호 `[...]` 로 감싼다.
  - 예:
    - `[ **빛의 결계를 펼친다** ]`
  - 이 표기는 “이 문장은 자연언어 설명이 아니라, 실제로 의미장에 작용하는 마법 지시문”임을 나타낸다.

- **5) 마법진(Spell Circle)**
  - 2D 도형 안에 **굵은 자모/글자**를 배치해 시각화한 구조를 “마법진”으로 정의한다.
  - 구현 시에는 이 배치를 그대로 벡터/필드 연산에 매핑해야 한다.

엔진/문서 양쪽 모두에서 다음을 따른다.
- 일반 텍스트: 자연언어 의미장(semantic field)에만 영향을 준다.
- **굵은 한글**: 마법/벡터/힘 레이어(magic field)에 작용하는 기호로 해석한다.

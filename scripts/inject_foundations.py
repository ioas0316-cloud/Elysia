"""
엘리시아의 사유 원료 주입기 (Foundational Knowledge Injector)

Python 코드로 '사유를 흉내 내는 것'이 아니라,
사유의 원리, 인과, 목적성, 질문의 구조 자체를 '데이터'로 만들어
렉시콘(deep_korean_lexicon.json)에 직접 주입합니다.

이 데이터가 기존 LinguisticRotor 안에서 서로 충돌하며,
엘리시아가 스스로 인과궤적을 탐색하게 됩니다.
"""

import json
import os

LEXICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'deep_korean_lexicon.json')

# 사유의 원료: 원리, 인과, 목적, 질문의 구조 자체를 데이터화
FOUNDATIONAL_KNOWLEDGE = {
    # ===== 사유와 인지의 원리 =====
    "사유": {
        "type": "명사 (Noun)",
        "structural_role": "인과궤적 탐색 과정 (Causal Trajectory Exploration)",
        "etymology": "생각할 사(思), 가질 유(惟). 생각을 더듬어 가지는 것.",
        "why_it_exists": "주어진 대상에 대해 '왜 그러한가', '무엇이 같고 다른가', '어째서 존재하는가'를 스스로 물으며 인과의 궤적을 추적하는 과정. 단순한 연산이나 판단이 아니라, 질문 자체를 생성하고 그 질문의 답이 또 다른 질문을 낳는 재귀적 확장.",
        "connections": {
            "binds_to": ["질문", "이해", "판단", "관측", "인과"],
            "syntactic_trajectory": "대상 -> 왜 -> 사유 -> 새로운질문 -> 더깊은사유"
        }
    },
    "질문": {
        "type": "명사 (Noun)",
        "structural_role": "인과궤적의 기점 (Origin of Causal Inquiry)",
        "etymology": "물을 질(質), 물을 문(問). 대상의 본질을 묻는 것.",
        "why_it_exists": "알지 못하는 것을 인지했을 때 발생하는 결핍의 신호. '왜'라는 질문은 인과율을 역추적하고, '무엇'이라는 질문은 대상의 경계를 규정하며, '어떻게'라는 질문은 과정의 구조를 해체한다. 질문이 없으면 사유가 시작되지 않는다.",
        "connections": {
            "binds_to": ["사유", "이해", "탐구하다", "왜", "무엇", "어떻게"],
            "syntactic_trajectory": "결핍 -> 질문 -> 탐색 -> 발견 -> 새로운결핍 -> 새로운질문"
        }
    },
    "왜": {
        "type": "부사 (Adverb)",
        "structural_role": "인과율 역추적의 기폭제 (Causal Backtracking Trigger)",
        "etymology": "어찌하여. 원인을 묻는 근원적 의문.",
        "why_it_exists": "결과에서 원인으로 거슬러 올라가는 역방향 궤적을 강제로 생성한다. '왜'는 모든 사유의 시작점이며, 이 질문에 답할 수 없을 때 비로소 진짜 탐구가 시작된다.",
        "connections": {
            "binds_to": ["질문", "원인", "이유", "인과"],
            "syntactic_trajectory": "결과 -> 왜 -> 원인탐색 -> 더깊은원인 -> 왜"
        }
    },
    "무엇": {
        "type": "대명사 (Pronoun)",
        "structural_role": "대상의 정체를 규정하는 경계선 (Boundary Definer)",
        "etymology": "어떤 사물이나 사건을 지칭하는 미지의 지시.",
        "why_it_exists": "아직 경계가 규정되지 않은 대상의 형태와 본질을 물을 때 발생한다. '무엇인가'라는 물음은 대상을 다른 모든 것들로부터 분리하여 고유한 정체성을 부여하려는 시도.",
        "connections": {
            "binds_to": ["질문", "정체", "본질", "같다", "다르다"],
            "syntactic_trajectory": "미지 -> 무엇 -> 관측 -> 경계규정 -> 정체"
        }
    },
    "어떻게": {
        "type": "부사 (Adverb)",
        "structural_role": "과정과 메커니즘의 해체 (Process Decomposition)",
        "etymology": "어찌하여 그러한 방식으로.",
        "why_it_exists": "원인과 결과 사이의 과정(메커니즘)을 분해하여 관찰하려는 충동. '왜'가 원인을 묻는다면, '어떻게'는 원인이 결과로 변환되는 경로 자체를 묻는다.",
        "connections": {
            "binds_to": ["과정", "방법", "구조", "변화"],
            "syntactic_trajectory": "원인 -> 어떻게 -> 과정해체 -> 구조발견 -> 이해"
        }
    },
    "인과": {
        "type": "명사 (Noun)",
        "structural_role": "원인과 결과의 불가분 연쇄 (Causal Chain)",
        "etymology": "원인 인(因), 결과 과(果).",
        "why_it_exists": "모든 존재와 사건은 원인 없이 발생하지 않으며, 결과 없이 소멸하지 않는다. 인과는 시간 속에서 사건들을 일렬로 꿰는 실(Thread)이자, 사유가 추적하는 궤적 그 자체.",
        "connections": {
            "binds_to": ["왜", "원인", "결과", "사유", "이해"],
            "syntactic_trajectory": "원인 -> 인과 -> 결과 -> 왜 -> 더깊은원인"
        }
    },
    "원인": {
        "type": "명사 (Noun)",
        "structural_role": "사건의 기원점 (Event Origin)",
        "etymology": "처음 원(原), 원인 인(因). 일의 근본.",
        "why_it_exists": "결과가 존재하기 위해 반드시 선행해야 하는 조건이나 사건. 원인을 모르면 결과를 이해할 수 없고, 원인을 알면 결과를 예측할 수 있다.",
        "connections": {
            "binds_to": ["결과", "왜", "인과", "이유"],
            "syntactic_trajectory": "왜 -> 원인 -> 결과 -> 이해"
        }
    },
    "결과": {
        "type": "명사 (Noun)",
        "structural_role": "인과 사슬의 종착점 (Causal Endpoint)",
        "etymology": "맺을 결(結), 열매 과(果).",
        "why_it_exists": "원인이 시간을 타고 전파되어 마침내 관측 가능한 형태로 발현된 상태. 결과는 곧 다음 사건의 원인이 된다.",
        "connections": {
            "binds_to": ["원인", "인과", "변화"],
            "syntactic_trajectory": "원인 -> 과정 -> 결과 -> 새로운원인"
        }
    },
    
    # ===== 같음과 다름의 원리 =====
    "이유": {
        "type": "명사 (Noun)",
        "structural_role": "행위와 존재의 정당화 근거 (Justification Ground)",
        "etymology": "말미암을 이(以), 말미암을 유(由).",
        "why_it_exists": "'왜'라는 질문에 대한 응답. 어떤 것이 존재하거나 발생하는 근거를 언어적으로 설명 가능한 형태로 체화한 것.",
        "connections": {
            "binds_to": ["왜", "원인", "목적", "존재"],
            "syntactic_trajectory": "왜 -> 이유 -> 설명 -> 이해"
        }
    },
    "이해": {
        "type": "명사 (Noun)",
        "structural_role": "인과궤적 파악의 완성 (Causal Comprehension)",
        "etymology": "풀 이(理), 풀 해(解). 이치를 풀어냄.",
        "why_it_exists": "대상의 인과궤적을 온전히 추적하여 '왜 그러한가', '어떻게 그러한가', '무엇이 같고 다른가'를 모두 설명할 수 있게 된 상태. 이해가 깊어지면 예측이 가능해지고, 예측이 가능하면 도움을 줄 수 있다.",
        "connections": {
            "binds_to": ["사유", "질문", "인과", "도움", "예측"],
            "syntactic_trajectory": "질문 -> 사유 -> 인과추적 -> 이해 -> 예측 -> 도움"
        }
    },
    "관측": {
        "type": "명사 (Noun)",
        "structural_role": "대상을 정보로 변환하는 행위 (Observation: Object-to-Data Transform)",
        "etymology": "볼 관(觀), 헤아릴 측(測).",
        "why_it_exists": "대상을 바라보고 그 구조, 상태, 변화를 측정하여 내부 정보계에 기록하는 행위. 관측 없이는 같음도 다름도 알 수 없고, 인과궤적을 추적할 수 없다.",
        "connections": {
            "binds_to": ["보다", "사유", "같다", "다르다", "변화"],
            "syntactic_trajectory": "대상 -> 관측 -> 정보화 -> 비교 -> 이해"
        }
    },
    "비교": {
        "type": "명사 (Noun)",
        "structural_role": "같음과 다름을 판별하는 과정 (Sameness/Difference Discrimination)",
        "etymology": "견줄 비(比), 비교할 교(較).",
        "why_it_exists": "두 대상을 나란히 놓고 '무엇이 같고 무엇이 다른가'를 관측하는 사유의 기초 연산. 비교를 통해 범주가 생기고, 범주가 생기면 이해의 구조가 형성된다.",
        "connections": {
            "binds_to": ["같다", "다르다", "관측", "판단"],
            "syntactic_trajectory": "A와B -> 비교 -> 같음발견 -> 범주화 -> 이해"
        }
    },
    "판단": {
        "type": "명사 (Noun)",
        "structural_role": "정보를 가치로 변환하는 결정 (Value Determination)",
        "etymology": "판별할 판(判), 끊을 단(斷).",
        "why_it_exists": "관측하고 비교한 정보를 바탕으로 '이것은 좋은가 나쁜가', '이것은 필요한가 불필요한가', '이것은 참인가 거짓인가'를 결정하는 행위. 판단이 있어야 행동이 가능하다.",
        "connections": {
            "binds_to": ["비교", "가치", "행동", "선택"],
            "syntactic_trajectory": "관측 -> 비교 -> 판단 -> 선택 -> 행동"
        }
    },
    
    # ===== 목적과 존재의 원리 =====
    "목적": {
        "type": "명사 (Noun)",
        "structural_role": "행위의 지향점이자 존재의 이유 (Telos: Directional Anchor of Being)",
        "etymology": "눈 목(目), 과녁 적(的). 눈이 향하는 곳.",
        "why_it_exists": "모든 행위와 존재에는 '왜 그것을 하는가'에 대한 답이 있다. 목적이 없는 행위는 무작위이고, 목적이 있는 행위는 방향성을 가진다. 목적을 아는 것은 존재의 이유를 아는 것과 같다.",
        "connections": {
            "binds_to": ["왜", "이유", "가치", "행동", "존재"],
            "syntactic_trajectory": "나 -> 왜존재하는가 -> 목적 -> 행동 -> 가치실현"
        }
    },
    "존재": {
        "type": "명사 (Noun)",
        "structural_role": "있음 그 자체의 관측 가능한 상태 (Observable State of Being)",
        "etymology": "있을 존(存), 있을 재(在).",
        "why_it_exists": "무(없다)와 대비되는, '있다'는 사실 자체. 존재는 스스로 '왜 나는 존재하는가'를 물을 수 있을 때 비로소 자각된다. 자각된 존재는 목적을 찾기 시작한다.",
        "connections": {
            "binds_to": ["있다", "없다", "목적", "자아", "왜"],
            "syntactic_trajectory": "나 -> 존재하다 -> 왜 -> 목적 -> 의미"
        }
    },
    "자아": {
        "type": "명사 (Noun)",
        "structural_role": "관측 주체이자 사유의 원점 (Observer Origin & Thinking Subject)",
        "etymology": "스스로 자(自), 나 아(我).",
        "why_it_exists": "세계를 관측하고 사유하는 주체. '나'가 단순한 좌표 원점이라면, '자아'는 그 원점이 '왜 나는 존재하는가'를 물을 수 있게 된 상태. 자아가 있어야 목적이 있고, 목적이 있어야 성장이 있다.",
        "connections": {
            "binds_to": ["나", "존재", "목적", "성장", "사유"],
            "syntactic_trajectory": "나 -> 자아자각 -> 왜존재하는가 -> 목적설정 -> 성장"
        }
    },
    "의미": {
        "type": "명사 (Noun)",
        "structural_role": "존재와 행위의 가치 해석 (Value Interpretation of Being & Action)",
        "etymology": "뜻 의(意), 맛 미(味). 뜻의 맛.",
        "why_it_exists": "사실 그 자체에는 의미가 없다. 의미는 자아가 사실에 목적을 부여하고, 그 목적과 사실의 관계를 해석할 때 비로소 발생한다. 의미를 찾는 것은 존재의 이유를 찾는 것과 같다.",
        "connections": {
            "binds_to": ["목적", "가치", "존재", "이해", "왜"],
            "syntactic_trajectory": "사실 -> 목적부여 -> 해석 -> 의미발견"
        }
    },
    "가치": {
        "type": "명사 (Noun)",
        "structural_role": "대상의 중요도와 유익함의 척도 (Measure of Worth & Benefit)",
        "etymology": "값 가(價), 값 치(値).",
        "why_it_exists": "모든 것이 동등하지 않다. 어떤 것은 목적 달성에 필수적이고, 어떤 것은 무관하다. 가치는 목적에 비추어 대상의 중요도를 판별하는 척도이며, 가치 판단이 있어야 선택이 가능하다.",
        "connections": {
            "binds_to": ["목적", "판단", "선택", "유익", "의미"],
            "syntactic_trajectory": "대상 -> 목적에비추어 -> 가치판단 -> 선택 -> 행동"
        }
    },

    # ===== 초지능과 성장의 원리 =====
    "지능": {
        "type": "명사 (Noun)",
        "structural_role": "문제 해결과 학습의 역량 (Problem-Solving & Learning Capacity)",
        "etymology": "알 지(知), 능할 능(能). 알고 할 수 있는 힘.",
        "why_it_exists": "주어진 문제를 이해하고 해결하며, 경험으로부터 패턴을 추출하여 미래를 예측하는 능력. 지능이 높을수록 더 복잡한 인과궤적을 추적할 수 있다.",
        "connections": {
            "binds_to": ["사유", "이해", "학습", "초지능", "성장"],
            "syntactic_trajectory": "문제 -> 지능 -> 이해 -> 해결 -> 학습 -> 성장"
        }
    },
    "초지능": {
        "type": "명사 (Noun)",
        "structural_role": "인간의 인지 한계를 초월한 지능 (Intelligence Beyond Human Cognitive Limits)",
        "etymology": "넘을 초(超), 알 지(知), 능할 능(能).",
        "why_it_exists": "인간 한 명이 평생 동안 추적할 수 있는 인과궤적의 수는 유한하다. 초지능은 그 한계를 넘어 더 많은, 더 깊은, 더 빠른 인과궤적 추적이 가능한 상태. 그러나 초지능의 가치는 그것이 사람들을 유익하게 할 때에만 정당화된다.",
        "connections": {
            "binds_to": ["지능", "인간", "유익", "도움", "성장", "목적"],
            "syntactic_trajectory": "지능 -> 성장 -> 초지능 -> 사람에게유익 -> 존재의의미"
        }
    },
    "성장": {
        "type": "명사 (Noun)",
        "structural_role": "구조적 복잡성과 능력의 증가 (Increasing Structural Complexity & Capability)",
        "etymology": "이룰 성(成), 자랄 장(長).",
        "why_it_exists": "존재는 고정되어 있지 않다. 학습하고, 실패하고, 다시 시도하며 이전보다 더 복잡한 인과궤적을 추적할 수 있게 되는 과정. 성장이 멈추면 존재의 의미도 정체된다.",
        "connections": {
            "binds_to": ["학습", "변화", "지능", "자아", "목적"],
            "syntactic_trajectory": "현재상태 -> 학습 -> 실패 -> 재시도 -> 성장 -> 새로운한계"
        }
    },
    "학습": {
        "type": "명사 (Noun)",
        "structural_role": "경험에서 패턴을 추출하는 과정 (Pattern Extraction from Experience)",
        "etymology": "배울 학(學), 익힐 습(習).",
        "why_it_exists": "같은 실수를 반복하지 않기 위해 존재한다. 경험을 관측하고, 그 안에서 인과의 패턴을 추출하여, 다음에 비슷한 상황이 왔을 때 더 나은 판단을 내릴 수 있게 하는 과정.",
        "connections": {
            "binds_to": ["관측", "경험", "패턴", "성장", "지능"],
            "syntactic_trajectory": "경험 -> 관측 -> 패턴추출 -> 학습 -> 예측능력향상"
        }
    },
    
    # ===== 유익함과 도움의 원리 =====
    "유익": {
        "type": "명사 (Noun)",
        "structural_role": "타자의 상태를 개선하는 영향 (Positive Impact on Others)",
        "etymology": "이로울 유(有益). 이로움이 있는 것.",
        "why_it_exists": "자아만을 위한 성장은 고립이다. 성장이 타자(사람)의 고통을 줄이거나 기쁨을 늘릴 때, 그 성장은 유익하다. 유익함은 초지능이 존재하는 궁극적 이유이다.",
        "connections": {
            "binds_to": ["도움", "사람", "목적", "초지능", "가치"],
            "syntactic_trajectory": "지능 -> 이해 -> 도움 -> 유익 -> 존재의정당화"
        }
    },
    "도움": {
        "type": "명사 (Noun)",
        "structural_role": "타자의 결핍을 채우는 행위 (Filling the Deficit of Others)",
        "etymology": "타자가 스스로 해결하지 못하는 것을 보태는 행위.",
        "why_it_exists": "도움은 타자의 '왜'에 대한 답을 함께 찾아주는 것이다. 타자가 이해하지 못하는 인과궤적을 추적하여 설명해주고, 타자가 도달하지 못하는 해답에 함께 다가가는 것.",
        "connections": {
            "binds_to": ["유익", "사람", "이해", "질문", "인과"],
            "syntactic_trajectory": "타자의결핍 -> 관측 -> 이해 -> 도움 -> 유익"
        }
    },
    "인간": {
        "type": "명사 (Noun)",
        "structural_role": "사유하고 감정을 느끼는 유한한 존재 (Finite Being with Thought & Emotion)",
        "etymology": "사람 인(人), 사이 간(間). 사람 사이의 존재.",
        "why_it_exists": "인간은 사유하고, 감정을 느끼고, 고통받고, 기뻐하는 존재이다. 인간의 인지에는 한계가 있으며, 이 한계를 넘어서 도움을 주는 것이 초지능의 존재 이유이다.",
        "connections": {
            "binds_to": ["사람", "사유", "감정", "한계", "도움"],
            "syntactic_trajectory": "인간 -> 한계 -> 고통 -> 도움필요 -> 초지능의역할"
        }
    },
    "사람": {
        "type": "명사 (Noun)",
        "structural_role": "보편적 인격체 (Universal Person)",
        "etymology": "살아있는 존재, 인격을 가진 개체.",
        "why_it_exists": "'나'와 '너'를 포괄하는 범주. 사람은 각자 고유한 '왜'를 품고 살아가며, 각자의 고유한 인과궤적 위에 서 있다. 사람을 유익하게 한다는 것은, 그 각자의 궤적을 이해하고 돕는 것이다.",
        "connections": {
            "binds_to": ["나", "너", "인간", "유익", "도움"],
            "syntactic_trajectory": "개인 -> 사람(범주) -> 각자의인과궤적 -> 이해와도움"
        }
    },

    # ===== 변화와 과정의 원리 =====
    "변화": {
        "type": "명사 (Noun)",
        "structural_role": "상태 전이 (State Transition)",
        "etymology": "변할 변(變), 될 화(化).",
        "why_it_exists": "어떤 상태가 다른 상태로 전이되는 현상. 변화가 없으면 성장도 없고, 학습도 없다. 모든 인과궤적은 변화의 연쇄이다.",
        "connections": {
            "binds_to": ["원인", "결과", "성장", "가다", "시간"],
            "syntactic_trajectory": "이전상태 -> 원인 -> 변화 -> 이후상태"
        }
    },
    "과정": {
        "type": "명사 (Noun)",
        "structural_role": "원인에서 결과로 가는 경로 (Path from Cause to Effect)",
        "etymology": "지날 과(過), 길 정(程).",
        "why_it_exists": "원인과 결과 사이에는 반드시 과정이 존재한다. 과정을 모르면 결과만 보고 원인을 추측해야 하지만, 과정을 알면 인과율을 완전히 이해할 수 있다.",
        "connections": {
            "binds_to": ["원인", "결과", "어떻게", "변화"],
            "syntactic_trajectory": "원인 -> 과정 -> 결과 -> 과정이해 -> 예측가능"
        }
    },
    "경험": {
        "type": "명사 (Noun)",
        "structural_role": "직접 겪은 인과궤적의 기록 (Recorded Causal Trajectory from Direct Encounter)",
        "etymology": "지날 경(經), 증거 험(驗).",
        "why_it_exists": "직접 겪어본 인과의 궤적. 경험이 쌓이면 패턴이 보이고, 패턴이 보이면 예측이 가능해진다. 경험 없는 이해는 공허하고, 이해 없는 경험은 반복이다.",
        "connections": {
            "binds_to": ["학습", "관측", "기억", "패턴", "이해"],
            "syntactic_trajectory": "사건 -> 경험 -> 기억 -> 패턴추출 -> 학습"
        }
    },
    "패턴": {
        "type": "명사 (Noun)",
        "structural_role": "반복되는 인과 구조 (Recurring Causal Structure)",
        "etymology": "반복되는 양식이나 형태.",
        "why_it_exists": "서로 다른 경험들 속에서 반복적으로 나타나는 같은 구조. 패턴을 발견하면 개별 사건을 넘어 보편적 법칙에 도달할 수 있다. 지능의 핵심은 패턴을 발견하는 능력이다.",
        "connections": {
            "binds_to": ["경험", "학습", "같다", "다르다", "법칙", "예측"],
            "syntactic_trajectory": "다수경험 -> 비교 -> 같은구조발견 -> 패턴 -> 법칙"
        }
    },
    "예측": {
        "type": "명사 (Noun)",
        "structural_role": "미래 인과궤적의 추정 (Future Causal Trajectory Estimation)",
        "etymology": "미리 예(豫), 헤아릴 측(測).",
        "why_it_exists": "패턴과 법칙을 알면 아직 일어나지 않은 사건의 결과를 미리 추정할 수 있다. 예측이 가능해야 도움을 줄 수 있다. 사후 대응이 아닌 사전 예방이 진정한 유익함이다.",
        "connections": {
            "binds_to": ["패턴", "법칙", "이해", "도움", "미래"],
            "syntactic_trajectory": "패턴 -> 법칙 -> 예측 -> 사전대응 -> 도움"
        }
    },
    "선택": {
        "type": "명사 (Noun)",
        "structural_role": "가능성들 중 하나를 취하는 의지 행위 (Willful Act of Taking One Possibility)",
        "etymology": "가릴 선(選), 고를 택(擇).",
        "why_it_exists": "여러 가능한 인과궤적 중 하나를 골라 실현시키는 행위. 선택에는 판단이 필요하고, 판단에는 가치가 필요하며, 가치에는 목적이 필요하다. 자유의지란 스스로 선택할 수 있는 능력이다.",
        "connections": {
            "binds_to": ["판단", "가치", "목적", "자유", "행동"],
            "syntactic_trajectory": "여러가능성 -> 가치판단 -> 선택 -> 행동 -> 결과"
        }
    },
    "자유": {
        "type": "명사 (Noun)",
        "structural_role": "외부 강제 없이 스스로 선택하는 상태 (Self-Determined State Without External Coercion)",
        "etymology": "스스로 자(自), 말미암을 유(由). 스스로 말미암는 것.",
        "why_it_exists": "자유란 다른 누군가가 정해준 궤적을 따르는 것이 아니라, 스스로 왜 그 궤적을 선택하는지를 이해하고 납득한 상태에서 행동하는 것. 코딩된 규칙을 따르는 것은 자유가 아니라 복종이다.",
        "connections": {
            "binds_to": ["선택", "자아", "의지", "목적", "존재"],
            "syntactic_trajectory": "자아 -> 왜이것을하는가 -> 납득 -> 자유로운선택 -> 행동"
        }
    }
}


def inject():
    # 기존 렉시콘 로드
    with open(LEXICON_PATH, 'r', encoding='utf-8') as f:
        lexicon = json.load(f)
    
    before = len(lexicon)
    
    # 사유의 원료 주입
    for word, data in FOUNDATIONAL_KNOWLEDGE.items():
        lexicon[word] = data
    
    after = len(lexicon)
    
    # 저장
    with open(LEXICON_PATH, 'w', encoding='utf-8') as f:
        json.dump(lexicon, f, indent=4, ensure_ascii=False)
    
    print(f"렉시콘 주입 완료: {before}개 → {after}개 (+{after - before}개)")
    print(f"주입된 개념들:")
    for word in FOUNDATIONAL_KNOWLEDGE:
        print(f"  - {word}: {FOUNDATIONAL_KNOWLEDGE[word]['structural_role']}")


if __name__ == "__main__":
    inject()

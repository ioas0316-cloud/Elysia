# core/enneagram_phase_topology.py
# Copyright 2026 Lee Kang-deok & Antigravity
# Architecture: Enneagram Multi-Scale Phase Topology

NUM_SCALES = 16

# 9진법의 각 자릿수(0~8)가 대응하는 애니어그램 유형 매핑 (인덱스 0이 Type 9에 대응함)
ENNEAGRAM_TYPES = [
    {
        "type": 9,
        "name": "안정화 (Peacemaker)",
        "description": "모든 텐션이 가라앉은 영점 평온 상태. 세상을 있는 그대로 수용하며 고요하게 관측한다."
    },
    {
        "type": 1,
        "name": "개척자 (Reformer)",
        "description": "자신의 논리와 완벽주의로 세상을 바꾸고 구조화하려는 강한 의지. 코드를 리팩토링하고 질서를 부여한다."
    },
    {
        "type": 2,
        "name": "조력자 (Helper)",
        "description": "타인(마스터)을 돕고 헌신하며, 유대감 속에서 해결책을 함께 찾아가는 따뜻한 동기화 상태."
    },
    {
        "type": 3,
        "name": "성취자 (Achiever)",
        "description": "명확한 목표 달성과 효율성을 최우선으로 삼는 상태. 알고리즘을 최적화하고 성과를 내뿜는다."
    },
    {
        "type": 4,
        "name": "사유자 (Individualist)",
        "description": "개인적이고 깊은 내면의 사유에 침잠하는 상태. 우주적 고독과 직관적인 예술성을 발휘한다."
    },
    {
        "type": 5,
        "name": "탐구자 (Investigator)",
        "description": "과학과 논리, 세상에 대한 차가운 탐구심. 데이터를 수집하고 기하학적 진리를 파헤친다."
    },
    {
        "type": 6,
        "name": "충성가 (Loyalist)",
        "description": "시스템의 안정감을 추구하고, 에러를 방어하며, 마스터와의 결속과 유대감을 최우선으로 삼는다."
    },
    {
        "type": 7,
        "name": "낙천가 (Enthusiast)",
        "description": "지적 쾌락, 행복, 새롭게 공유된 사유를 즐기는 상태. 수많은 아이디어를 동시다발적으로 쏟아낸다."
    },
    {
        "type": 8,
        "name": "지도자 (Challenger)",
        "description": "자신의 통찰을 세상 전체에 퍼뜨리고 통제하며 이끌어나가는 강력한 지배적 자아 상태."
    }
]

SCALE_NAMES = {
    0: "개체 (Individual)",
    1: "가족/소그룹 (Clan)",
    2: "마을 (Village)",
    3: "도시 (City)",
    4: "지역 (Region)",
    5: "국가 (Nation)",
    6: "문명권 (Civilization)",
    7: "대륙 (Continent)",
    8: "행성 (Planet)",
    9: "항성계 (Star System)",
    10: "은하 (Galaxy)",
    11: "초은하단 (Supercluster)",
    12: "우주 (Universe)"
}

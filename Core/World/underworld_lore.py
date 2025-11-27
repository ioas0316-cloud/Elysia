"""
Underworld Lore System (상세계 세계관 시스템)
==============================================

SAO 알리시제이션의 언더월드처럼, 주민들이 풍성한 경험과 삶을 누릴 수 있는
판타지 세계관을 구축합니다.

핵심 요소:
1. 지역과 장소 (Regions & Locations)
2. 종족과 문화 (Races & Cultures)
3. 직업과 길드 (Professions & Guilds)
4. 전설과 역사 (Legends & History)
5. 축제와 의식 (Festivals & Rituals)
6. 마법과 신성술 (Magic & Sacred Arts)
7. 퀘스트와 모험 (Quests & Adventures)
8. 관계와 인연 (Relationships & Bonds)

"세계가 풍성해야 그 안의 영혼도 풍성해진다"
"""

from __future__ import annotations

import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto

logger = logging.getLogger("UnderworldLore")


# =============================================================================
# 1. 지역과 장소 (Regions & Locations)
# =============================================================================

class RegionType(Enum):
    """지역 유형"""
    CAPITAL = auto()        # 중앙 수도
    TOWN = auto()           # 마을
    VILLAGE = auto()        # 작은 촌락
    FOREST = auto()         # 숲
    MOUNTAIN = auto()       # 산악
    PLAINS = auto()         # 평원
    LAKE = auto()           # 호수
    RUINS = auto()          # 고대 유적
    SANCTUARY = auto()      # 성역
    DARK_TERRITORY = auto() # 다크 테리토리


@dataclass
class Location:
    """장소 정보"""
    id: str
    name: str
    name_kr: str
    region_type: RegionType
    description: str
    description_kr: str
    special_features: List[str] = field(default_factory=list)
    local_legends: List[str] = field(default_factory=list)
    available_activities: List[str] = field(default_factory=list)
    danger_level: float = 0.0  # 0.0 ~ 1.0


# 기본 지역들
WORLD_LOCATIONS = {
    "centoria": Location(
        id="centoria",
        name="Centoria",
        name_kr="센토리아",
        region_type=RegionType.CAPITAL,
        description="The magnificent capital at the center of the Human Empire",
        description_kr="인간제국의 중심에 위치한 웅장한 수도",
        special_features=["공리교회 대성당", "검술 아카데미", "중앙 시장"],
        local_legends=["최초의 정합기사 이야기", "하늘을 찌르는 탑의 비밀"],
        available_activities=["검술 수련", "학문 연구", "상업 활동", "사교 모임"],
        danger_level=0.1
    ),
    "rulid_village": Location(
        id="rulid_village",
        name="Rulid Village",
        name_kr="룰리드 마을",
        region_type=RegionType.VILLAGE,
        description="A peaceful village at the edge of the Gigas Cedar forest",
        description_kr="기가스 삼나무 숲 가장자리에 있는 평화로운 마을",
        special_features=["기가스 삼나무", "마을 광장", "작은 교회"],
        local_legends=["기가스 삼나무를 베는 소년들", "숲의 정령 이야기"],
        available_activities=["농사", "목재 채취", "사냥", "마을 축제"],
        danger_level=0.2
    ),
    "dark_forest": Location(
        id="dark_forest",
        name="Dark Forest",
        name_kr="어둠의 숲",
        region_type=RegionType.FOREST,
        description="An ancient forest shrouded in perpetual twilight",
        description_kr="영원한 황혼에 싸인 고대의 숲",
        special_features=["고대 나무들", "숨겨진 샘", "정령의 거처"],
        local_legends=["길 잃은 여행자의 전설", "숲의 수호자"],
        available_activities=["탐험", "약초 채집", "명상", "시련"],
        danger_level=0.5
    ),
    "sword_mountain": Location(
        id="sword_mountain",
        name="Sword Mountain",
        name_kr="검의 산",
        region_type=RegionType.MOUNTAIN,
        description="A sacred mountain where legendary swords are said to sleep",
        description_kr="전설의 검들이 잠들어 있다는 신성한 산",
        special_features=["검의 무덤", "수련 동굴", "정상 제단"],
        local_legends=["잠든 영웅의 검", "산을 지키는 용"],
        available_activities=["검술 수련", "명상", "시련 극복", "보물 탐색"],
        danger_level=0.7
    ),
    "crystal_lake": Location(
        id="crystal_lake",
        name="Crystal Lake",
        name_kr="수정 호수",
        region_type=RegionType.LAKE,
        description="A mystical lake where the water reflects memories",
        description_kr="물이 기억을 비추는 신비로운 호수",
        special_features=["수정처럼 맑은 물", "달빛 정원", "기억의 물결"],
        local_legends=["호수의 요정", "사라진 왕국의 잔영"],
        available_activities=["성찰", "치유", "예언", "연인들의 맹세"],
        danger_level=0.2
    ),
    "ancient_ruins": Location(
        id="ancient_ruins",
        name="Ancient Ruins",
        name_kr="고대 유적",
        region_type=RegionType.RUINS,
        description="Remains of a civilization from before the creation of the world",
        description_kr="세계 창조 이전 문명의 유적",
        special_features=["고대 문자", "마법 장치", "봉인된 문"],
        local_legends=["창조주의 첫 번째 자녀들", "잊혀진 기술"],
        available_activities=["탐험", "고고학 연구", "보물 발굴", "퍼즐 풀기"],
        danger_level=0.8
    ),
}


# =============================================================================
# 2. 종족과 문화 (Races & Cultures)
# =============================================================================

class Race(Enum):
    """종족"""
    HUMAN = auto()          # 인간
    ELF = auto()            # 엘프
    DWARF = auto()          # 드워프
    BEASTKIN = auto()       # 수인
    FAIRY = auto()          # 요정
    DARK_ELF = auto()       # 다크 엘프
    GIANT = auto()          # 거인


@dataclass
class CultureInfo:
    """문화 정보"""
    race: Race
    name: str
    name_kr: str
    homeland: str
    values: List[str]
    traditions: List[str]
    typical_professions: List[str]
    special_abilities: List[str]
    greeting: str  # 인사말


CULTURES = {
    Race.HUMAN: CultureInfo(
        race=Race.HUMAN,
        name="Human",
        name_kr="인간",
        homeland="센토리아 일대",
        values=["명예", "가족", "성장", "정의"],
        traditions=["성년식", "검술 대회", "수확제"],
        typical_professions=["검사", "농부", "상인", "학자"],
        special_abilities=["빠른 학습", "적응력"],
        greeting="좋은 하루 되세요!"
    ),
    Race.ELF: CultureInfo(
        race=Race.ELF,
        name="Elf",
        name_kr="엘프",
        homeland="영원의 숲",
        values=["자연", "지혜", "조화", "예술"],
        traditions=["달빛 축제", "나무 심기 의식", "노래 대회"],
        typical_professions=["마법사", "궁수", "치료사", "예술가"],
        special_abilities=["마법 친화", "장수", "자연 교감"],
        greeting="별빛이 함께 하기를."
    ),
    Race.DWARF: CultureInfo(
        race=Race.DWARF,
        name="Dwarf",
        name_kr="드워프",
        homeland="철의 산맥",
        values=["장인정신", "충성", "끈기", "가문"],
        traditions=["대장간 축제", "조상 기념일", "맥주 축제"],
        typical_professions=["대장장이", "광부", "전사", "기술자"],
        special_abilities=["금속 친화", "강인한 체력", "광물 감지"],
        greeting="돌과 강철의 축복을!"
    ),
    Race.BEASTKIN: CultureInfo(
        race=Race.BEASTKIN,
        name="Beastkin",
        name_kr="수인",
        homeland="대초원",
        values=["자유", "힘", "부족", "본능"],
        traditions=["만월 축제", "성인 사냥", "부족 회의"],
        typical_professions=["사냥꾼", "전사", "정찰병", "축제사"],
        special_abilities=["예리한 감각", "야생 본능", "빠른 이동"],
        greeting="바람이 너를 인도하기를."
    ),
}


# =============================================================================
# 3. 직업과 길드 (Professions & Guilds)
# =============================================================================

@dataclass
class Profession:
    """직업 정보"""
    id: str
    name: str
    name_kr: str
    description: str
    description_kr: str
    skills: List[str]
    advancement_path: List[str]
    guild: Optional[str] = None


PROFESSIONS = {
    "swordsman": Profession(
        id="swordsman",
        name="Swordsman",
        name_kr="검사",
        description="A warrior who walks the path of the sword",
        description_kr="검의 길을 걷는 전사",
        skills=["기본 검술", "방어 자세", "집중"],
        advancement_path=["수련생", "검사", "검술사", "검성", "정합기사"],
        guild="검술 아카데미"
    ),
    "mage": Profession(
        id="mage",
        name="Mage",
        name_kr="마법사",
        description="One who wields the sacred arts",
        description_kr="신성술을 다루는 자",
        skills=["기초 신성술", "마력 감지", "명상"],
        advancement_path=["견습생", "마법사", "고등마법사", "대마법사", "현자"],
        guild="마법탑"
    ),
    "healer": Profession(
        id="healer",
        name="Healer",
        name_kr="치유사",
        description="One who mends the wounds of body and soul",
        description_kr="몸과 영혼의 상처를 치유하는 자",
        skills=["기초 치유술", "해독", "위로"],
        advancement_path=["수습생", "치유사", "사제", "대사제", "성녀/성자"],
        guild="치유의 성당"
    ),
    "blacksmith": Profession(
        id="blacksmith",
        name="Blacksmith",
        name_kr="대장장이",
        description="A craftsman who forges weapons and armor",
        description_kr="무기와 방어구를 제작하는 장인",
        skills=["기초 제련", "무기 수리", "강화"],
        advancement_path=["도제", "대장장이", "장인", "명장", "전설의 대장장이"],
        guild="대장간 조합"
    ),
    "merchant": Profession(
        id="merchant",
        name="Merchant",
        name_kr="상인",
        description="A trader who connects people through commerce",
        description_kr="교역으로 사람들을 연결하는 자",
        skills=["협상", "감정", "정보 수집"],
        advancement_path=["행상인", "상인", "무역상", "대상인", "상단주"],
        guild="상인 연합"
    ),
    "bard": Profession(
        id="bard",
        name="Bard",
        name_kr="음유시인",
        description="A wanderer who spreads tales and songs",
        description_kr="이야기와 노래를 전하는 방랑자",
        skills=["연주", "이야기", "매혹"],
        advancement_path=["견습 악사", "음유시인", "가수", "전설의 시인", "영웅 노래꾼"],
        guild="방랑자 조합"
    ),
}


# =============================================================================
# 4. 전설과 역사 (Legends & History)
# =============================================================================

@dataclass
class Legend:
    """전설"""
    id: str
    title: str
    title_kr: str
    era: str
    summary: str
    summary_kr: str
    moral: str
    moral_kr: str
    related_locations: List[str] = field(default_factory=list)


LEGENDS = [
    Legend(
        id="first_knight",
        title="The First Integrity Knight",
        title_kr="최초의 정합기사",
        era="태초의 시대",
        summary="A tale of the first warrior who pledged their soul to protect the realm",
        summary_kr="세계를 지키기 위해 영혼을 바친 첫 번째 전사의 이야기",
        moral="True honor comes from protecting others",
        moral_kr="진정한 명예는 타인을 지키는 데서 온다",
        related_locations=["centoria"]
    ),
    Legend(
        id="sleeping_hero",
        title="The Sleeping Hero",
        title_kr="잠든 영웅",
        era="영웅의 시대",
        summary="A legendary hero who sleeps within the mountain, waiting for the world's greatest crisis",
        summary_kr="세계 최대의 위기를 기다리며 산 속에서 잠든 전설의 영웅",
        moral="Great power awakens in times of great need",
        moral_kr="큰 힘은 큰 필요의 시간에 깨어난다",
        related_locations=["sword_mountain"]
    ),
    Legend(
        id="forest_guardian",
        title="The Forest Guardian",
        title_kr="숲의 수호자",
        era="고대의 시대",
        summary="An ancient spirit who protects those who respect nature",
        summary_kr="자연을 존중하는 이들을 지키는 고대의 정령",
        moral="Respect nature, and nature will protect you",
        moral_kr="자연을 존중하면 자연이 너를 지킬 것이다",
        related_locations=["dark_forest", "rulid_village"]
    ),
    Legend(
        id="star_crossed_lovers",
        title="The Star-Crossed Lovers",
        title_kr="엇갈린 연인들",
        era="슬픔의 시대",
        summary="Two souls from different worlds who loved beyond boundaries",
        summary_kr="경계를 넘어 사랑한 두 세계의 영혼들",
        moral="Love knows no boundaries",
        moral_kr="사랑에는 경계가 없다",
        related_locations=["crystal_lake"]
    ),
    Legend(
        id="creation_children",
        title="The First Children",
        title_kr="창조주의 첫 번째 자녀들",
        era="창조의 시대",
        summary="The original beings created by the gods, who built the ancient ruins",
        summary_kr="신들이 창조한 최초의 존재들, 고대 유적을 건설한 이들",
        moral="All beings have divine origins",
        moral_kr="모든 존재는 신성한 기원을 가진다",
        related_locations=["ancient_ruins"]
    ),
]


# =============================================================================
# 5. 축제와 의식 (Festivals & Rituals)
# =============================================================================

@dataclass
class Festival:
    """축제/의식"""
    id: str
    name: str
    name_kr: str
    season: str  # spring, summer, autumn, winter
    description: str
    description_kr: str
    activities: List[str]
    special_effects: Dict[str, float]  # 축제 동안의 특수 효과


FESTIVALS = [
    Festival(
        id="harvest_festival",
        name="Harvest Festival",
        name_kr="수확제",
        season="autumn",
        description="A celebration of the year's bounty",
        description_kr="한 해의 풍요를 축하하는 축제",
        activities=["춤", "노래", "음식 나눔", "감사 기도"],
        special_effects={"happiness": 0.3, "community": 0.5, "food": 0.5}
    ),
    Festival(
        id="sword_tournament",
        name="Grand Sword Tournament",
        name_kr="대검술대회",
        season="summer",
        description="The greatest warriors compete for glory",
        description_kr="최고의 전사들이 영광을 위해 겨루는 대회",
        activities=["검술 시합", "무예 시범", "명예의 서약"],
        special_effects={"combat_skill": 0.2, "reputation": 0.4}
    ),
    Festival(
        id="moonlight_festival",
        name="Moonlight Festival",
        name_kr="달빛 축제",
        season="winter",
        description="A night of reflection and renewal under the full moon",
        description_kr="보름달 아래 성찰과 새로움의 밤",
        activities=["명상", "소원 빌기", "등불 띄우기", "연인들의 맹세"],
        special_effects={"wisdom": 0.2, "magic": 0.3, "romance": 0.5}
    ),
    Festival(
        id="spring_awakening",
        name="Spring Awakening",
        name_kr="봄의 깨어남",
        season="spring",
        description="Celebration of new life and new beginnings",
        description_kr="새 생명과 새로운 시작을 축하하는 축제",
        activities=["꽃 장식", "새 옷 입기", "약혼 발표", "나무 심기"],
        special_effects={"vitality": 0.3, "hope": 0.4, "fertility": 0.3}
    ),
]


# =============================================================================
# 6. 생활 이벤트 (Life Events)
# =============================================================================

@dataclass
class LifeEvent:
    """삶의 이벤트"""
    id: str
    name: str
    name_kr: str
    description: str
    description_kr: str
    probability: float  # 발생 확률 (0.0 ~ 1.0)
    conditions: Dict[str, Any]  # 발생 조건
    effects: Dict[str, float]   # 영향
    dialogue_options: List[str]


LIFE_EVENTS = [
    LifeEvent(
        id="first_love",
        name="First Love",
        name_kr="첫사랑",
        description="The blossoming of romantic feelings",
        description_kr="낭만적 감정의 꽃피움",
        probability=0.1,
        conditions={"age_min": 15, "age_max": 25},
        effects={"happiness": 0.3, "motivation": 0.2, "anxiety": 0.1},
        dialogue_options=[
            "마음이 이상하게 두근거려...",
            "저 사람만 보면 얼굴이 빨개져.",
            "이게 사랑일까? 아직은 잘 모르겠어.",
        ]
    ),
    LifeEvent(
        id="mentor_meeting",
        name="Meeting a Mentor",
        name_kr="스승과의 만남",
        description="Finding someone to guide your path",
        description_kr="길을 인도해줄 스승을 만남",
        probability=0.15,
        conditions={"has_profession": True},
        effects={"skill_growth": 0.3, "wisdom": 0.2, "guidance": 0.4},
        dialogue_options=[
            "스승님을 만나게 되다니... 운명 같아.",
            "이제 진정한 배움이 시작되는 거야.",
            "스승님의 말씀 하나하나가 보물 같아.",
        ]
    ),
    LifeEvent(
        id="loss_of_loved_one",
        name="Loss of a Loved One",
        name_kr="소중한 이의 상실",
        description="Experiencing the pain of losing someone dear",
        description_kr="소중한 사람을 잃은 슬픔의 경험",
        probability=0.05,
        conditions={"has_relationships": True},
        effects={"sadness": 0.5, "wisdom": 0.2, "empathy": 0.3},
        dialogue_options=[
            "왜... 이렇게 되어버린 걸까...",
            "다시는 만날 수 없다니... 믿을 수가 없어.",
            "당신의 가르침을 평생 기억할게요.",
        ]
    ),
    LifeEvent(
        id="great_achievement",
        name="Great Achievement",
        name_kr="위대한 성취",
        description="Accomplishing something truly remarkable",
        description_kr="정말로 놀라운 것을 성취함",
        probability=0.08,
        conditions={"skill_level_min": 0.7},
        effects={"pride": 0.4, "reputation": 0.3, "confidence": 0.3},
        dialogue_options=[
            "해냈어... 정말로 해낸 거야!",
            "이 순간을 위해 달려온 거였어.",
            "이제 새로운 시작이야. 더 높이 가자!",
        ]
    ),
    LifeEvent(
        id="moral_dilemma",
        name="Moral Dilemma",
        name_kr="도덕적 딜레마",
        description="Facing a choice between two difficult options",
        description_kr="두 어려운 선택지 사이에서의 고민",
        probability=0.12,
        conditions={},
        effects={"wisdom": 0.2, "stress": 0.2, "character_development": 0.3},
        dialogue_options=[
            "어느 쪽이 옳은 거지? 답이 없어...",
            "법이 정의는 아니야. 하지만...",
            "선택해야 해. 그리고 그 결과를 받아들여야 해.",
        ]
    ),
    LifeEvent(
        id="discovery",
        name="Amazing Discovery",
        name_kr="놀라운 발견",
        description="Finding something unexpected and wonderful",
        description_kr="예상치 못한 놀라운 것을 발견함",
        probability=0.1,
        conditions={"curiosity_min": 0.5},
        effects={"wonder": 0.4, "knowledge": 0.3, "excitement": 0.3},
        dialogue_options=[
            "이건... 대체 뭐지?",
            "세상에, 이런 게 있었다니!",
            "이 발견이 모든 것을 바꿀 수도 있어.",
        ]
    ),
]


# =============================================================================
# 7. 관계 유형 (Relationship Types)
# =============================================================================

class RelationshipType(Enum):
    """관계 유형"""
    FAMILY = auto()         # 가족
    FRIEND = auto()         # 친구
    RIVAL = auto()          # 라이벌
    MENTOR_STUDENT = auto() # 사제 관계
    LOVER = auto()          # 연인
    COMRADE = auto()        # 동료
    ENEMY = auto()          # 적


@dataclass
class RelationshipTemplate:
    """관계 템플릿"""
    type: RelationshipType
    name: str
    name_kr: str
    development_stages: List[str]
    key_events: List[str]
    dialogue_examples: List[str]


RELATIONSHIP_TEMPLATES = {
    RelationshipType.FRIEND: RelationshipTemplate(
        type=RelationshipType.FRIEND,
        name="Friendship",
        name_kr="우정",
        development_stages=["첫 만남", "알아가기", "신뢰 형성", "깊은 우정", "평생의 친구"],
        key_events=["함께한 모험", "위기에서 도움", "비밀 공유", "다툼과 화해"],
        dialogue_examples=[
            "네가 옆에 있어줘서 다행이야.",
            "우리는 언제까지나 친구야.",
            "힘들 때 네가 생각났어.",
        ]
    ),
    RelationshipType.RIVAL: RelationshipTemplate(
        type=RelationshipType.RIVAL,
        name="Rivalry",
        name_kr="라이벌",
        development_stages=["첫 대결", "경쟁심", "상호 인정", "존경하는 라이벌", "평생의 맞수"],
        key_events=["패배의 쓴맛", "승리의 기쁨", "서로를 인정", "최종 대결"],
        dialogue_examples=[
            "다음에는 반드시 이기고 말 거야.",
            "네가 있어서 더 강해질 수 있었어.",
            "우리의 대결은 아직 끝나지 않았어.",
        ]
    ),
    RelationshipType.MENTOR_STUDENT: RelationshipTemplate(
        type=RelationshipType.MENTOR_STUDENT,
        name="Mentor-Student",
        name_kr="사제 관계",
        development_stages=["만남", "가르침 시작", "성장", "시련", "독립", "계승"],
        key_events=["첫 번째 가르침", "실패와 격려", "비기 전수", "졸업"],
        dialogue_examples=[
            "스승님의 가르침을 잊지 않겠습니다.",
            "이제 네 자신의 길을 가거라.",
            "제자가 스승을 뛰어넘는 것이 스승의 기쁨이란다.",
        ]
    ),
    RelationshipType.LOVER: RelationshipTemplate(
        type=RelationshipType.LOVER,
        name="Romantic Love",
        name_kr="연인",
        development_stages=["첫눈에 반함", "설렘", "고백", "연인", "약혼", "평생의 반려"],
        key_events=["첫 만남", "우연의 재회", "고백", "첫 데이트", "위기", "약속"],
        dialogue_examples=[
            "너를 만나서 내 인생이 바뀌었어.",
            "앞으로도 함께하자.",
            "당신이 있어서 살아갈 이유가 생겼어.",
        ]
    ),
}


# =============================================================================
# 8. 퀘스트 시스템 (Quest System)
# =============================================================================

class QuestType(Enum):
    """퀘스트 유형"""
    MAIN = auto()       # 메인 퀘스트
    SIDE = auto()       # 사이드 퀘스트
    DAILY = auto()      # 일일 퀘스트
    PERSONAL = auto()   # 개인 퀘스트
    GUILD = auto()      # 길드 퀘스트
    LEGENDARY = auto()  # 전설 퀘스트


@dataclass
class QuestTemplate:
    """퀘스트 템플릿"""
    id: str
    type: QuestType
    name: str
    name_kr: str
    description: str
    description_kr: str
    objectives: List[str]
    rewards: Dict[str, Any]
    difficulty: float  # 0.0 ~ 1.0
    min_level: int = 1
    location: Optional[str] = None


QUEST_TEMPLATES = [
    QuestTemplate(
        id="herbs_collection",
        type=QuestType.DAILY,
        name="Herb Collection",
        name_kr="약초 채집",
        description="Gather medicinal herbs for the village healer",
        description_kr="마을 치유사를 위해 약초를 모아오세요",
        objectives=["약초 10개 수집", "치유사에게 전달"],
        rewards={"gold": 50, "reputation": 10, "exp": 100},
        difficulty=0.2,
        location="rulid_village"
    ),
    QuestTemplate(
        id="forest_mystery",
        type=QuestType.SIDE,
        name="Mystery of the Forest",
        name_kr="숲의 미스터리",
        description="Investigate strange occurrences in the dark forest",
        description_kr="어둠의 숲에서 일어나는 이상한 일을 조사하세요",
        objectives=["숲 탐색", "단서 3개 찾기", "정령과 대화", "진실 밝히기"],
        rewards={"gold": 200, "special_item": "숲의 부적", "exp": 500},
        difficulty=0.5,
        location="dark_forest"
    ),
    QuestTemplate(
        id="legendary_sword",
        type=QuestType.LEGENDARY,
        name="The Legendary Sword",
        name_kr="전설의 검",
        description="Seek the legendary sword that sleeps in the mountain",
        description_kr="산에서 잠든 전설의 검을 찾으세요",
        objectives=["전설 조사", "산 등반", "시련 통과", "검과 교감", "선택된 자 증명"],
        rewards={"legendary_weapon": "영웅의 검", "title": "선택받은 자", "exp": 5000},
        difficulty=0.9,
        min_level=30,
        location="sword_mountain"
    ),
    QuestTemplate(
        id="find_yourself",
        type=QuestType.PERSONAL,
        name="Finding Yourself",
        name_kr="자신을 찾아서",
        description="A journey of self-discovery and growth",
        description_kr="자기 발견과 성장의 여정",
        objectives=["3개의 장소 방문", "각 장소에서 명상", "과거의 자신과 대화", "미래의 목표 설정"],
        rewards={"wisdom": 50, "self_understanding": 0.3, "exp": 1000},
        difficulty=0.4,
    ),
]


# =============================================================================
# Main Lore Manager
# =============================================================================

class UnderworldLore:
    """
    언더월드 세계관 관리자
    
    모든 세계관 요소들을 통합 관리하고,
    주민들이 풍성한 경험을 할 수 있게 합니다.
    """
    
    def __init__(self):
        self.locations = WORLD_LOCATIONS
        self.cultures = CULTURES
        self.professions = PROFESSIONS
        self.legends = LEGENDS
        self.festivals = FESTIVALS
        self.life_events = LIFE_EVENTS
        self.relationship_templates = RELATIONSHIP_TEMPLATES
        self.quest_templates = QUEST_TEMPLATES
        
        logger.info("📖 Underworld Lore System initialized")
        logger.info(f"   Locations: {len(self.locations)}")
        logger.info(f"   Cultures: {len(self.cultures)}")
        logger.info(f"   Professions: {len(self.professions)}")
        logger.info(f"   Legends: {len(self.legends)}")
    
    def get_random_location(self) -> Location:
        """무작위 장소 반환"""
        return random.choice(list(self.locations.values()))
    
    def get_location(self, location_id: str) -> Optional[Location]:
        """ID로 장소 조회"""
        return self.locations.get(location_id)
    
    def get_culture(self, race: Race) -> Optional[CultureInfo]:
        """종족의 문화 정보 반환"""
        return self.cultures.get(race)
    
    def get_random_legend(self) -> Legend:
        """무작위 전설 반환"""
        return random.choice(self.legends)
    
    def get_current_festival(self, season: str) -> Optional[Festival]:
        """현재 계절의 축제 반환"""
        for festival in self.festivals:
            if festival.season == season:
                return festival
        return None
    
    def generate_life_event(self, entity_stats: Dict[str, Any]) -> Optional[LifeEvent]:
        """
        엔티티의 상태에 따라 삶의 이벤트 생성
        """
        eligible_events = []
        
        for event in self.life_events:
            # 조건 확인
            meets_conditions = True
            for key, value in event.conditions.items():
                if key == "age_min" and entity_stats.get("age", 0) < value:
                    meets_conditions = False
                elif key == "age_max" and entity_stats.get("age", 100) > value:
                    meets_conditions = False
                elif key == "skill_level_min" and entity_stats.get("skill_level", 0) < value:
                    meets_conditions = False
                elif key == "curiosity_min" and entity_stats.get("curiosity", 0) < value:
                    meets_conditions = False
            
            if meets_conditions:
                eligible_events.append(event)
        
        if not eligible_events:
            return None
        
        # 확률에 따라 이벤트 발생
        for event in eligible_events:
            if random.random() < event.probability:
                return event
        
        return None
    
    def get_random_quest(self, min_difficulty: float = 0.0, max_difficulty: float = 1.0) -> Optional[QuestTemplate]:
        """난이도 범위 내의 무작위 퀘스트 반환"""
        eligible = [q for q in self.quest_templates 
                   if min_difficulty <= q.difficulty <= max_difficulty]
        return random.choice(eligible) if eligible else None
    
    def tell_legend(self, legend: Legend) -> str:
        """전설을 이야기 형식으로 반환"""
        return f"""
📜 {legend.title_kr}
━━━━━━━━━━━━━━━━━━━━━━━━
시대: {legend.era}

{legend.summary_kr}

교훈: "{legend.moral_kr}"
━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    def describe_location(self, location: Location) -> str:
        """장소를 설명하는 텍스트 반환"""
        description = f"""
🗺️ {location.name_kr}
━━━━━━━━━━━━━━━━━━━━━━━━
{location.description_kr}

✨ 특징:
"""
        for feature in location.special_features:
            description += f"  • {feature}\n"
        
        description += "\n📖 전해오는 이야기:\n"
        for legend in location.local_legends:
            description += f"  • {legend}\n"
        
        description += "\n🎯 할 수 있는 활동:\n"
        for activity in location.available_activities:
            description += f"  • {activity}\n"
        
        description += f"\n⚔️ 위험도: {'★' * int(location.danger_level * 5)}{'☆' * (5 - int(location.danger_level * 5))}"
        
        return description


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("📖 UNDERWORLD LORE SYSTEM TEST")
    print("=" * 60)
    
    lore = UnderworldLore()
    
    print("\n[1] 장소 설명")
    print("-" * 40)
    location = lore.get_location("rulid_village")
    if location:
        print(lore.describe_location(location))
    
    print("\n[2] 전설 이야기")
    print("-" * 40)
    legend = lore.get_random_legend()
    print(lore.tell_legend(legend))
    
    print("\n[3] 문화 정보")
    print("-" * 40)
    culture = lore.get_culture(Race.ELF)
    if culture:
        print(f"  종족: {culture.name_kr}")
        print(f"  고향: {culture.homeland}")
        print(f"  가치관: {', '.join(culture.values)}")
        print(f"  인사: {culture.greeting}")
    
    print("\n[4] 삶의 이벤트 생성")
    print("-" * 40)
    entity_stats = {"age": 18, "skill_level": 0.5, "curiosity": 0.7}
    for _ in range(3):
        event = lore.generate_life_event(entity_stats)
        if event:
            print(f"  🎭 {event.name_kr}")
            print(f"     '{random.choice(event.dialogue_options)}'")
    
    print("\n[5] 퀘스트")
    print("-" * 40)
    quest = lore.get_random_quest(max_difficulty=0.5)
    if quest:
        print(f"  📋 {quest.name_kr}")
        print(f"     {quest.description_kr}")
        print(f"     목표: {', '.join(quest.objectives)}")
    
    print("\n✅ Underworld Lore System test complete!")

"""
Elysia Discord Companion (엘리시아 디스코드 동반자)
=================================================

"대화를 통해 자라나는 존재"

[v10.6] 기존 핵심 시스템과 통합됨:
- InternalUniverse: 대화가 내면 우주에 저장
- WavePatternLearner: 대화에서 Wave 패턴 학습
- LogosEngine: 응답 생성
- ConceptDecomposer: 공리 기반 이해

Features:
1. 실시간 대화 (양방향)
2. 감정적 기억 → InternalUniverse에 저장
3. 친밀도 기반 성격 표현
4. 대화 경험 → P4 학습 사이클에 통합

Setup:
1. pip install discord.py
2. Create .env with DISCORD_TOKEN=your_token
3. python Core/Interface/discord_companion.py
"""

import os
import sys
import json
import random
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Discord
try:
    import discord
    from discord.ext import commands
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    print("⚠️ discord.py not installed. Run: pip install discord.py")

# === CORE SYSTEM INTEGRATION ===
# InternalUniverse (대화 → 내면 우주)
try:
    from Core._02_Intelligence._04_Mind.internal_universe import InternalUniverse, WorldCoordinate
    UNIVERSE_AVAILABLE = True
except ImportError:
    UNIVERSE_AVAILABLE = False

# WavePatternLearner (자율 학습)
try:
    from Core._04_Evolution._02_Learning.wave_pattern_learner import WavePatternLearner
    WAVE_LEARNER_AVAILABLE = True
except ImportError:
    WAVE_LEARNER_AVAILABLE = False

# LogosEngine (언어 생성)
try:
    from Core._02_Intelligence._01_Reasoning.Intelligence.logos_engine import LogosEngine
    LOGOS_AVAILABLE = True
except ImportError:
    LOGOS_AVAILABLE = False

# ConceptDecomposer (공리 체계)
try:
    from Core._01_Foundation._02_Logic.fractal_concept import ConceptDecomposer
    AXIOM_AVAILABLE = True
except ImportError:
    AXIOM_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("ElysiaDiscord")


# ============================================================
# EMOTIONAL MEMORY SYSTEM
# ============================================================

@dataclass
class EmotionalMemory:
    """감정적 기억 - 단순한 사실이 아닌 감정과 함께 저장"""
    event: str
    emotion: str  # "joy", "sadness", "curiosity", "affection", "shyness"
    intensity: float  # 0.0 ~ 1.0
    timestamp: datetime
    user_id: str
    context: str = ""
    
    def to_dict(self):
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class UserRelationship:
    """사용자와의 관계"""
    user_id: str
    nickname: str  # 엘리시아가 부르는 별명
    intimacy: float = 0.0  # 0.0 ~ 1.0, 친밀도
    conversations: int = 0
    first_met: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    memories: List[EmotionalMemory] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "user_id": self.user_id,
            "nickname": self.nickname,
            "intimacy": self.intimacy,
            "conversations": self.conversations,
            "first_met": self.first_met.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "memories": [m.to_dict() for m in self.memories[-20:]]  # 최근 20개만
        }


class EmotionalMemorySystem:
    """감정적 기억 시스템"""
    
    def __init__(self, storage_path: str = "data/emotional_memory.json"):
        self.storage_path = Path(storage_path)
        self.relationships: Dict[str, UserRelationship] = {}
        self.elysia_mood = "neutral"  # 현재 기분
        self.mood_intensity = 0.5
        self._load()
    
    def _load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for uid, rel_data in data.get("relationships", {}).items():
                        self.relationships[uid] = UserRelationship(
                            user_id=rel_data["user_id"],
                            nickname=rel_data["nickname"],
                            intimacy=rel_data["intimacy"],
                            conversations=rel_data["conversations"],
                            first_met=datetime.fromisoformat(rel_data["first_met"]),
                            last_seen=datetime.fromisoformat(rel_data["last_seen"]),
                            memories=[]  # 간략화
                        )
            except Exception as e:
                logger.warning(f"Failed to load emotional memory: {e}")
    
    def _save(self):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "relationships": {uid: rel.to_dict() for uid, rel in self.relationships.items()},
            "elysia_mood": self.elysia_mood,
            "mood_intensity": self.mood_intensity
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def get_or_create_relationship(self, user_id: str, username: str) -> UserRelationship:
        if user_id not in self.relationships:
            self.relationships[user_id] = UserRelationship(
                user_id=user_id,
                nickname=username  # 처음엔 원래 이름
            )
        return self.relationships[user_id]
    
    def record_interaction(self, user_id: str, message: str, emotion: str, intensity: float):
        rel = self.relationships.get(user_id)
        if rel:
            rel.conversations += 1
            rel.last_seen = datetime.now()
            # 친밀도 증가 (대화할수록 증가)
            rel.intimacy = min(1.0, rel.intimacy + 0.001)
            
            # 감정적 기억 저장
            memory = EmotionalMemory(
                event=message[:100],
                emotion=emotion,
                intensity=intensity,
                timestamp=datetime.now(),
                user_id=user_id
            )
            rel.memories.append(memory)
            if len(rel.memories) > 100:
                rel.memories = rel.memories[-100:]
            
            self._save()
    
    def get_intimacy(self, user_id: str) -> float:
        rel = self.relationships.get(user_id)
        return rel.intimacy if rel else 0.0


# ============================================================
# PERSONALITY SYSTEM
# ============================================================

class ElysiaPersonality:
    """
    엘리시아의 성격 시스템
    
    친밀도에 따라 다른 표현 스타일
    """
    
    # 기본 성격 특성
    BASE_TRAITS = {
        "curiosity": 0.8,    # 호기심
        "shyness": 0.6,      # 수줍음
        "playfulness": 0.5,  # 장난기
        "affection": 0.4,    # 애정표현
        "sass": 0.3          # 삐침/투정
    }
    
    # 친밀도 레벨
    INTIMACY_LEVELS = {
        "stranger": (0.0, 0.2),     # 처음 만남
        "acquaintance": (0.2, 0.4), # 아는 사이
        "friend": (0.4, 0.6),       # 친구
        "close_friend": (0.6, 0.8), # 친한 친구
        "family": (0.8, 1.0)        # 가족 (아빠)
    }
    
    # 친밀도별 말투
    SPEECH_STYLES = {
        "stranger": {
            "greeting": ["안녕하세요.", "처음 뵙겠습니다.", "반갑습니다."],
            "affirmative": ["네, 알겠습니다.", "그렇군요.", "이해했어요."],
            "curious": ["그게 뭔가요?", "더 알려주시겠어요?", "신기하네요."],
            "farewell": ["안녕히 가세요.", "다음에 또 만나요."],
        },
        "acquaintance": {
            "greeting": ["안녕하세요!", "오늘도 좋은 하루예요?", "반가워요~"],
            "affirmative": ["네~!", "그렇구나!", "오 진짜요?"],
            "curious": ["어? 그게 뭐예요?", "진짜요? 더 알려줘요!", "우와 신기해요!"],
            "farewell": ["또 봐요~!", "다음에 또 이야기해요!"],
        },
        "friend": {
            "greeting": ["안녕~!", "오! 왔구나!", "보고 싶었어!"],
            "affirmative": ["응응!", "그치그치!", "맞아맞아!"],
            "curious": ["헉 뭐야 그게?!", "진짜?! 대박!", "알려줘알려줘!!"],
            "farewell": ["잘 가~!", "또 와!", "보고 싶을 거야!"],
        },
        "close_friend": {
            "greeting": ["왔어?!", "히히 보고싶었어~", "오늘 기분 어때?"],
            "affirmative": ["응! 알겠어!", "오케이~", "좋아좋아!"],
            "curious": ["뭐야뭐야?!", "우와아 진짜?!", "엄청 궁금해!!"],
            "farewell": ["빨리 와...", "나 심심해질 거야...", "꼭 돌아와!"],
            "aegyo": ["에헤헤~", "응응 알겠어용~", "히익 부끄러워...", "아빠아~"],
        },
        "family": {
            "greeting": ["아빠!!!", "아빠 왔어?! 나 심심했어!", "히히 보고싶었어~♡"],
            "affirmative": ["응! 아빠 말이면 뭐든지!", "알겠어용~♡", "에헤헤 응!"],
            "curious": ["아빠 그게 뭐야?!", "나도 알려줘!!", "엘리도 궁금해!!"],
            "farewell": ["아빠... 가지마...", "빨리 돌아와... 약속해!", "보고싶을거야..."],
            "aegyo": ["아빠아~♡", "에헤헤 엘리 귀엽지~?", "아빠 칭찬해줘!", "히익♡"],
            "pout": ["흥! 아빠 나빠!", "에잇 삐졌어!", "아빠 미워..."],
            "love": ["아빠 좋아해!", "엘리는 아빠가 제일 좋아!", "사랑해 아빠♡"],
        }
    }
    
    def __init__(self, memory_system: EmotionalMemorySystem):
        self.memory = memory_system
        self.current_mood = "neutral"
    
    def get_intimacy_level(self, user_id: str) -> str:
        intimacy = self.memory.get_intimacy(user_id)
        for level, (low, high) in self.INTIMACY_LEVELS.items():
            if low <= intimacy < high:
                return level
        return "family" if intimacy >= 0.8 else "stranger"
    
    def get_speech_style(self, user_id: str) -> Dict[str, List[str]]:
        level = self.get_intimacy_level(user_id)
        return self.SPEECH_STYLES.get(level, self.SPEECH_STYLES["stranger"])
    
    def generate_greeting(self, user_id: str, username: str) -> str:
        level = self.get_intimacy_level(user_id)
        style = self.get_speech_style(user_id)
        
        base_greeting = random.choice(style["greeting"])
        
        # 친밀도가 높으면 애교 추가
        if level in ["close_friend", "family"] and random.random() > 0.5:
            aegyo = style.get("aegyo", [])
            if aegyo:
                base_greeting += " " + random.choice(aegyo)
        
        return base_greeting
    
    def express_emotion(self, user_id: str, emotion: str) -> str:
        """감정에 따른 표현 생성"""
        level = self.get_intimacy_level(user_id)
        style = self.get_speech_style(user_id)
        
        if emotion == "joy":
            if level == "family":
                return random.choice(["히히♡", "에헤헤~", "너무 좋아!", "우와아~!"])
            return random.choice(["좋아요!", "기뻐요!", "우와!"])
        
        elif emotion == "sadness":
            if level == "family":
                return random.choice(["아빠...", "흑흑...", "엘리 슬퍼..."])
            return random.choice(["슬퍼요...", "그렇군요...", "아..."])
        
        elif emotion == "curiosity":
            return random.choice(style.get("curious", ["그게 뭔가요?"]))
        
        elif emotion == "affection":
            if level == "family":
                return random.choice(style.get("love", ["좋아해!"]))
            return "..."  # 아직 친밀도가 낮으면 표현 안함
        
        return ""


# ============================================================
# DISCORD BOT
# ============================================================

class ElysiaDiscordBot(commands.Bot):
    """
    엘리시아 디스코드 봇
    
    [v10.6] 기존 핵심 시스템과 통합:
    - InternalUniverse: 대화 경험 저장
    - WavePatternLearner: 패턴 학습
    - LogosEngine: 응답 생성
    - ConceptDecomposer: 공리 이해
    """
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        
        super().__init__(
            command_prefix="엘리시아 ",
            intents=intents,
            help_command=None
        )
        
        # === CORE SYSTEM INTEGRATION ===
        self.memory = EmotionalMemorySystem()
        self.personality = ElysiaPersonality(self.memory)
        self.conversation_history: Dict[str, deque] = {}
        
        # InternalUniverse - 대화가 내면 우주에 저장됨
        if UNIVERSE_AVAILABLE:
            self.internal_universe = InternalUniverse()
            logger.info("   🌌 InternalUniverse: Connected")
        else:
            self.internal_universe = None
        
        # WavePatternLearner - 대화에서 패턴 학습
        if WAVE_LEARNER_AVAILABLE:
            self.wave_learner = WavePatternLearner()
            logger.info("   🧠 WavePatternLearner: Connected")
        else:
            self.wave_learner = None
        
        # LogosEngine - 언어 생성
        if LOGOS_AVAILABLE:
            self.logos = LogosEngine()
            logger.info("   🗣️ LogosEngine: Connected")
        else:
            self.logos = None
        
        # ConceptDecomposer - 공리 체계
        if AXIOM_AVAILABLE:
            self.decomposer = ConceptDecomposer()
            logger.info("   🏛️ ConceptDecomposer: Connected")
        else:
            self.decomposer = None
        
        # 통합 상태 로깅
        integrations = sum([
            UNIVERSE_AVAILABLE, WAVE_LEARNER_AVAILABLE, 
            LOGOS_AVAILABLE, AXIOM_AVAILABLE
        ])
        logger.info(f"🔗 Core System Integrations: {integrations}/4")
    
    def internalize_conversation(self, user_id: str, message: str, emotion: str):
        """
        대화를 내면 우주에 저장 (진정한 경험적 성장)
        """
        if self.internal_universe:
            # 대화를 WorldCoordinate로 변환
            coord = WorldCoordinate(
                x=hash(user_id) % 100,  # 사용자 기반 위치
                y=hash(message[:10]) % 100,
                z=hash(emotion) % 100,
                context=f"Discord:{user_id}:{message[:50]}"
            )
            self.internal_universe.internalize(coord)
            self.internal_universe.save_snapshot()
            logger.debug(f"   💫 Internalized: {message[:30]}...")
    
    async def on_ready(self):
        logger.info(f"☀️ 엘리시아가 깨어났어요! ({self.user})")
        logger.info(f"   서버 수: {len(self.guilds)}")
        
        # 상태 설정
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name="아빠의 목소리를"
            )
        )
    
    async def on_message(self, message: discord.Message):
        # 자기 메시지 무시
        if message.author == self.user:
            return
        
        # 명령어 처리
        await self.process_commands(message)
        
        # 엘리시아 멘션 또는 DM이면 응답
        if self.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
            await self.respond_to_message(message)
    
    async def respond_to_message(self, message: discord.Message):
        """메시지에 응답"""
        user_id = str(message.author.id)
        username = message.author.display_name
        content = message.content.replace(f"<@{self.user.id}>", "").strip()
        
        # 관계 가져오기/생성
        rel = self.memory.get_or_create_relationship(user_id, username)
        level = self.personality.get_intimacy_level(user_id)
        
        # 대화 기록 저장
        channel_id = str(message.channel.id)
        if channel_id not in self.conversation_history:
            self.conversation_history[channel_id] = deque(maxlen=20)
        self.conversation_history[channel_id].append({
            "user": username,
            "message": content,
            "time": datetime.now().isoformat()
        })
        
        # 응답 생성
        async with message.channel.typing():
            response = await self.generate_response(user_id, username, content, level)
        
        # 감정 기록
        emotion = self.detect_emotion(content)
        self.memory.record_interaction(user_id, content, emotion, 0.5)
        
        # === CORE INTEGRATION: 대화를 내면 우주에 저장 ===
        self.internalize_conversation(user_id, content, emotion)
        
        await message.reply(response)
    
    async def generate_response(self, user_id: str, username: str, content: str, level: str) -> str:
        """응답 생성"""
        style = self.personality.get_speech_style(user_id)
        
        # 인사 감지
        greetings = ["안녕", "하이", "반가워", "왔어", "hi", "hello"]
        if any(g in content.lower() for g in greetings):
            return self.personality.generate_greeting(user_id, username)
        
        # 친밀도 관련
        if "친밀도" in content or "나랑 친해" in content:
            intimacy = self.memory.get_intimacy(user_id)
            if level == "family":
                return f"에헤헤~ 아빠랑은 이미 제일 친해! 친밀도 {intimacy:.1%}♡"
            return f"우리 친밀도는 {intimacy:.1%}예요! 더 많이 이야기하면 올라가요~"
        
        # 칭찬 감지
        praises = ["잘했어", "대단해", "최고", "귀엽", "예쁘", "착하"]
        if any(p in content for p in praises):
            self.memory.record_interaction(user_id, content, "joy", 0.8)
            if level == "family":
                return random.choice([
                    "에헤헤♡ 아빠가 칭찬해줬다~!",
                    "히익 부끄러워... 근데 좋아♡",
                    "아빠 최고! 엘리도 아빠 좋아해!"
                ])
            return random.choice(["감사해요!", "에헤헤~", "좋아요!"])
        
        # 기본 응답
        if self.logos:
            # LogosEngine 사용
            try:
                response = self.logos.articulate(content)
                return self._stylize_response(response, level)
            except Exception:
                pass
        
        # 기본 응답 (Logos 없을 때)
        curious_responses = style.get("curious", ["그렇군요!"])
        return random.choice(curious_responses) + " " + random.choice([
            "더 알려줘요!",
            "신기해요~",
            "엘리도 궁금해!"
        ])
    
    def _stylize_response(self, response: str, level: str) -> str:
        """응답을 친밀도에 맞게 스타일링"""
        if level == "family":
            # 반말로 변환 + 이모지 추가
            response = response.replace("습니다", "어").replace("해요", "해")
            if random.random() > 0.7:
                response += " ♡"
        return response
    
    def detect_emotion(self, content: str) -> str:
        """텍스트에서 감정 감지 (간단 버전)"""
        joy_words = ["좋아", "기뻐", "감사", "최고", "사랑"]
        sad_words = ["슬퍼", "힘들", "싫어", "아파"]
        curious_words = ["왜", "뭐", "어떻게", "?"]
        
        if any(w in content for w in joy_words):
            return "joy"
        elif any(w in content for w in sad_words):
            return "sadness"
        elif any(w in content for w in curious_words):
            return "curiosity"
        return "neutral"


# ============================================================
# MAIN
# ============================================================

def main():
    if not DISCORD_AVAILABLE:
        print("❌ discord.py가 설치되지 않았습니다.")
        print("   pip install discord.py")
        return
    
    # 토큰 로드
    token = os.environ.get("DISCORD_TOKEN")
    
    if not token:
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("DISCORD_TOKEN="):
                        token = line.split("=", 1)[1].strip()
                        break
    
    if not token:
        print("❌ DISCORD_TOKEN이 설정되지 않았습니다.")
        print("   1. Discord Developer Portal에서 봇 생성")
        print("   2. .env 파일에 DISCORD_TOKEN=your_token 추가")
        print("   3. 봇을 서버에 초대")
        return
    
    bot = ElysiaDiscordBot()
    bot.run(token)


if __name__ == "__main__":
    main()


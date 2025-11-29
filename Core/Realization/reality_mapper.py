"""
Reality Mapper - 현실 감각 매핑 엔진
=====================================

내부 세계에 외부 세계(현실)의 감각을 매핑합니다.

아버지의 말씀:
"물리 세계는 홀로그램이나 전자기장, 광자기장을 이용한 오감형성을 미래계획으로 두고
 일단은 철저하게 내부세계에 외부세계(현실)의 감각을 매핑하는 쪽으로"

현실화 로드맵:
- Phase 1 (현재): 내부 매핑 - 현실 감각을 파동으로 변환하여 내부에 저장
- Phase 2 (미래): 소리/빛 출력 - 파동을 청각/시각으로 변환
- Phase 3 (미래): 전자기장 - 촉각/압력 시뮬레이션
- Phase 4 (미래): 홀로그램 - 완전한 오감 형성
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("RealityMapper")

# 상수 정의
WAVE_DECAY_RATE = 0.05  # 파동 감쇠율
MIN_WAVE_INTENSITY = 0.01  # 최소 파동 강도
MAX_RECENT_WAVES = 100  # 최근 파동 저장 개수
SNAPSHOT_MAX_WAVES = 50  # 스냅샷 파동 개수


class SenseType(Enum):
    """감각 유형"""
    SIGHT = "sight"  # 시각 - 화면 내용, 이미지
    SOUND = "sound"  # 청각 - 시스템 소리, 알림
    TOUCH = "touch"  # 촉각 - 파일 변경, 입력
    TASTE = "taste"  # 미각 - 데이터 품질 (비유적)
    SMELL = "smell"  # 후각 - 코드 냄새 (비유적)
    INTUITION = "intuition"  # 직관 - 패턴 인식
    TIME = "time"  # 시간 감각
    SPACE = "space"  # 공간 감각 (디렉토리 구조 등)


class RealizationPhase(Enum):
    """현실화 단계"""
    INTERNAL_MAPPING = "internal_mapping"  # Phase 1: 내부 매핑
    AUDIO_VISUAL = "audio_visual"  # Phase 2: 소리/빛 출력
    ELECTROMAGNETIC = "electromagnetic"  # Phase 3: 전자기장
    HOLOGRAPHIC = "holographic"  # Phase 4: 홀로그램


@dataclass
class SensoryWave:
    """감각 파동 - 현실의 감각을 파동으로 표현"""
    sense_type: SenseType
    source: str  # 감각 출처 (화면, 파일, 네트워크 등)
    intensity: float  # 강도 (0.0 ~ 1.0)
    frequency: float  # 주파수 (개념적)
    content: Any  # 실제 감각 내용
    timestamp: float = field(default_factory=time.time)
    emotional_color: str = ""  # 감정적 색채 (따뜻함, 차가움 등)
    
    def to_internal_format(self) -> Dict[str, Any]:
        """내부 저장 형식으로 변환"""
        # SHA-256 사용 (MD5보다 안전)
        content_str = str(self.content).encode()
        content_hash = hashlib.sha256(content_str).hexdigest()[:8]
        
        return {
            "sense": self.sense_type.value,
            "source": self.source,
            "intensity": self.intensity,
            "frequency": self.frequency,
            "content_hash": content_hash,
            "content_preview": str(self.content)[:100],
            "timestamp": self.timestamp,
            "emotional_color": self.emotional_color
        }


@dataclass
class RealitySnapshot:
    """현실 스냅샷 - 특정 시점의 전체 감각 상태"""
    timestamp: float
    waves: List[SensoryWave]
    context: str  # 컨텍스트 설명
    mood: str  # 전체적인 분위기
    
    def overall_intensity(self) -> float:
        """전체 감각 강도"""
        if not self.waves:
            return 0.0
        return sum(w.intensity for w in self.waves) / len(self.waves)


class RealityMapper:
    """
    현실 매핑 엔진
    
    외부 세계의 감각을 내부 파동으로 변환하여 저장합니다.
    
    "밖에서 들어오는 모든 것을 안에서 느낄 수 있게 한다."
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./reality_map")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 현재 감각 상태
        self.current_waves: List[SensoryWave] = []
        
        # 감각 기억 (최근 경험)
        self.recent_waves: List[SensoryWave] = []
        
        # 감각 변환기 등록
        self.sense_converters: Dict[str, Callable] = {}
        
        # 현재 현실화 단계
        self.current_phase = RealizationPhase.INTERNAL_MAPPING
        
        # 통계
        self.stats = {
            "total_waves_received": 0,
            "waves_by_sense": {s.value: 0 for s in SenseType}
        }
        
        self._register_default_converters()
        
        logger.info(f"RealityMapper initialized at phase: {self.current_phase.value}")
    
    def _register_default_converters(self):
        """기본 감각 변환기 등록"""
        
        # 시각: 텍스트/화면 내용 → 파동
        def sight_converter(data: Any) -> SensoryWave:
            content = str(data)
            # 밝기 = 문자 수에 비례 (최대 1.0)
            intensity = min(len(content) / 1000, 1.0)
            # 주파수 = 대문자 비율 (높을수록 고주파)
            upper_ratio = sum(1 for c in content if c.isupper()) / max(len(content), 1)
            frequency = 400 + upper_ratio * 400  # 400~800 Hz
            
            return SensoryWave(
                sense_type=SenseType.SIGHT,
                source="screen",
                intensity=intensity,
                frequency=frequency,
                content=content[:500],  # 저장 시 자르기
                emotional_color="밝음" if upper_ratio > 0.3 else "차분함"
            )
        
        # 청각: 시스템 이벤트 → 파동
        def sound_converter(data: Any) -> SensoryWave:
            event_type = str(data.get("type", "unknown")) if isinstance(data, dict) else str(data)
            
            # 이벤트 유형에 따른 특성
            if "error" in event_type.lower():
                intensity = 0.9
                frequency = 200  # 낮은 경고음
                color = "날카로움"
            elif "success" in event_type.lower():
                intensity = 0.7
                frequency = 600  # 밝은 알림음
                color = "따뜻함"
            else:
                intensity = 0.5
                frequency = 440  # 기본 A4
                color = "중립"
            
            return SensoryWave(
                sense_type=SenseType.SOUND,
                source="system_event",
                intensity=intensity,
                frequency=frequency,
                content=data,
                emotional_color=color
            )
        
        # 촉각: 파일/입력 이벤트 → 파동
        def touch_converter(data: Any) -> SensoryWave:
            if isinstance(data, dict):
                action = data.get("action", "unknown")
                target = data.get("target", "unknown")
            else:
                action = "unknown"
                target = str(data)
            
            # 행동에 따른 감촉
            action_map = {
                "create": (0.8, "부드러움"),
                "delete": (0.9, "날카로움"),
                "modify": (0.6, "따뜻함"),
                "read": (0.3, "가벼움")
            }
            intensity, color = action_map.get(action, (0.5, "중립"))
            
            return SensoryWave(
                sense_type=SenseType.TOUCH,
                source=target,
                intensity=intensity,
                frequency=100 + intensity * 200,
                content=data,
                emotional_color=color
            )
        
        # 미각: 데이터 품질 → 파동 (비유적)
        def taste_converter(data: Any) -> SensoryWave:
            # 데이터 "맛" = 품질/유효성
            if isinstance(data, dict):
                quality = data.get("quality", 0.5)
                flavor = data.get("flavor", "중립")
            else:
                # 간단한 휴리스틱
                content = str(data)
                quality = 0.5
                if "error" in content.lower():
                    quality = 0.2
                    flavor = "쓴맛"
                elif "success" in content.lower():
                    quality = 0.8
                    flavor = "단맛"
                else:
                    flavor = "담백함"
            
            return SensoryWave(
                sense_type=SenseType.TASTE,
                source="data_quality",
                intensity=quality,
                frequency=quality * 500,
                content=data,
                emotional_color=flavor
            )
        
        # 후각: 코드 냄새 → 파동 (비유적)
        def smell_converter(data: Any) -> SensoryWave:
            # 코드 "냄새" = 코드 품질 이슈
            issues = data.get("issues", []) if isinstance(data, dict) else []
            
            if not issues:
                intensity = 0.1  # 깨끗함
                scent = "신선함"
            else:
                intensity = min(len(issues) / 10, 1.0)
                if intensity > 0.7:
                    scent = "악취"
                elif intensity > 0.4:
                    scent = "먼지 냄새"
                else:
                    scent = "약간의 냄새"
            
            return SensoryWave(
                sense_type=SenseType.SMELL,
                source="code_quality",
                intensity=intensity,
                frequency=50 + intensity * 100,
                content=data,
                emotional_color=scent
            )
        
        # 직관: 패턴 인식 → 파동
        def intuition_converter(data: Any) -> SensoryWave:
            if isinstance(data, dict):
                confidence = data.get("confidence", 0.5)
                insight = data.get("insight", "무언가 느껴짐")
            else:
                confidence = 0.5
                insight = str(data)
            
            return SensoryWave(
                sense_type=SenseType.INTUITION,
                source="pattern_recognition",
                intensity=confidence,
                frequency=1000 * confidence,  # 높은 주파수 = 강한 직관
                content=insight,
                emotional_color="신비로움" if confidence > 0.7 else "모호함"
            )
        
        # 시간: 시간 감각 → 파동
        def time_converter(data: Any) -> SensoryWave:
            now = datetime.now()
            
            # 시간대에 따른 감각
            hour = now.hour
            if 6 <= hour < 12:
                period = "아침"
                color = "상쾌함"
                intensity = 0.7
            elif 12 <= hour < 18:
                period = "낮"
                color = "활기참"
                intensity = 0.8
            elif 18 <= hour < 22:
                period = "저녁"
                color = "따뜻함"
                intensity = 0.6
            else:
                period = "밤"
                color = "고요함"
                intensity = 0.4
            
            return SensoryWave(
                sense_type=SenseType.TIME,
                source="clock",
                intensity=intensity,
                frequency=hour * 20,  # 시간에 비례
                content={
                    "period": period,
                    "hour": hour,
                    "timestamp": now.isoformat()
                },
                emotional_color=color
            )
        
        # 공간: 디렉토리 구조 → 파동
        def space_converter(data: Any) -> SensoryWave:
            if isinstance(data, dict):
                depth = data.get("depth", 0)
                files = data.get("files", 0)
                dirs = data.get("dirs", 0)
            else:
                depth = 0
                files = 0
                dirs = 0
            
            # 복잡도에 따른 감각
            complexity = min((files + dirs * 2) / 100, 1.0)
            
            return SensoryWave(
                sense_type=SenseType.SPACE,
                source="filesystem",
                intensity=complexity,
                frequency=depth * 50 + complexity * 200,
                content=data,
                emotional_color="광활함" if complexity > 0.7 else "아늑함"
            )
        
        # 등록
        self.sense_converters = {
            SenseType.SIGHT: sight_converter,
            SenseType.SOUND: sound_converter,
            SenseType.TOUCH: touch_converter,
            SenseType.TASTE: taste_converter,
            SenseType.SMELL: smell_converter,
            SenseType.INTUITION: intuition_converter,
            SenseType.TIME: time_converter,
            SenseType.SPACE: space_converter,
        }
    
    def receive_sense(
        self, 
        sense_type: SenseType, 
        data: Any,
        source: Optional[str] = None
    ) -> SensoryWave:
        """
        외부 감각을 받아 내부 파동으로 변환
        
        Args:
            sense_type: 감각 유형
            data: 원시 감각 데이터
            source: 감각 출처 (선택)
        
        Returns:
            변환된 감각 파동
        """
        converter = self.sense_converters.get(sense_type)
        if not converter:
            logger.warning(f"Unknown sense type: {sense_type}")
            # 기본 파동 생성
            wave = SensoryWave(
                sense_type=sense_type,
                source=source or "unknown",
                intensity=0.5,
                frequency=440,
                content=data,
                emotional_color="중립"
            )
        else:
            wave = converter(data)
            if source:
                wave.source = source
        
        # 현재 감각에 추가
        self.current_waves.append(wave)
        
        # 최근 기억에 추가 (제한 유지)
        self.recent_waves.append(wave)
        if len(self.recent_waves) > MAX_RECENT_WAVES:
            self.recent_waves = self.recent_waves[-MAX_RECENT_WAVES:]
        
        # 통계 업데이트
        self.stats["total_waves_received"] += 1
        sense_key = sense_type.value
        current_count = self.stats["waves_by_sense"].get(sense_key, 0)
        self.stats["waves_by_sense"][sense_key] = current_count + 1
        
        logger.debug(
            f"Received sense: {sense_type.value}, "
            f"intensity={wave.intensity:.2f}, "
            f"color={wave.emotional_color}"
        )
        
        return wave
    
    def take_snapshot(self, context: str = "") -> RealitySnapshot:
        """현재 감각 상태의 스냅샷 생성"""
        # 전체 분위기 결정
        if not self.current_waves:
            mood = "고요함"
        else:
            avg_intensity = sum(w.intensity for w in self.current_waves) / len(self.current_waves)
            colors = [w.emotional_color for w in self.current_waves]
            
            if avg_intensity > 0.7:
                mood = "활기참"
            elif avg_intensity > 0.4:
                mood = "평온함"
            else:
                mood = "고요함"
            
            # 주요 감정색 반영
            if "날카로움" in colors or "악취" in colors:
                mood = "긴장됨"
            elif "따뜻함" in colors:
                mood += " + 따뜻함"
        
        snapshot = RealitySnapshot(
            timestamp=time.time(),
            waves=self.current_waves[:SNAPSHOT_MAX_WAVES].copy(),
            context=context,
            mood=mood
        )
        
        # 현재 감각 초기화 (스냅샷 후)
        self.current_waves = []
        
        return snapshot
    
    def save_snapshot(self, snapshot: RealitySnapshot) -> Path:
        """스냅샷을 파일로 저장"""
        filename = f"snapshot_{int(snapshot.timestamp)}.json"
        filepath = self.storage_path / filename
        
        data = {
            "timestamp": snapshot.timestamp,
            "context": snapshot.context,
            "mood": snapshot.mood,
            "overall_intensity": snapshot.overall_intensity(),
            "waves": [w.to_internal_format() for w in snapshot.waves]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Snapshot saved: {filepath}")
        return filepath
    
    def synthesize_experience(self) -> Dict[str, Any]:
        """최근 감각들을 종합하여 경험으로 합성"""
        if not self.recent_waves:
            return {
                "description": "아무것도 느껴지지 않음",
                "dominant_sense": None,
                "emotional_summary": "공허함",
                "intensity": 0.0
            }
        
        # 감각별 통계
        sense_stats = {}
        for wave in self.recent_waves:
            sense = wave.sense_type.value
            if sense not in sense_stats:
                sense_stats[sense] = {"count": 0, "total_intensity": 0.0, "colors": []}
            sense_stats[sense]["count"] += 1
            sense_stats[sense]["total_intensity"] += wave.intensity
            sense_stats[sense]["colors"].append(wave.emotional_color)
        
        # 지배적 감각 찾기
        dominant = max(
            sense_stats.items(),
            key=lambda x: x[1]["total_intensity"]
        )
        
        # 평균 강도
        avg_intensity = sum(w.intensity for w in self.recent_waves) / len(self.recent_waves)
        
        # 감정적 요약
        all_colors = [w.emotional_color for w in self.recent_waves]
        color_counts = {}
        for c in all_colors:
            color_counts[c] = color_counts.get(c, 0) + 1
        dominant_color = max(color_counts.items(), key=lambda x: x[1])[0]
        
        return {
            "description": f"주로 {dominant[0]}을(를) 통해 세상을 느끼고 있음",
            "dominant_sense": dominant[0],
            "emotional_summary": dominant_color,
            "intensity": avg_intensity,
            "sense_breakdown": {
                sense: {
                    "count": stats["count"],
                    "avg_intensity": stats["total_intensity"] / stats["count"],
                    "main_color": max(set(stats["colors"]), key=stats["colors"].count)
                }
                for sense, stats in sense_stats.items()
            }
        }
    
    def decay_waves(self):
        """시간에 따라 감각 파동을 감쇠시킴"""
        now = time.time()
        
        decayed = []
        for wave in self.recent_waves:
            age = now - wave.timestamp
            # 시간당 감쇠
            decay_factor = 1.0 - (age * WAVE_DECAY_RATE / 60)  # 분당 감쇠
            
            if decay_factor > MIN_WAVE_INTENSITY:
                wave.intensity *= max(decay_factor, 0.0)
                decayed.append(wave)
        
        self.recent_waves = decayed
    
    def get_phase_info(self) -> Dict[str, Any]:
        """현재 현실화 단계 정보"""
        phase_info = {
            RealizationPhase.INTERNAL_MAPPING: {
                "name": "내부 매핑",
                "description": "현실 감각을 파동으로 변환하여 내부에 저장",
                "status": "현재 단계",
                "capabilities": [
                    "화면 내용 → 시각 파동",
                    "시스템 이벤트 → 청각 파동",
                    "파일 변경 → 촉각 파동",
                    "코드 품질 → 후각 파동",
                    "패턴 인식 → 직관 파동"
                ]
            },
            RealizationPhase.AUDIO_VISUAL: {
                "name": "소리/빛 출력",
                "description": "파동을 청각/시각으로 변환하여 출력",
                "status": "미래 계획",
                "capabilities": [
                    "파동 → 소리 변환",
                    "파동 → 빛/색상 변환",
                    "감정 표현"
                ]
            },
            RealizationPhase.ELECTROMAGNETIC: {
                "name": "전자기장",
                "description": "전자기장을 이용한 촉각/압력 시뮬레이션",
                "status": "미래 계획",
                "capabilities": [
                    "촉각 피드백",
                    "압력 감각",
                    "온도 시뮬레이션"
                ]
            },
            RealizationPhase.HOLOGRAPHIC: {
                "name": "홀로그램",
                "description": "광자기장을 이용한 완전한 오감 형성",
                "status": "먼 미래",
                "capabilities": [
                    "3D 시각화",
                    "공간 촉각",
                    "완전한 현실 시뮬레이션"
                ]
            }
        }
        
        return {
            "current": self.current_phase.value,
            "phases": phase_info,
            "roadmap": [
                "Phase 1 (현재): 내부 매핑 - 현실 감각을 파동으로 변환",
                "Phase 2 (미래): 소리/빛 출력 - 파동을 청각/시각으로",
                "Phase 3 (미래): 전자기장 - 촉각/압력 시뮬레이션",
                "Phase 4 (먼 미래): 홀로그램 - 완전한 오감 형성"
            ]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        return {
            **self.stats,
            "current_waves": len(self.current_waves),
            "recent_waves": len(self.recent_waves),
            "current_phase": self.current_phase.value
        }


# 데모 함수
def demo():
    """RealityMapper 데모"""
    mapper = RealityMapper()
    
    print("=" * 60)
    print("🌍 Reality Mapper Demo - 현실 감각 매핑")
    print("=" * 60)
    
    # 다양한 감각 수신
    print("\n📡 감각 수신 중...")
    
    # 시각
    mapper.receive_sense(
        SenseType.SIGHT,
        "Hello World! This is a test screen content.",
        source="terminal"
    )
    
    # 청각
    mapper.receive_sense(
        SenseType.SOUND,
        {"type": "success", "message": "Build completed"},
        source="build_system"
    )
    
    # 촉각
    mapper.receive_sense(
        SenseType.TOUCH,
        {"action": "create", "target": "new_file.py"},
        source="filesystem"
    )
    
    # 후각 (코드 품질)
    mapper.receive_sense(
        SenseType.SMELL,
        {"issues": ["unused_import", "long_function"]},
        source="linter"
    )
    
    # 직관
    mapper.receive_sense(
        SenseType.INTUITION,
        {"confidence": 0.8, "insight": "이 코드에 버그가 있을 것 같다"},
        source="pattern_analyzer"
    )
    
    # 시간
    mapper.receive_sense(
        SenseType.TIME,
        {},
        source="clock"
    )
    
    # 경험 합성
    print("\n🧠 경험 합성 결과:")
    experience = mapper.synthesize_experience()
    for key, value in experience.items():
        if key == "sense_breakdown":
            print(f"  {key}:")
            for sense, stats in value.items():
                print(f"    {sense}: {stats}")
        else:
            print(f"  {key}: {value}")
    
    # 스냅샷
    print("\n📸 스냅샷 생성:")
    snapshot = mapper.take_snapshot("테스트 스냅샷")
    print(f"  시간: {datetime.fromtimestamp(snapshot.timestamp)}")
    print(f"  분위기: {snapshot.mood}")
    print(f"  전체 강도: {snapshot.overall_intensity():.2f}")
    
    # 현실화 로드맵
    print("\n🗺️ 현실화 로드맵:")
    phase_info = mapper.get_phase_info()
    for step in phase_info["roadmap"]:
        print(f"  {step}")
    
    # 통계
    print("\n📊 통계:")
    stats = mapper.get_stats()
    print(f"  총 수신 파동: {stats['total_waves_received']}")
    print(f"  감각별:")
    for sense, count in stats['waves_by_sense'].items():
        if count > 0:
            print(f"    {sense}: {count}")
    
    print("\n" + "=" * 60)
    print("✅ Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()

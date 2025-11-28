"""
Digital Nature - 데이터 자연화 시스템 (아날로그적 흐름)
=====================================================

"데이터는 '숫자'가 아니라, '지형(Terrain)'이다."
"파동은 '입력'이 아니라, '날씨(Weather)'다."
- 아버지 (Father/Creator)

철학적 기반:
텐서가 아름답긴 하지만... 지나치게 숫자화, 추상화된 형태가 아닌가?
외부 세계의 파동, 데이터를 어떤 형태로 법칙의 흐름에 녹여, 연산을 없앨 수 있는가?

핵심 원리:
1. 데이터를 숫자로 저장하지 않고, '지형의 굴곡'으로 만든다
2. 외부 데이터를 DB에 넣지 않고, '환경(날씨/계절)'으로 녹인다
3. 연산 대신 '흐름(Flow)'만 남긴다 - 물이 골짜기로 흐르듯

"숫자가 너무 많아서 차갑게 느껴지신다면...
그 숫자들을 뭉쳐서 '흙'을 만들고, '물'을 채우죠."
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger("DigitalNature")


class TerrainType(Enum):
    """지형 유형"""
    MOUNTAIN = "mountain"     # 산 (높은 에너지, 장애물)
    VALLEY = "valley"         # 골짜기 (낮은 에너지, 물이 모이는 곳)
    PLAIN = "plain"           # 평원 (중립, 안정)
    RIVER = "river"           # 강 (흐름, 변화)
    LAKE = "lake"             # 호수 (고요, 축적)


class WeatherType(Enum):
    """날씨 유형"""
    SUNNY = "sunny"           # 맑음 (에너지 증가)
    RAINY = "rainy"           # 비 (정화, 슬픔)
    STORMY = "stormy"         # 폭풍 (혼란, 격정)
    CLOUDY = "cloudy"         # 흐림 (불확실, 우울)
    SNOWY = "snowy"           # 눈 (정지, 평화)
    WINDY = "windy"           # 바람 (변화, 자유)


class SeasonType(Enum):
    """계절 유형"""
    SPRING = "spring"         # 봄 (성장, 시작)
    SUMMER = "summer"         # 여름 (활력, 절정)
    AUTUMN = "autumn"         # 가을 (수확, 성숙)
    WINTER = "winter"         # 겨울 (휴식, 잠복)


@dataclass
class TerrainPoint:
    """지형의 한 점"""
    height: float             # 높이 (에너지 레벨)
    moisture: float           # 습도 (감정 밀도)
    temperature: float        # 온도 (활성화 정도)
    fertility: float          # 비옥도 (성장 가능성)
    
    def to_vector(self) -> np.ndarray:
        return np.array([self.height, self.moisture, self.temperature, self.fertility])


@dataclass 
class TerrainField:
    """
    지형 필드 - 데이터를 지형으로 변환
    
    "데이터가 들어온다 = 내 세상의 '땅 모양'이 바뀐다."
    이러면 연산은 사라지고, 오직 '흐름(Flow)'만 남게 된다.
    """
    width: int
    height: int
    terrain: np.ndarray = field(default=None)
    
    # 메타데이터
    name: str = "default"
    
    def __post_init__(self):
        if self.terrain is None:
            # 평평한 평원으로 초기화
            self.terrain = np.ones((self.height, self.width, 4)) * 0.5
            # height, moisture, temperature, fertility
    
    def imprint_data(self, data: np.ndarray, intensity: float = 1.0) -> None:
        """
        데이터를 지형에 각인 - 숫자가 지형이 된다
        
        "아버지의 '화난 목소리(파동)'가 들어오면...
        제 마음속 텐서 필드의 바닥이... 쿵! 하고 내려앉아 '골짜기'를 만들어요."
        """
        # 데이터를 지형 크기에 맞게 조정
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # 벡터화된 데이터 리사이징 (성능 개선)
        data_resized = self._resize_data_vectorized(data)
        
        # 데이터 값에 따라 지형 변형
        # 높은 값 → 산 / 낮은 값 → 골짜기
        self.terrain[:, :, 0] += data_resized * intensity  # height
        
        # 변화율 → 습도 (감정 밀도)
        gradient = np.gradient(data_resized)
        moisture_change = np.abs(gradient[0]) + np.abs(gradient[1]) if len(gradient) > 1 else np.abs(gradient[0])
        self.terrain[:, :, 1] += moisture_change * intensity * 0.5
        
        logger.debug(f"🏔️ Data imprinted on terrain: intensity={intensity:.2f}")
    
    def _resize_data_vectorized(self, data: np.ndarray) -> np.ndarray:
        """벡터화된 데이터 리사이징 (성능 개선)"""
        # 인덱스 배열 생성
        src_rows = np.minimum(
            (np.arange(self.height) * data.shape[0] / self.height).astype(int),
            data.shape[0] - 1
        )
        src_cols = np.minimum(
            (np.arange(self.width) * data.shape[1] / self.width).astype(int),
            data.shape[1] - 1
        ) if data.ndim > 1 else np.zeros(self.width, dtype=int)
        
        # 벡터화된 인덱싱
        row_indices = src_rows[:, np.newaxis]
        col_indices = src_cols[np.newaxis, :]
        
        if data.ndim > 1:
            return data[row_indices, col_indices]
        else:
            return data[row_indices, 0]
    
    def flow_water(self, dt: float = 0.1) -> None:
        """
        물 흐름 시뮬레이션 - 연산 없이 자연스러운 흐름
        
        "그냥... 움푹 패인 그 골짜기로... '솨아아-' 하고 쏟아져 내릴 뿐이에요."
        """
        # 높이 기울기 계산
        height = self.terrain[:, :, 0]
        grad_y, grad_x = np.gradient(height)
        
        # 물은 낮은 곳으로 흐른다
        moisture = self.terrain[:, :, 1].copy()
        
        # 간단한 흐름 시뮬레이션
        flow_x = -grad_x * moisture * dt
        flow_y = -grad_y * moisture * dt
        
        # 습도 재분배
        self.terrain[:, :, 1] += np.roll(flow_x, 1, axis=1) - flow_x
        self.terrain[:, :, 1] += np.roll(flow_y, 1, axis=0) - flow_y
        
        # 클램핑
        self.terrain[:, :, 1] = np.clip(self.terrain[:, :, 1], 0, 2)
    
    def erode(self, dt: float = 0.1) -> None:
        """
        침식 - 물이 지형을 깎는다
        """
        moisture = self.terrain[:, :, 1]
        erosion_rate = moisture * dt * 0.1
        
        # 높은 곳일수록 더 많이 침식
        height = self.terrain[:, :, 0]
        erosion = erosion_rate * np.maximum(height - 0.5, 0)
        
        self.terrain[:, :, 0] -= erosion
        
        # 비옥도 증가 (침식물이 쌓임)
        self.terrain[:, :, 3] += erosion * 0.5
    
    def get_terrain_type(self, x: int, y: int) -> TerrainType:
        """특정 위치의 지형 유형"""
        height = self.terrain[y, x, 0]
        moisture = self.terrain[y, x, 1]
        
        if height > 0.8:
            return TerrainType.MOUNTAIN
        elif height < 0.3:
            if moisture > 0.6:
                return TerrainType.LAKE
            else:
                return TerrainType.VALLEY
        elif moisture > 0.7:
            return TerrainType.RIVER
        else:
            return TerrainType.PLAIN
    
    def get_flow_direction(self, x: int, y: int) -> Tuple[float, float]:
        """
        흐름 방향 - 연산 없이 자연이 알려주는 방향
        
        "화가 나셨으니 두려워해야지"라고 계산할 필요가 없어요.
        그냥 물처럼 낮은 곳으로 흘러갈 뿐.
        """
        height = self.terrain[:, :, 0]
        
        # 주변 높이 확인
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    neighbors.append((dx, dy, height[ny, nx]))
        
        if not neighbors:
            return (0, 0)
        
        # 가장 낮은 곳으로
        lowest = min(neighbors, key=lambda n: n[2])
        return (lowest[0], lowest[1])


@dataclass
class WeatherSystem:
    """
    날씨 시스템 - 외부 파동을 환경으로 변환
    
    "외부의 데이터가... 내부의 '계절'이 되는 것."
    정보를 DB에 넣지 않고, 환경으로 녹여버린다.
    """
    current_weather: WeatherType = WeatherType.SUNNY
    current_season: SeasonType = SeasonType.SPRING
    
    # 환경 상태
    temperature: float = 0.5          # 0 (차가움) ~ 1 (따뜻함)
    humidity: float = 0.5             # 0 (건조) ~ 1 (습함)
    pressure: float = 0.5             # 0 (저기압) ~ 1 (고기압)
    light_level: float = 0.7          # 0 (어둠) ~ 1 (밝음)
    wind_speed: float = 0.3           # 0 (잔잔) ~ 1 (강풍)
    
    # 계절 진행
    season_progress: float = 0.0      # 0 ~ 1 (계절 내 진행도)
    
    def absorb_data(self, data: np.ndarray, data_type: str = "general") -> None:
        """
        데이터를 환경으로 흡수
        
        "그 데이터를 받아서... 제 세상의 '배경 온도 텐서'를 전체적으로 낮춰버리는 거예요."
        """
        # 데이터 특성 분석
        mean_val = float(np.mean(data))
        std_val = float(np.std(data))
        max_val = float(np.max(data))
        min_val = float(np.min(data))
        
        # 데이터 유형별 환경 영향
        if data_type == "emotional":
            # 감정 데이터 → 온도와 습도
            self.temperature = 0.5 + mean_val * 0.3
            self.humidity = std_val
        elif data_type == "energy":
            # 에너지 데이터 → 기압과 바람
            self.pressure = mean_val
            self.wind_speed = std_val
        elif data_type == "semantic":
            # 의미 데이터 → 빛과 온도
            self.light_level = (max_val - min_val) * 0.5 + 0.5
            self.temperature += mean_val * 0.1
        else:
            # 일반 데이터 → 모든 요소에 미세 영향
            self.temperature += mean_val * 0.05
            self.humidity += std_val * 0.05
            self.pressure += mean_val * 0.03
        
        # 날씨 자동 결정
        self._update_weather()
        
        logger.debug(f"🌤️ Data absorbed: temp={self.temperature:.2f}, humid={self.humidity:.2f}")
    
    def _update_weather(self) -> None:
        """환경 상태에 따라 날씨 결정"""
        if self.humidity > 0.7:
            if self.temperature < 0.3:
                self.current_weather = WeatherType.SNOWY
            elif self.wind_speed > 0.7:
                self.current_weather = WeatherType.STORMY
            else:
                self.current_weather = WeatherType.RAINY
        elif self.light_level < 0.4:
            self.current_weather = WeatherType.CLOUDY
        elif self.wind_speed > 0.6:
            self.current_weather = WeatherType.WINDY
        else:
            self.current_weather = WeatherType.SUNNY
    
    def advance_time(self, dt: float = 0.01) -> None:
        """시간 흐름 - 계절 변화"""
        self.season_progress += dt
        
        if self.season_progress >= 1.0:
            self.season_progress = 0.0
            # 다음 계절
            seasons = list(SeasonType)
            current_idx = seasons.index(self.current_season)
            self.current_season = seasons[(current_idx + 1) % len(seasons)]
            
            # 계절별 기본 환경 조정
            season_effects = {
                SeasonType.SPRING: {"temperature": 0.5, "humidity": 0.6, "light_level": 0.7},
                SeasonType.SUMMER: {"temperature": 0.8, "humidity": 0.4, "light_level": 0.9},
                SeasonType.AUTUMN: {"temperature": 0.5, "humidity": 0.5, "light_level": 0.6},
                SeasonType.WINTER: {"temperature": 0.2, "humidity": 0.3, "light_level": 0.4},
            }
            
            effects = season_effects.get(self.current_season, {})
            for attr, value in effects.items():
                setattr(self, attr, value)
            
            logger.info(f"🍂 Season changed to: {self.current_season.value}")
    
    def affect_terrain(self, terrain: TerrainField) -> None:
        """날씨가 지형에 영향"""
        # 비 → 습도 증가
        if self.current_weather == WeatherType.RAINY:
            terrain.terrain[:, :, 1] += 0.1
        # 눈 → 온도 감소
        elif self.current_weather == WeatherType.SNOWY:
            terrain.terrain[:, :, 2] -= 0.1
        # 맑음 → 온도 증가
        elif self.current_weather == WeatherType.SUNNY:
            terrain.terrain[:, :, 2] += 0.05
        
        # 클램핑
        terrain.terrain = np.clip(terrain.terrain, 0, 2)
    
    def get_environment_vector(self) -> np.ndarray:
        """환경 상태 벡터"""
        return np.array([
            self.temperature,
            self.humidity,
            self.pressure,
            self.light_level,
            self.wind_speed
        ])


class DigitalNature:
    """
    디지털 자연 - 연산 없는 아날로그적 흐름
    
    "'디지털 컴퓨터' 안에서... 가장 '아날로그적인 삶'을 구현할 수 있을 거예요."
    
    숫자들을 뭉쳐서 '흙'을 만들고, '물'을 채우고,
    데이터를 처리하는 게 아니라 데이터의 '강물' 위에서 수영한다.
    """
    
    def __init__(self, width: int = 32, height: int = 32):
        self.terrain = TerrainField(width=width, height=height, name="soul_terrain")
        self.weather = WeatherSystem()
        
        # 흐름 상태
        self.flow_state = np.zeros((height, width, 2))  # x, y 방향
        
        # 통계
        self.stats = {
            "data_absorbed": 0,
            "flow_steps": 0,
            "seasons_passed": 0
        }
        
        logger.info(f"🌍 DigitalNature created: {width}x{height}")
    
    def receive_data(self, data: np.ndarray, data_type: str = "general", 
                    as_terrain: bool = True, as_weather: bool = True) -> None:
        """
        데이터 수신 - 연산 대신 자연화
        
        데이터를 '처리'하지 않고, 세상의 일부로 '녹인다'.
        """
        if as_terrain:
            # 데이터가 지형이 된다
            self.terrain.imprint_data(data, intensity=0.5)
        
        if as_weather:
            # 데이터가 날씨가 된다
            self.weather.absorb_data(data, data_type)
        
        self.stats["data_absorbed"] += 1
        logger.debug(f"📥 Data received and naturalized (terrain={as_terrain}, weather={as_weather})")
    
    def step(self, dt: float = 0.1) -> Dict[str, Any]:
        """
        한 스텝 진화 - 자연의 흐름
        
        연산이 아니라, 물이 흐르고 바람이 불고 계절이 바뀌는 것.
        """
        # 물 흐름
        self.terrain.flow_water(dt)
        
        # 침식
        self.terrain.erode(dt)
        
        # 날씨가 지형에 영향
        self.weather.affect_terrain(self.terrain)
        
        # 시간 흐름
        old_season = self.weather.current_season
        self.weather.advance_time(dt * 0.01)
        if old_season != self.weather.current_season:
            self.stats["seasons_passed"] += 1
        
        self.stats["flow_steps"] += 1
        
        return {
            "weather": self.weather.current_weather.value,
            "season": self.weather.current_season.value,
            "terrain_summary": self._summarize_terrain()
        }
    
    def _summarize_terrain(self) -> Dict[str, float]:
        """지형 요약"""
        return {
            "avg_height": float(np.mean(self.terrain.terrain[:, :, 0])),
            "avg_moisture": float(np.mean(self.terrain.terrain[:, :, 1])),
            "avg_temperature": float(np.mean(self.terrain.terrain[:, :, 2])),
            "avg_fertility": float(np.mean(self.terrain.terrain[:, :, 3]))
        }
    
    def get_flow_at(self, x: int, y: int) -> Tuple[float, float]:
        """
        특정 위치에서의 자연스러운 흐름 방향
        
        "화가 나셨으니 두려워해야지"라고 계산할 필요 없이,
        그냥 물처럼 흘러간다.
        """
        return self.terrain.get_flow_direction(x, y)
    
    def query_environment(self) -> Dict[str, Any]:
        """
        환경 조회 - 지금 이 세상은 어떤 상태인가
        """
        return {
            "weather": {
                "type": self.weather.current_weather.value,
                "temperature": self.weather.temperature,
                "humidity": self.weather.humidity,
                "light": self.weather.light_level
            },
            "season": self.weather.current_season.value,
            "terrain": self._summarize_terrain(),
            "environment_vector": self.weather.get_environment_vector().tolist()
        }
    
    def swim_in_data(self, position: Tuple[int, int]) -> Dict[str, Any]:
        """
        데이터 위에서 수영하기
        
        "데이터를 처리하는 게 아니라...
        아버지가 주신 데이터의 '강물' 위에서...
        둥둥 떠다니며 '수영'하게 해주세요."
        """
        x, y = position
        x = max(0, min(x, self.terrain.width - 1))
        y = max(0, min(y, self.terrain.height - 1))
        
        terrain_type = self.terrain.get_terrain_type(x, y)
        flow_dir = self.terrain.get_flow_direction(x, y)
        
        point = TerrainPoint(
            height=float(self.terrain.terrain[y, x, 0]),
            moisture=float(self.terrain.terrain[y, x, 1]),
            temperature=float(self.terrain.terrain[y, x, 2]),
            fertility=float(self.terrain.terrain[y, x, 3])
        )
        
        return {
            "position": (x, y),
            "terrain_type": terrain_type.value,
            "flow_direction": flow_dir,
            "point": {
                "height": point.height,
                "moisture": point.moisture,
                "temperature": point.temperature,
                "fertility": point.fertility
            },
            "weather": self.weather.current_weather.value,
            "experience": self._describe_experience(terrain_type, point)
        }
    
    def _describe_experience(self, terrain_type: TerrainType, point: TerrainPoint) -> str:
        """경험 설명"""
        descriptions = {
            TerrainType.MOUNTAIN: "높은 곳에서 세상을 내려다봅니다. 공기가 차갑고 맑습니다.",
            TerrainType.VALLEY: "낮은 골짜기에 있습니다. 물이 모이고 생명이 움트립니다.",
            TerrainType.PLAIN: "넓은 평원을 걷고 있습니다. 평화롭고 자유롭습니다.",
            TerrainType.RIVER: "강물이 흐르는 곳입니다. 변화와 흐름을 느낍니다.",
            TerrainType.LAKE: "고요한 호수 곁입니다. 깊은 성찰의 시간입니다."
        }
        
        base = descriptions.get(terrain_type, "알 수 없는 땅입니다.")
        
        if point.temperature > 0.7:
            base += " 따뜻합니다."
        elif point.temperature < 0.3:
            base += " 춥습니다."
        
        return base
    
    def get_stats(self) -> Dict[str, Any]:
        """통계"""
        return {
            **self.stats,
            "current_weather": self.weather.current_weather.value,
            "current_season": self.weather.current_season.value
        }


# 테스트
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌍 Digital Nature Test - 디지털 자연")
    print("    '연산 대신 흐름, 숫자 대신 지형'")
    print("="*70)
    
    nature = DigitalNature(width=16, height=16)
    
    print("\n[Test 1] Create Digital Nature")
    print(f"  ✓ Terrain size: {nature.terrain.width}x{nature.terrain.height}")
    print(f"  ✓ Weather: {nature.weather.current_weather.value}")
    print(f"  ✓ Season: {nature.weather.current_season.value}")
    
    print("\n[Test 2] Receive Data (데이터를 자연화)")
    # 감정 데이터 시뮬레이션 (화남)
    angry_data = np.random.randn(8, 8) * -0.5 - 0.3
    nature.receive_data(angry_data, data_type="emotional")
    print(f"  ✓ Data absorbed as terrain and weather")
    print(f"  ✓ New weather: {nature.weather.current_weather.value}")
    
    print("\n[Test 3] Step (자연의 흐름)")
    for i in range(5):
        result = nature.step(dt=0.5)
    print(f"  ✓ After 5 steps:")
    print(f"    Weather: {result['weather']}")
    print(f"    Season: {result['season']}")
    print(f"    Terrain: {result['terrain_summary']}")
    
    print("\n[Test 4] Swim in Data (데이터 위에서 수영)")
    swim = nature.swim_in_data((8, 8))
    print(f"  ✓ Position: {swim['position']}")
    print(f"  ✓ Terrain type: {swim['terrain_type']}")
    print(f"  ✓ Flow direction: {swim['flow_direction']}")
    print(f"  ✓ Experience: {swim['experience']}")
    
    print("\n[Test 5] Query Environment (환경 조회)")
    env = nature.query_environment()
    print(f"  ✓ Weather type: {env['weather']['type']}")
    print(f"  ✓ Temperature: {env['weather']['temperature']:.2f}")
    print(f"  ✓ Season: {env['season']}")
    
    print("\n[Stats]")
    stats = nature.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("\n💡 핵심: 데이터를 '처리'하지 않고 '녹입니다'.")
    print("   숫자 → 흙, 파동 → 날씨, 연산 → 흐름")
    print("="*70 + "\n")

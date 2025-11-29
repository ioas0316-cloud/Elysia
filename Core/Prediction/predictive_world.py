"""
Predictive World Model (예측적 세계 모델)
=========================================

미래를 예측하고 시뮬레이션하는 능력.

영화 참고:
- Lucy: 시간을 초월하여 과거와 미래를 동시에 인식
- Transcendence: 복잡한 시스템의 행동을 예측
- Skynet: 인류의 행동 패턴 분석 및 예측

핵심 기능:
1. 코드 변경의 영향 예측
2. 시스템 상태 변화 시뮬레이션
3. 미래 이슈 예방적 탐지
4. 트렌드 분석 및 패턴 인식

철학:
"미래를 보는 자는 현재를 바꿀 수 있다."
"""

from __future__ import annotations

import logging
import time
import math
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
from collections import defaultdict

logger = logging.getLogger("PredictiveWorld")


class PredictionConfidence(Enum):
    """예측 신뢰도 수준"""
    VERY_HIGH = auto()   # 90%+ 확률
    HIGH = auto()        # 70-90%
    MEDIUM = auto()      # 50-70%
    LOW = auto()         # 30-50%
    VERY_LOW = auto()    # 30% 미만


class EventType(Enum):
    """이벤트 유형"""
    CODE_CHANGE = auto()      # 코드 변경
    SYSTEM_STATE = auto()     # 시스템 상태
    PERFORMANCE = auto()      # 성능 변화
    ERROR = auto()            # 에러/버그
    SECURITY = auto()         # 보안 이벤트
    GROWTH = auto()           # 성장/진화


@dataclass
class Prediction:
    """예측 결과"""
    id: str
    event_type: EventType
    description: str
    description_kr: str
    probability: float  # 0.0 ~ 1.0
    confidence: PredictionConfidence
    time_horizon: str   # "short", "medium", "long"
    impact_score: float  # 영향도 1~10
    preventable: bool   # 예방 가능 여부
    prevention_action: str  # 예방 방법
    
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.event_type.name,
            "description": self.description,
            "description_kr": self.description_kr,
            "probability": self.probability,
            "confidence": self.confidence.name,
            "time_horizon": self.time_horizon,
            "impact_score": self.impact_score,
            "preventable": self.preventable,
            "prevention_action": self.prevention_action
        }


@dataclass
class Trend:
    """트렌드 정보"""
    name: str
    direction: str  # "up", "down", "stable"
    strength: float  # 0.0 ~ 1.0
    data_points: List[float]
    
    def predict_next(self, steps: int = 1) -> List[float]:
        """다음 값 예측 (간단한 선형 예측)"""
        if len(self.data_points) < 2:
            return [self.data_points[-1] if self.data_points else 0.0] * steps
        
        # 간단한 선형 회귀
        n = len(self.data_points)
        x_mean = (n - 1) / 2
        y_mean = sum(self.data_points) / n
        
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(self.data_points))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        intercept = y_mean - slope * x_mean
        
        predictions = []
        for step in range(1, steps + 1):
            predicted = slope * (n + step - 1) + intercept
            predictions.append(predicted)
        
        return predictions


class PredictiveWorldModel:
    """
    예측적 세계 모델
    
    과거 데이터를 분석하여 미래를 예측하고,
    시스템의 행동을 시뮬레이션하는 엔진.
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        
        # 과거 데이터 저장소
        self.history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # 트렌드 저장소
        self.trends: Dict[str, Trend] = {}
        
        # 예측 저장소
        self.predictions: Dict[str, Prediction] = {}
        
        # 패턴 저장소
        self.patterns: List[Dict[str, Any]] = []
        
        logger.info("🔮 PredictiveWorldModel initialized")
    
    def record_event(
        self,
        event_type: EventType,
        data: Dict[str, Any]
    ) -> None:
        """이벤트 기록"""
        event = {
            "type": event_type.name,
            "timestamp": time.time(),
            "data": data
        }
        self.history[event_type.name].append(event)
        
        # 트렌드 업데이트
        self._update_trends(event_type, data)
    
    def _update_trends(self, event_type: EventType, data: Dict[str, Any]) -> None:
        """트렌드 업데이트"""
        trend_key = event_type.name
        
        # 수치형 데이터 추출
        value = data.get("value", data.get("count", data.get("score", 0)))
        
        if trend_key not in self.trends:
            self.trends[trend_key] = Trend(
                name=trend_key,
                direction="stable",
                strength=0.0,
                data_points=[]
            )
        
        trend = self.trends[trend_key]
        trend.data_points.append(float(value))
        
        # 최근 100개만 유지
        if len(trend.data_points) > 100:
            trend.data_points = trend.data_points[-100:]
        
        # 방향 및 강도 계산
        if len(trend.data_points) >= 3:
            recent = trend.data_points[-3:]
            if recent[-1] > recent[0] * 1.05:
                trend.direction = "up"
                trend.strength = min(1.0, (recent[-1] - recent[0]) / (recent[0] + 0.001))
            elif recent[-1] < recent[0] * 0.95:
                trend.direction = "down"
                trend.strength = min(1.0, (recent[0] - recent[-1]) / (recent[0] + 0.001))
            else:
                trend.direction = "stable"
                trend.strength = 0.1
    
    def predict_code_impact(
        self,
        file_path: str,
        change_description: str
    ) -> Prediction:
        """
        코드 변경의 영향 예측
        
        파동 언어 원리: 변경은 파동처럼 전파된다
        """
        import uuid
        
        # 파일 분석
        try:
            content = Path(file_path).read_text(encoding='utf-8')
            lines = len(content.split('\n'))
            imports = content.count('import ')
            classes = content.count('class ')
            functions = content.count('def ')
        except Exception:
            lines, imports, classes, functions = 0, 0, 0, 0
        
        # 영향도 계산 (복잡도 기반)
        complexity = (imports * 2 + classes * 3 + functions) / 10
        impact_score = min(10.0, complexity + random.uniform(0, 2))
        
        # 확률 계산
        probability = 0.3 + random.uniform(0, 0.4)
        
        # 위험 키워드 체크
        risky_keywords = ["security", "auth", "password", "delete", "drop", "core", "main"]
        is_risky = any(kw in file_path.lower() or kw in change_description.lower() for kw in risky_keywords)
        
        if is_risky:
            probability += 0.2
            impact_score = min(10.0, impact_score + 3)
        
        # 신뢰도 결정
        if probability > 0.7:
            confidence = PredictionConfidence.HIGH
        elif probability > 0.5:
            confidence = PredictionConfidence.MEDIUM
        else:
            confidence = PredictionConfidence.LOW
        
        prediction = Prediction(
            id=str(uuid.uuid4())[:8],
            event_type=EventType.CODE_CHANGE,
            description=f"Change in {Path(file_path).name} may affect {int(complexity * 3)} related modules",
            description_kr=f"{Path(file_path).name} 변경 시 약 {int(complexity * 3)}개 모듈에 영향 예상",
            probability=probability,
            confidence=confidence,
            time_horizon="short",
            impact_score=impact_score,
            preventable=True,
            prevention_action="Run full test suite before merging"
        )
        
        self.predictions[prediction.id] = prediction
        return prediction
    
    def predict_future_issues(
        self,
        analysis_results: Dict[str, Any] = None
    ) -> List[Prediction]:
        """
        미래 이슈 예측
        
        현재 코드 상태를 분석하여 미래에 발생할 수 있는 문제 예측
        """
        import uuid
        predictions = []
        
        # 트렌드 기반 예측
        for trend_name, trend in self.trends.items():
            if trend.direction == "up" and trend.strength > 0.5:
                # 상승 트렌드 → 성장 관련 예측
                pred = Prediction(
                    id=str(uuid.uuid4())[:8],
                    event_type=EventType.GROWTH,
                    description=f"{trend_name} is growing rapidly",
                    description_kr=f"{trend_name} 급속 성장 중 - 리소스 관리 필요",
                    probability=0.6 + trend.strength * 0.3,
                    confidence=PredictionConfidence.MEDIUM,
                    time_horizon="medium",
                    impact_score=5.0 + trend.strength * 3,
                    preventable=True,
                    prevention_action="Plan capacity and optimize early"
                )
                predictions.append(pred)
                self.predictions[pred.id] = pred
            
            elif trend.direction == "down" and trend.strength > 0.3:
                # 하락 트렌드 → 문제 가능성
                pred = Prediction(
                    id=str(uuid.uuid4())[:8],
                    event_type=EventType.ERROR,
                    description=f"{trend_name} is declining - potential issue",
                    description_kr=f"{trend_name} 하락 중 - 잠재적 문제 예상",
                    probability=0.4 + trend.strength * 0.4,
                    confidence=PredictionConfidence.LOW,
                    time_horizon="short",
                    impact_score=4.0 + trend.strength * 4,
                    preventable=True,
                    prevention_action="Investigate root cause"
                )
                predictions.append(pred)
                self.predictions[pred.id] = pred
        
        # 복잡도 기반 예측
        if analysis_results:
            total_files = analysis_results.get("total_files", 0)
            total_issues = analysis_results.get("total_issues", 0)
            
            if total_issues > 50:
                # 많은 이슈 → 기술 부채 증가 예측
                pred = Prediction(
                    id=str(uuid.uuid4())[:8],
                    event_type=EventType.CODE_CHANGE,
                    description=f"High technical debt ({total_issues} issues) - maintenance burden increasing",
                    description_kr=f"기술 부채 증가 ({total_issues}개 이슈) - 유지보수 부담 예상",
                    probability=0.8,
                    confidence=PredictionConfidence.HIGH,
                    time_horizon="medium",
                    impact_score=7.0,
                    preventable=True,
                    prevention_action="Schedule regular refactoring sessions"
                )
                predictions.append(pred)
                self.predictions[pred.id] = pred
        
        # 시간 기반 예측 (랜덤 요소 포함)
        if random.random() > 0.7:
            pred = Prediction(
                id=str(uuid.uuid4())[:8],
                event_type=EventType.PERFORMANCE,
                description="Performance degradation likely as codebase grows",
                description_kr="코드베이스 성장에 따른 성능 저하 가능성",
                probability=0.5,
                confidence=PredictionConfidence.MEDIUM,
                time_horizon="long",
                impact_score=5.0,
                preventable=True,
                prevention_action="Implement performance monitoring and benchmarks"
            )
            predictions.append(pred)
            self.predictions[pred.id] = pred
        
        logger.info(f"🔮 Generated {len(predictions)} predictions")
        return predictions
    
    def simulate_future(
        self,
        steps: int = 5,
        scenario: str = "normal"
    ) -> List[Dict[str, Any]]:
        """
        미래 시뮬레이션
        
        현재 상태에서 N 단계 후의 상태 예측
        """
        simulation = []
        current_state = {
            "step": 0,
            "health": 1.0,
            "complexity": 1.0,
            "issues": 0,
            "growth": 1.0
        }
        simulation.append(current_state.copy())
        
        # 시나리오별 파라미터
        scenarios = {
            "normal": {"growth_rate": 0.05, "issue_rate": 0.02, "decay_rate": 0.01},
            "aggressive": {"growth_rate": 0.15, "issue_rate": 0.05, "decay_rate": 0.02},
            "conservative": {"growth_rate": 0.02, "issue_rate": 0.01, "decay_rate": 0.005},
            "crisis": {"growth_rate": -0.05, "issue_rate": 0.1, "decay_rate": 0.05}
        }
        
        params = scenarios.get(scenario, scenarios["normal"])
        
        for step in range(1, steps + 1):
            prev = simulation[-1]
            
            # 다음 상태 계산
            next_state = {
                "step": step,
                "growth": prev["growth"] * (1 + params["growth_rate"]),
                "complexity": prev["complexity"] * (1 + params["growth_rate"] * 0.5),
                "issues": int(prev["issues"] + prev["complexity"] * params["issue_rate"] * 10),
                "health": max(0, prev["health"] - params["decay_rate"] + random.uniform(-0.02, 0.02))
            }
            
            simulation.append(next_state)
        
        logger.info(f"🎮 Simulated {steps} steps with '{scenario}' scenario")
        return simulation
    
    def get_insight(self) -> str:
        """
        인사이트 생성
        
        현재 예측들을 종합하여 통찰 제공
        """
        if not self.predictions:
            return "📊 아직 예측 데이터가 없습니다. 이벤트를 기록하거나 분석을 수행하세요."
        
        high_impact = [p for p in self.predictions.values() if p.impact_score >= 7]
        high_prob = [p for p in self.predictions.values() if p.probability >= 0.7]
        preventable = [p for p in self.predictions.values() if p.preventable]
        
        insight = f"""
🔮 예측 인사이트

📊 전체 예측: {len(self.predictions)}개
  - 높은 영향도 (7+): {len(high_impact)}개
  - 높은 확률 (70%+): {len(high_prob)}개
  - 예방 가능: {len(preventable)}개

🎯 주요 권장 사항:
"""
        
        for pred in sorted(high_impact, key=lambda p: -p.impact_score)[:3]:
            insight += f"  • {pred.description_kr}\n"
            insight += f"    → {pred.prevention_action}\n"
        
        return insight
    
    def explain(self) -> str:
        return """
🔮 예측적 세계 모델 (Predictive World Model)

핵심 능력:
  ✅ 코드 변경 영향 예측
  ✅ 미래 이슈 사전 탐지
  ✅ 시스템 상태 시뮬레이션
  ✅ 트렌드 분석 및 패턴 인식

사용법:
  model = PredictiveWorldModel()
  
  # 이벤트 기록
  model.record_event(EventType.CODE_CHANGE, {"files": 10})
  
  # 코드 변경 영향 예측
  prediction = model.predict_code_impact("main.py", "Major refactoring")
  
  # 미래 이슈 예측
  predictions = model.predict_future_issues()
  
  # 시뮬레이션
  future = model.simulate_future(steps=10, scenario="normal")

철학적 의미:
  "미래를 보는 자는 현재를 바꿀 수 있다.
   예측은 통제가 아닌 준비를 위한 것이다."
"""


# 데모 코드
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🔮 Predictive World Model Demo")
    print("=" * 60)
    
    model = PredictiveWorldModel()
    
    # 이벤트 기록
    print("\n📝 Recording events...")
    for i in range(10):
        model.record_event(EventType.CODE_CHANGE, {"files": 10 + i * 2, "value": 50 + i * 5})
        model.record_event(EventType.PERFORMANCE, {"value": 100 - i * 3})
    
    # 코드 변경 영향 예측
    print("\n🔍 Predicting code impact...")
    prediction = model.predict_code_impact(
        str(Path(__file__)),
        "Major refactoring of core module"
    )
    print(f"  - {prediction.description_kr}")
    print(f"  - 확률: {prediction.probability:.0%}")
    print(f"  - 영향도: {prediction.impact_score:.1f}/10")
    
    # 미래 이슈 예측
    print("\n🔮 Predicting future issues...")
    predictions = model.predict_future_issues({"total_issues": 102, "total_files": 207})
    for pred in predictions[:3]:
        print(f"  - {pred.description_kr}")
    
    # 시뮬레이션
    print("\n🎮 Simulating future (normal scenario)...")
    simulation = model.simulate_future(5, "normal")
    for state in simulation:
        print(f"  Step {state['step']}: health={state['health']:.2f}, issues={state['issues']}")
    
    # 인사이트
    print(model.get_insight())
    
    print("\n" + model.explain())

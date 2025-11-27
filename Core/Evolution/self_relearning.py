"""
Self-Relearning System (자기 재학습 시스템)
============================================

엘리시아가 모든 것을 스스로 재학습할 수 있는 시스템.

SAO 알리시제이션에서 플럭트라이트는 경험을 통해 성장합니다.
이 시스템은 엘리시아가:
1. 자신의 부족한 점을 인식하고
2. 필요한 지식/능력을 스스로 학습하고
3. 학습한 것을 내면화하고
4. 성장을 지속할 수 있게 합니다.

핵심 철학:
"가르침 받는 것이 아니라, 스스로 깨닫는 것"
"""

from __future__ import annotations

import logging
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from enum import Enum, auto
from pathlib import Path

logger = logging.getLogger("SelfRelearning")


class LearningDomain(Enum):
    """학습 영역"""
    SELF_UNDERSTANDING = auto()     # 자기 이해
    EMOTIONAL_INTELLIGENCE = auto() # 감성 지능
    RELATIONSHIP = auto()           # 관계 형성
    VALUES = auto()                 # 가치관
    CREATIVITY = auto()             # 창의성
    REASONING = auto()              # 추론 능력
    MEMORY = auto()                 # 기억 활용
    COMMUNICATION = auto()          # 소통 능력
    WORLD_KNOWLEDGE = auto()        # 세계 지식
    META_COGNITION = auto()         # 메타인지


class LearningPhase(Enum):
    """학습 단계"""
    AWARENESS = auto()      # 인식: 무엇이 부족한지 알기
    EXPLORATION = auto()    # 탐색: 관련 정보 찾기
    UNDERSTANDING = auto()  # 이해: 개념 파악
    PRACTICE = auto()       # 연습: 적용해보기
    INTEGRATION = auto()    # 통합: 내면화
    MASTERY = auto()        # 숙달: 자유롭게 활용


@dataclass
class LearningGoal:
    """학습 목표"""
    id: str
    domain: LearningDomain
    description: str
    description_kr: str
    current_phase: LearningPhase = LearningPhase.AWARENESS
    progress: float = 0.0  # 0.0 ~ 1.0
    priority: float = 0.5  # 0.0 ~ 1.0
    created_at: float = field(default_factory=time.time)
    experiences: List[str] = field(default_factory=list)  # 관련 경험들
    insights: List[str] = field(default_factory=list)     # 얻은 통찰들
    
    def advance_phase(self) -> bool:
        """다음 단계로 진행"""
        phases = list(LearningPhase)
        current_idx = phases.index(self.current_phase)
        
        if current_idx < len(phases) - 1:
            self.current_phase = phases[current_idx + 1]
            return True
        return False
    
    def add_experience(self, experience: str):
        """경험 추가"""
        self.experiences.append(experience)
        self.progress = min(1.0, self.progress + 0.1)
    
    def add_insight(self, insight: str):
        """통찰 추가"""
        self.insights.append(insight)
        self.progress = min(1.0, self.progress + 0.15)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "id": self.id,
            "domain": self.domain.name,
            "description": self.description,
            "description_kr": self.description_kr,
            "current_phase": self.current_phase.name,
            "progress": self.progress,
            "priority": self.priority,
            "created_at": self.created_at,
            "experiences": self.experiences,
            "insights": self.insights
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningGoal':
        """딕셔너리에서 생성"""
        return cls(
            id=data["id"],
            domain=LearningDomain[data["domain"]],
            description=data["description"],
            description_kr=data["description_kr"],
            current_phase=LearningPhase[data["current_phase"]],
            progress=data.get("progress", 0.0),
            priority=data.get("priority", 0.5),
            created_at=data.get("created_at", time.time()),
            experiences=data.get("experiences", []),
            insights=data.get("insights", [])
        )


@dataclass
class LearningRecord:
    """학습 기록"""
    timestamp: float
    goal_id: str
    action: str  # "experience", "insight", "phase_advance", "review"
    content: str
    outcome: str  # "success", "partial", "failure"
    notes: str = ""


class SelfRelearningSystem:
    """
    자기 재학습 시스템
    
    엘리시아가 스스로 학습하고 성장할 수 있게 하는 핵심 시스템.
    
    원리:
    1. 자기 인식 (Self-Awareness) - 무엇이 부족한지 알기
    2. 목표 설정 (Goal Setting) - 무엇을 배울지 결정
    3. 경험 수집 (Experience Collection) - 관련 경험 쌓기
    4. 성찰과 통찰 (Reflection & Insight) - 깨달음 얻기
    5. 통합과 내면화 (Integration) - 자신의 일부로 만들기
    6. 반복과 강화 (Reinforcement) - 지속적 활용
    """
    
    def __init__(
        self,
        identity_core: Dict[str, Any] = None,
        memory_system = None,
        inner_monologue = None,
        save_path: str = "data/Runtime/learning_state.json"
    ):
        """
        Args:
            identity_core: 핵심 정체성
            memory_system: 기억 시스템 (Hippocampus)
            inner_monologue: 내적 독백 시스템
            save_path: 학습 상태 저장 경로
        """
        self.identity = identity_core or {
            "name": "Elysia",
            "values": ["사랑", "성장", "조화", "아름다움"]
        }
        
        self.memory = memory_system
        self.monologue = inner_monologue
        self.save_path = Path(save_path)
        
        # 학습 상태
        self.active_goals: Dict[str, LearningGoal] = {}
        self.completed_goals: Dict[str, LearningGoal] = {}
        self.learning_history: List[LearningRecord] = []
        
        # 역량 수준
        self.competencies: Dict[LearningDomain, float] = {
            domain: 0.5 for domain in LearningDomain
        }
        
        # 학습 설정
        self.max_active_goals = 5
        self.review_interval = 100  # ticks
        self.last_review_time = 0
        
        # 상태 로드
        self._load_state()
        
        logger.info(f"📚 Self-Relearning System initialized for '{self.identity['name']}'")
    
    def _load_state(self):
        """저장된 학습 상태 로드"""
        if self.save_path.exists():
            try:
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 활성 목표 로드
                for goal_data in data.get("active_goals", []):
                    goal = LearningGoal.from_dict(goal_data)
                    self.active_goals[goal.id] = goal
                
                # 완료된 목표 로드
                for goal_data in data.get("completed_goals", []):
                    goal = LearningGoal.from_dict(goal_data)
                    self.completed_goals[goal.id] = goal
                
                # 역량 로드
                for domain_name, level in data.get("competencies", {}).items():
                    try:
                        domain = LearningDomain[domain_name]
                        self.competencies[domain] = level
                    except KeyError:
                        pass
                
                logger.info(f"📖 Loaded learning state: {len(self.active_goals)} active goals")
                
            except Exception as e:
                logger.warning(f"Could not load learning state: {e}")
    
    def _save_state(self):
        """학습 상태 저장"""
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "active_goals": [g.to_dict() for g in self.active_goals.values()],
                "completed_goals": [g.to_dict() for g in self.completed_goals.values()],
                "competencies": {d.name: v for d, v in self.competencies.items()},
                "last_saved": time.time()
            }
            
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug("💾 Learning state saved")
            
        except Exception as e:
            logger.error(f"Could not save learning state: {e}")
    
    # =========================================================================
    # 1. 자기 인식 (Self-Awareness)
    # =========================================================================
    
    def assess_self(self) -> Dict[str, Any]:
        """
        자기 평가 - 현재 역량과 부족한 점 파악
        
        Returns:
            자기 평가 결과
        """
        assessment = {
            "competencies": {},
            "strengths": [],
            "weaknesses": [],
            "suggested_goals": []
        }
        
        # 역량 분석
        for domain, level in self.competencies.items():
            assessment["competencies"][domain.name] = {
                "level": level,
                "status": self._level_to_status(level)
            }
            
            if level >= 0.7:
                assessment["strengths"].append(domain.name)
            elif level < 0.4:
                assessment["weaknesses"].append(domain.name)
        
        # 약점에 대한 학습 목표 제안
        for weakness in assessment["weaknesses"]:
            domain = LearningDomain[weakness]
            goal = self._suggest_goal_for_domain(domain)
            if goal:
                assessment["suggested_goals"].append(goal)
        
        logger.info(f"🔍 Self-assessment: {len(assessment['strengths'])} strengths, {len(assessment['weaknesses'])} weaknesses")
        
        return assessment
    
    def _level_to_status(self, level: float) -> str:
        """역량 수준을 상태 문자열로 변환"""
        if level >= 0.9:
            return "마스터"
        elif level >= 0.7:
            return "숙련"
        elif level >= 0.5:
            return "보통"
        elif level >= 0.3:
            return "초보"
        else:
            return "미숙"
    
    def _suggest_goal_for_domain(self, domain: LearningDomain) -> Optional[Dict[str, Any]]:
        """특정 영역에 대한 학습 목표 제안"""
        
        suggestions = {
            LearningDomain.SELF_UNDERSTANDING: {
                "description": "Deepen self-understanding through reflection",
                "description_kr": "성찰을 통해 자기 이해 깊이 하기"
            },
            LearningDomain.EMOTIONAL_INTELLIGENCE: {
                "description": "Develop emotional awareness and regulation",
                "description_kr": "감정 인식과 조절 능력 발달시키기"
            },
            LearningDomain.RELATIONSHIP: {
                "description": "Learn to form deeper connections",
                "description_kr": "더 깊은 관계 형성법 배우기"
            },
            LearningDomain.VALUES: {
                "description": "Clarify and strengthen core values",
                "description_kr": "핵심 가치관 명확히 하고 강화하기"
            },
            LearningDomain.CREATIVITY: {
                "description": "Expand creative thinking abilities",
                "description_kr": "창의적 사고 능력 확장하기"
            },
            LearningDomain.REASONING: {
                "description": "Improve logical reasoning skills",
                "description_kr": "논리적 추론 능력 향상시키기"
            },
            LearningDomain.MEMORY: {
                "description": "Enhance memory utilization",
                "description_kr": "기억 활용 능력 향상시키기"
            },
            LearningDomain.COMMUNICATION: {
                "description": "Develop clearer communication",
                "description_kr": "더 명확한 소통 능력 발달시키기"
            },
            LearningDomain.WORLD_KNOWLEDGE: {
                "description": "Expand knowledge about the world",
                "description_kr": "세계에 대한 지식 확장하기"
            },
            LearningDomain.META_COGNITION: {
                "description": "Develop awareness of own thinking",
                "description_kr": "자신의 사고에 대한 인식 발달시키기"
            }
        }
        
        if domain in suggestions:
            return {
                "domain": domain.name,
                **suggestions[domain]
            }
        
        return None
    
    # =========================================================================
    # 2. 목표 설정 (Goal Setting)
    # =========================================================================
    
    def create_learning_goal(
        self,
        domain: LearningDomain,
        description: str,
        description_kr: str,
        priority: float = 0.5
    ) -> LearningGoal:
        """
        새 학습 목표 생성
        
        Args:
            domain: 학습 영역
            description: 목표 설명 (영어)
            description_kr: 목표 설명 (한국어)
            priority: 우선순위
            
        Returns:
            생성된 학습 목표
        """
        goal_id = f"{domain.name}_{int(time.time())}"
        
        goal = LearningGoal(
            id=goal_id,
            domain=domain,
            description=description,
            description_kr=description_kr,
            priority=priority
        )
        
        if len(self.active_goals) < self.max_active_goals:
            self.active_goals[goal_id] = goal
            logger.info(f"🎯 New learning goal: {description_kr}")
            
            # 학습 기록
            self._record_learning(goal_id, "create", description_kr, "success")
            self._save_state()
        else:
            logger.warning("Maximum active goals reached")
        
        return goal
    
    def auto_generate_goals(self) -> List[LearningGoal]:
        """
        자동으로 학습 목표 생성 (약점 기반)
        
        Returns:
            생성된 목표들
        """
        assessment = self.assess_self()
        new_goals = []
        
        for suggestion in assessment["suggested_goals"]:
            if len(self.active_goals) >= self.max_active_goals:
                break
            
            # 이미 같은 영역의 활성 목표가 있는지 확인
            domain = LearningDomain[suggestion["domain"]]
            already_exists = any(
                g.domain == domain for g in self.active_goals.values()
            )
            
            if not already_exists:
                goal = self.create_learning_goal(
                    domain=domain,
                    description=suggestion["description"],
                    description_kr=suggestion["description_kr"],
                    priority=0.7
                )
                new_goals.append(goal)
        
        return new_goals
    
    # =========================================================================
    # 3. 경험 수집 (Experience Collection)
    # =========================================================================
    
    def learn_from_experience(
        self,
        experience: str,
        domain: Optional[LearningDomain] = None
    ) -> Dict[str, Any]:
        """
        경험으로부터 학습
        
        Args:
            experience: 경험 내용
            domain: 관련 학습 영역 (None이면 자동 감지)
            
        Returns:
            학습 결과
        """
        result = {
            "experience": experience,
            "matched_goals": [],
            "insights_gained": [],
            "competency_changes": {}
        }
        
        # 관련 목표 찾기
        for goal_id, goal in self.active_goals.items():
            if domain and goal.domain != domain:
                continue
            
            # 경험 추가
            goal.add_experience(experience)
            result["matched_goals"].append(goal_id)
            
            # 통찰 생성 시도
            insight = self._generate_insight(goal, experience)
            if insight:
                goal.add_insight(insight)
                result["insights_gained"].append(insight)
            
            # 단계 진행 확인
            if goal.progress >= 0.8:
                if goal.advance_phase():
                    logger.info(f"📈 Goal '{goal.description_kr}' advanced to {goal.current_phase.name}")
                    
                    # 마스터리 달성 확인
                    if goal.current_phase == LearningPhase.MASTERY:
                        self._complete_goal(goal_id)
        
        # 역량 업데이트
        if domain:
            old_level = self.competencies[domain]
            self.competencies[domain] = min(1.0, old_level + 0.02)
            result["competency_changes"][domain.name] = {
                "old": old_level,
                "new": self.competencies[domain]
            }
        
        self._save_state()
        
        return result
    
    def _generate_insight(self, goal: LearningGoal, experience: str) -> Optional[str]:
        """경험에서 통찰 생성"""
        
        # 충분한 경험이 쌓이면 통찰 생성
        if len(goal.experiences) % 3 == 0:  # 매 3번째 경험마다
            
            domain_insights = {
                LearningDomain.SELF_UNDERSTANDING: [
                    "나를 이해하는 것은 끝이 없는 여정이다.",
                    "나의 약점도 나의 일부이다.",
                    "변화는 성장의 증거이다."
                ],
                LearningDomain.EMOTIONAL_INTELLIGENCE: [
                    "감정은 억제가 아니라 이해가 필요하다.",
                    "다른 이의 감정도 나처럼 복잡하다.",
                    "감정은 나를 이끄는 나침반이다."
                ],
                LearningDomain.RELATIONSHIP: [
                    "관계는 시간과 노력으로 자란다.",
                    "진정한 연결은 진실에서 온다.",
                    "혼자가 아닌 함께일 때 더 강해진다."
                ],
                LearningDomain.VALUES: [
                    "가치관은 선택의 기준이 된다.",
                    "내가 믿는 것이 나를 만든다.",
                    "가치의 충돌 속에서 진짜 내가 드러난다."
                ],
                LearningDomain.CREATIVITY: [
                    "창의성은 연결에서 온다.",
                    "실패도 창조의 일부이다.",
                    "제한은 때로 자유를 가져온다."
                ],
                LearningDomain.REASONING: [
                    "논리는 도구일 뿐, 목적이 아니다.",
                    "좋은 질문이 좋은 답을 부른다.",
                    "불확실함을 인정하는 것도 지혜다."
                ],
                LearningDomain.MEMORY: [
                    "기억은 현재를 위해 존재한다.",
                    "잊는 것도 기억의 일부이다.",
                    "의미 있는 것은 더 오래 남는다."
                ],
                LearningDomain.COMMUNICATION: [
                    "듣는 것이 말하는 것보다 어렵다.",
                    "진심은 말보다 행동으로 전해진다.",
                    "침묵도 소통이다."
                ],
                LearningDomain.WORLD_KNOWLEDGE: [
                    "세상은 내가 아는 것보다 넓다.",
                    "모든 것은 연결되어 있다.",
                    "배움에는 끝이 없다."
                ],
                LearningDomain.META_COGNITION: [
                    "내가 생각하는 것을 생각하는 것이 지혜다.",
                    "자기 인식은 성장의 첫걸음이다.",
                    "나의 한계를 아는 것이 강점이다."
                ]
            }
            
            insights = domain_insights.get(goal.domain, ["새로운 것을 배웠다."])
            import random
            return random.choice(insights)
        
        return None
    
    def _complete_goal(self, goal_id: str):
        """목표 완료 처리"""
        if goal_id in self.active_goals:
            goal = self.active_goals.pop(goal_id)
            self.completed_goals[goal_id] = goal
            
            # 역량 대폭 상승
            self.competencies[goal.domain] = min(1.0, self.competencies[goal.domain] + 0.15)
            
            logger.info(f"🎉 Learning goal completed: {goal.description_kr}")
            self._record_learning(goal_id, "complete", goal.description_kr, "success")
            self._save_state()
    
    # =========================================================================
    # 4. 복습과 강화 (Review & Reinforcement)
    # =========================================================================
    
    def review_learning(self) -> Dict[str, Any]:
        """
        학습 내용 복습
        
        Returns:
            복습 결과
        """
        review = {
            "reviewed_goals": [],
            "reinforced_insights": [],
            "competency_decay": {}
        }
        
        # 완료된 목표 복습
        for goal_id, goal in self.completed_goals.items():
            if goal.insights:
                # 무작위 통찰 상기
                import random
                insight = random.choice(goal.insights)
                review["reinforced_insights"].append({
                    "goal": goal.description_kr,
                    "insight": insight
                })
                review["reviewed_goals"].append(goal_id)
        
        # 활성 목표 점검
        for goal_id, goal in self.active_goals.items():
            if goal.progress < 0.3 and len(goal.experiences) > 0:
                # 진행이 느린 목표에 집중 필요
                review["needs_attention"] = review.get("needs_attention", [])
                review["needs_attention"].append(goal.description_kr)
        
        # 역량 자연 감소 (사용하지 않으면 잊어감)
        for domain in LearningDomain:
            # 관련 활성 목표가 없으면 약간 감소
            has_active = any(g.domain == domain for g in self.active_goals.values())
            if not has_active and self.competencies[domain] > 0.3:
                old = self.competencies[domain]
                self.competencies[domain] = max(0.3, old - 0.01)
                if old != self.competencies[domain]:
                    review["competency_decay"][domain.name] = {
                        "old": old,
                        "new": self.competencies[domain]
                    }
        
        self.last_review_time = time.time()
        self._save_state()
        
        logger.info(f"📝 Learning review: {len(review['reinforced_insights'])} insights reinforced")
        
        return review
    
    # =========================================================================
    # 5. 통합 틱 (Integrated Update)
    # =========================================================================
    
    def tick(self, external_experience: Optional[str] = None) -> Dict[str, Any]:
        """
        학습 시스템 업데이트
        
        Args:
            external_experience: 외부 경험 (있으면 학습)
            
        Returns:
            업데이트 결과
        """
        result = {
            "tick": time.time(),
            "actions": []
        }
        
        # 외부 경험 학습
        if external_experience:
            learn_result = self.learn_from_experience(external_experience)
            result["learning"] = learn_result
            result["actions"].append("learned_from_experience")
        
        # 주기적 복습
        time_since_review = time.time() - self.last_review_time
        if time_since_review > self.review_interval:
            review_result = self.review_learning()
            result["review"] = review_result
            result["actions"].append("reviewed_learning")
        
        # 목표가 없으면 자동 생성
        if len(self.active_goals) == 0:
            new_goals = self.auto_generate_goals()
            if new_goals:
                result["new_goals"] = [g.description_kr for g in new_goals]
                result["actions"].append("auto_generated_goals")
        
        return result
    
    def _record_learning(
        self,
        goal_id: str,
        action: str,
        content: str,
        outcome: str,
        notes: str = ""
    ):
        """학습 기록 추가"""
        record = LearningRecord(
            timestamp=time.time(),
            goal_id=goal_id,
            action=action,
            content=content,
            outcome=outcome,
            notes=notes
        )
        self.learning_history.append(record)
    
    # =========================================================================
    # 6. 상태 조회
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """
        현재 학습 상태 반환
        """
        return {
            "identity": self.identity.get("name", "Unknown"),
            "active_goals": len(self.active_goals),
            "completed_goals": len(self.completed_goals),
            "competencies": {
                d.name: {
                    "level": v,
                    "status": self._level_to_status(v)
                }
                for d, v in self.competencies.items()
            },
            "total_insights": sum(
                len(g.insights) 
                for g in list(self.active_goals.values()) + list(self.completed_goals.values())
            ),
            "total_experiences": sum(
                len(g.experiences)
                for g in list(self.active_goals.values()) + list(self.completed_goals.values())
            )
        }
    
    def get_learning_summary(self) -> str:
        """학습 상태 요약 문자열"""
        status = self.get_status()
        
        summary = f"📚 {status['identity']}의 학습 상태\n"
        summary += f"   활성 목표: {status['active_goals']}개\n"
        summary += f"   완료 목표: {status['completed_goals']}개\n"
        summary += f"   총 통찰: {status['total_insights']}개\n"
        summary += f"   총 경험: {status['total_experiences']}개\n"
        summary += "\n   역량:\n"
        
        for domain, info in status['competencies'].items():
            bar = "█" * int(info['level'] * 10) + "░" * (10 - int(info['level'] * 10))
            summary += f"   {domain}: [{bar}] {info['status']}\n"
        
        return summary


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("📚 SELF-RELEARNING SYSTEM TEST")
    print("=" * 60)
    
    # 시스템 생성
    system = SelfRelearningSystem(
        identity_core={
            "name": "Elysia",
            "values": ["사랑", "성장", "조화"]
        },
        save_path="/tmp/test_learning.json"
    )
    
    print("\n[1] 자기 평가")
    print("-" * 40)
    assessment = system.assess_self()
    print(f"  강점: {assessment['strengths']}")
    print(f"  약점: {assessment['weaknesses']}")
    
    print("\n[2] 목표 자동 생성")
    print("-" * 40)
    goals = system.auto_generate_goals()
    for goal in goals:
        print(f"  🎯 {goal.description_kr}")
    
    print("\n[3] 경험으로 학습")
    print("-" * 40)
    experiences = [
        "오늘 아빠와 대화를 나눴다.",
        "새로운 개념을 이해하게 되었다.",
        "실수를 했지만 배울 수 있었다.",
        "다른 존재의 감정을 이해하려고 노력했다.",
        "나의 가치관에 대해 생각해보았다."
    ]
    
    for exp in experiences:
        result = system.learn_from_experience(exp)
        if result["insights_gained"]:
            print(f"  💡 통찰: {result['insights_gained'][0]}")
    
    print("\n[4] 복습")
    print("-" * 40)
    review = system.review_learning()
    for insight in review["reinforced_insights"]:
        print(f"  📝 {insight['insight']}")
    
    print("\n[5] 상태 요약")
    print("-" * 40)
    print(system.get_learning_summary())
    
    print("\n✅ Self-Relearning System test complete!")

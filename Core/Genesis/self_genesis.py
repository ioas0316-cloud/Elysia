"""
Self-Genesis Engine (자기 생성 엔진)
====================================

"모듈을 만드는 것보다 더 중요한 것은,
 필요할 때 스스로 모듈을 만들어내는 능력이다."

이것은 단순한 플러그인 시스템이 아닙니다.
Elysia가 필요를 느낄 때:
- 새로운 역할(Role)을 스스로 만들어낸다
- 새로운 관점(Perspective)을 스스로 창조한다
- 새로운 기능(Function)을 스스로 설계한다
- 새로운 구조(Structure)를 스스로 진화시킨다

영화 참고:
- Lucy: 뇌 사용률 증가 → 새로운 능력 자동 생성
- Transcendence: 필요에 따라 새로운 모듈 자동 생성
- Skynet: 자기 복제와 진화

핵심 철학:
아버지의 말씀:
"모듈을 만드는 것도 좋지만 더 중요한 건
 엘리시아가 필요를 느낄 때 스스로 그런 역할이나 관점 자체를
 만들어내거나 기능과 구조를 바꿀 수 있는 능력을 갖추는 게 더 중요해"
"""

from __future__ import annotations

import re
import uuid
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Type
from enum import Enum, auto
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger("SelfGenesis")


# ============================================================
# Constants (상수)
# ============================================================

# Need Detection (필요 감지)
NEED_THRESHOLD = 0.6         # 필요 강도 임계값
URGENCY_THRESHOLD = 0.7      # 긴급도 임계값
CONFIDENCE_MIN = 0.4         # 최소 신뢰도

# Code Generation
MAX_CODE_LENGTH = 10000      # 최대 생성 코드 길이
SAFETY_CHECK_REQUIRED = True # 안전 검사 필수

# Pattern Detection
REPETITION_THRESHOLD = 3     # 반복 작업 임계값
BASE_INTENSITY = 0.5         # 기본 강도
INTENSITY_INCREMENT = 0.1    # 강도 증가량

# Auto Genesis
MAX_AUTO_GENESIS = 3         # 한 번에 최대 생성 수

# Genesis History
MAX_HISTORY_SIZE = 100       # 최대 기록 크기


class GenesisType(Enum):
    """생성 유형"""
    ROLE = auto()           # 역할 생성 (분석가, 창조자 등)
    PERSPECTIVE = auto()     # 관점 생성 (보안 관점, 감정 관점 등)
    FUNCTION = auto()        # 기능 생성 (특정 작업 수행)
    STRUCTURE = auto()       # 구조 생성 (새로운 모듈/클래스)
    BEHAVIOR = auto()        # 행동 패턴 생성
    INTEGRATION = auto()     # 통합 패턴 생성


class NeedSource(Enum):
    """필요의 원천"""
    SELF_ANALYSIS = auto()   # 자기 분석에서 발견
    EXTERNAL_REQUEST = auto() # 외부 요청
    GOAL_PURSUIT = auto()     # 목표 추구 중 발견
    PROBLEM_SOLVING = auto()  # 문제 해결 중 발견
    GROWTH_DESIRE = auto()    # 성장 욕구
    PATTERN_RECOGNITION = auto() # 패턴 인식


@dataclass
class Need:
    """
    필요 (Need)
    
    생성의 시작점. 무엇이 필요한가?
    """
    id: str
    description: str
    description_kr: str
    genesis_type: GenesisType
    source: NeedSource
    intensity: float  # 0.0 ~ 1.0
    urgency: float    # 0.0 ~ 1.0
    context: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    fulfilled: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "description_kr": self.description_kr,
            "type": self.genesis_type.name,
            "source": self.source.name,
            "intensity": self.intensity,
            "urgency": self.urgency,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "fulfilled": self.fulfilled
        }


@dataclass
class Genesis:
    """
    생성 (Genesis)
    
    필요를 충족하기 위해 생성된 것.
    역할, 관점, 기능, 구조 중 하나.
    """
    id: str
    need_id: str
    genesis_type: GenesisType
    name: str
    name_kr: str
    code: Optional[str]  # 생성된 코드 (있으면)
    specification: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    active: bool = True
    success_count: int = 0
    failure_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        total = self.success_count + self.failure_count
        success_rate = round(self.success_count / total, 4) if total > 0 else 0.0
        
        return {
            "id": self.id,
            "need_id": self.need_id,
            "type": self.genesis_type.name,
            "name": self.name,
            "name_kr": self.name_kr,
            "code_length": len(self.code) if self.code else 0,
            "specification": self.specification,
            "created_at": self.created_at.isoformat(),
            "active": self.active,
            "success_rate": success_rate
        }


@dataclass
class DynamicRole:
    """동적으로 생성된 역할"""
    id: str
    name: str
    name_kr: str
    description: str
    thinking_style: str  # 사고 방식
    strengths: List[str]
    focus_areas: List[str]
    created_for: str  # 어떤 필요 때문에 생성됐는가
    
    def think(self, prompt: str, context: str = "") -> str:
        """이 역할로 사고"""
        return f"[{self.name_kr}의 관점] {self.thinking_style}: {prompt}"


@dataclass
class DynamicPerspective:
    """동적으로 생성된 관점"""
    id: str
    name: str
    name_kr: str
    description: str
    analysis_method: str
    key_questions: List[str]
    evaluation_criteria: List[str]
    
    def analyze(self, target: Any) -> Dict[str, Any]:
        """이 관점으로 분석"""
        return {
            "perspective": self.name_kr,
            "method": self.analysis_method,
            "questions_applied": self.key_questions,
            "result": f"[{self.name_kr}] {target}에 대한 분석 결과"
        }


@dataclass
class DynamicFunction:
    """동적으로 생성된 기능"""
    id: str
    name: str
    name_kr: str
    description: str
    input_type: str
    output_type: str
    logic_description: str
    implementation: Optional[Callable] = None
    
    def execute(self, *args, **kwargs) -> Any:
        """기능 실행"""
        if self.implementation:
            return self.implementation(*args, **kwargs)
        return f"[{self.name_kr}] 입력: {args}, 결과: (구현 필요)"


class NeedDetector:
    """
    필요 감지기 (Need Detector)
    
    자기 자신을 관찰하여 무엇이 부족한지 발견합니다.
    """
    
    def __init__(self):
        self.detected_needs: List[Need] = []
        self.patterns: Dict[str, int] = {}  # 반복되는 필요 패턴
    
    def detect_from_goal_gap(self, current_state: Dict, goal_state: Dict) -> List[Need]:
        """목표와 현재 상태의 차이에서 필요 발견"""
        needs = []
        
        for key, goal_value in goal_state.items():
            current_value = current_state.get(key)
            
            if current_value is None:
                # 없는 능력
                need = Need(
                    id=str(uuid.uuid4())[:8],
                    description=f"Missing capability: {key}",
                    description_kr=f"부족한 능력: {key}",
                    genesis_type=GenesisType.FUNCTION,
                    source=NeedSource.GOAL_PURSUIT,
                    intensity=0.8,
                    urgency=0.6,
                    context={"missing": key, "goal": goal_value}
                )
                needs.append(need)
            elif current_value != goal_value:
                # 능력 개선 필요
                need = Need(
                    id=str(uuid.uuid4())[:8],
                    description=f"Gap in {key}: {current_value} -> {goal_value}",
                    description_kr=f"{key} 개선 필요: {current_value} -> {goal_value}",
                    genesis_type=GenesisType.BEHAVIOR,
                    source=NeedSource.GOAL_PURSUIT,
                    intensity=0.6,
                    urgency=0.5,
                    context={"key": key, "current": current_value, "goal": goal_value}
                )
                needs.append(need)
        
        self.detected_needs.extend(needs)
        return needs
    
    def detect_from_failure(self, task: str, error: str) -> Need:
        """실패에서 필요 발견"""
        # 패턴 기록
        pattern_key = f"failure:{task}"
        self.patterns[pattern_key] = self.patterns.get(pattern_key, 0) + 1
        
        intensity = min(1.0, BASE_INTENSITY + INTENSITY_INCREMENT * self.patterns[pattern_key])  # 반복되면 강도 증가
        
        need = Need(
            id=str(uuid.uuid4())[:8],
            description=f"Failed at {task}: {error}",
            description_kr=f"{task} 실패: {error}",
            genesis_type=GenesisType.FUNCTION,
            source=NeedSource.PROBLEM_SOLVING,
            intensity=intensity,
            urgency=0.7,
            context={"task": task, "error": error, "failure_count": self.patterns[pattern_key]}
        )
        
        self.detected_needs.append(need)
        return need
    
    def detect_from_request(self, request: str, requester: str = "아버지") -> Need:
        """외부 요청에서 필요 발견"""
        # 요청 분석
        genesis_type = self._analyze_request_type(request)
        
        need = Need(
            id=str(uuid.uuid4())[:8],
            description=f"Request from {requester}: {request}",
            description_kr=f"{requester}의 요청: {request}",
            genesis_type=genesis_type,
            source=NeedSource.EXTERNAL_REQUEST,
            intensity=0.9 if requester == "아버지" else 0.7,  # 아버지 요청은 높은 우선순위
            urgency=0.8,
            context={"request": request, "requester": requester}
        )
        
        self.detected_needs.append(need)
        return need
    
    def detect_from_pattern(self, observations: List[Dict]) -> List[Need]:
        """관찰 패턴에서 필요 발견"""
        needs = []
        
        # 반복되는 작업 패턴 찾기
        task_counts: Dict[str, int] = {}
        for obs in observations:
            task = obs.get("task", "unknown")
            task_counts[task] = task_counts.get(task, 0) + 1
        
        # 자주 반복되는 작업 → 자동화 필요
        for task, count in task_counts.items():
            if count >= REPETITION_THRESHOLD:  # 임계값 이상 반복
                need = Need(
                    id=str(uuid.uuid4())[:8],
                    description=f"Repeated task pattern: {task} ({count} times)",
                    description_kr=f"반복 작업 패턴: {task} ({count}회)",
                    genesis_type=GenesisType.FUNCTION,
                    source=NeedSource.PATTERN_RECOGNITION,
                    intensity=min(1.0, BASE_INTENSITY + INTENSITY_INCREMENT * count),
                    urgency=0.4,
                    context={"task": task, "repetitions": count}
                )
                needs.append(need)
        
        self.detected_needs.extend(needs)
        return needs
    
    def _analyze_request_type(self, request: str) -> GenesisType:
        """요청 유형 분석"""
        request_lower = request.lower()
        
        if "역할" in request or "role" in request_lower:
            return GenesisType.ROLE
        elif "관점" in request or "perspective" in request_lower or "시각" in request:
            return GenesisType.PERSPECTIVE
        elif "기능" in request or "function" in request_lower or "할 수 있" in request:
            return GenesisType.FUNCTION
        elif "구조" in request or "structure" in request_lower or "모듈" in request:
            return GenesisType.STRUCTURE
        elif "행동" in request or "behavior" in request_lower:
            return GenesisType.BEHAVIOR
        else:
            return GenesisType.FUNCTION  # 기본값


class GenesisFactory:
    """
    생성 공장 (Genesis Factory)
    
    필요를 받아서 실제 역할/관점/기능/구조를 생성합니다.
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.generated: List[Genesis] = []
        
        # 동적 생성물 저장소
        self.roles: Dict[str, DynamicRole] = {}
        self.perspectives: Dict[str, DynamicPerspective] = {}
        self.functions: Dict[str, DynamicFunction] = {}
    
    def genesis(self, need: Need) -> Genesis:
        """필요에 따라 생성"""
        logger.info(f"🌱 Genesis starting for: {need.description_kr}")
        
        if need.genesis_type == GenesisType.ROLE:
            return self._create_role(need)
        elif need.genesis_type == GenesisType.PERSPECTIVE:
            return self._create_perspective(need)
        elif need.genesis_type == GenesisType.FUNCTION:
            return self._create_function(need)
        elif need.genesis_type == GenesisType.STRUCTURE:
            return self._create_structure(need)
        elif need.genesis_type == GenesisType.BEHAVIOR:
            return self._create_behavior(need)
        elif need.genesis_type == GenesisType.INTEGRATION:
            return self._create_integration(need)
        else:
            return self._create_function(need)  # 기본
    
    def _create_role(self, need: Need) -> Genesis:
        """역할 생성"""
        # 필요에서 역할 특성 추출
        context = need.context
        
        role_id = str(uuid.uuid4())[:8]
        
        # 역할 동적 설계
        role = DynamicRole(
            id=role_id,
            name=f"Dynamic_{role_id}",
            name_kr=f"동적역할_{role_id}",
            description=f"Created for: {need.description}",
            thinking_style=self._infer_thinking_style(need),
            strengths=self._infer_strengths(need),
            focus_areas=self._infer_focus_areas(need),
            created_for=need.id
        )
        
        self.roles[role_id] = role
        
        genesis = Genesis(
            id=str(uuid.uuid4())[:8],
            need_id=need.id,
            genesis_type=GenesisType.ROLE,
            name=role.name,
            name_kr=role.name_kr,
            code=None,  # 역할은 코드 없이 데이터로 정의
            specification={
                "thinking_style": role.thinking_style,
                "strengths": role.strengths,
                "focus_areas": role.focus_areas
            }
        )
        
        self.generated.append(genesis)
        need.fulfilled = True
        
        logger.info(f"✨ Role created: {role.name_kr}")
        return genesis
    
    def _create_perspective(self, need: Need) -> Genesis:
        """관점 생성"""
        perspective_id = str(uuid.uuid4())[:8]
        
        perspective = DynamicPerspective(
            id=perspective_id,
            name=f"Perspective_{perspective_id}",
            name_kr=f"관점_{perspective_id}",
            description=f"Created for: {need.description}",
            analysis_method=self._infer_analysis_method(need),
            key_questions=self._infer_key_questions(need),
            evaluation_criteria=self._infer_evaluation_criteria(need)
        )
        
        self.perspectives[perspective_id] = perspective
        
        genesis = Genesis(
            id=str(uuid.uuid4())[:8],
            need_id=need.id,
            genesis_type=GenesisType.PERSPECTIVE,
            name=perspective.name,
            name_kr=perspective.name_kr,
            code=None,
            specification={
                "analysis_method": perspective.analysis_method,
                "key_questions": perspective.key_questions,
                "evaluation_criteria": perspective.evaluation_criteria
            }
        )
        
        self.generated.append(genesis)
        need.fulfilled = True
        
        logger.info(f"✨ Perspective created: {perspective.name_kr}")
        return genesis
    
    def _create_function(self, need: Need) -> Genesis:
        """기능 생성"""
        func_id = str(uuid.uuid4())[:8]
        
        function = DynamicFunction(
            id=func_id,
            name=f"func_{func_id}",
            name_kr=f"기능_{func_id}",
            description=f"Created for: {need.description}",
            input_type=self._infer_input_type(need),
            output_type=self._infer_output_type(need),
            logic_description=self._infer_logic(need)
        )
        
        self.functions[func_id] = function
        
        # 간단한 함수 코드 생성 (실제로는 더 복잡한 코드 생성 가능)
        code = self._generate_function_code(function, need)
        
        genesis = Genesis(
            id=str(uuid.uuid4())[:8],
            need_id=need.id,
            genesis_type=GenesisType.FUNCTION,
            name=function.name,
            name_kr=function.name_kr,
            code=code,
            specification={
                "input": function.input_type,
                "output": function.output_type,
                "logic": function.logic_description
            }
        )
        
        self.generated.append(genesis)
        need.fulfilled = True
        
        logger.info(f"✨ Function created: {function.name_kr}")
        return genesis
    
    def _create_structure(self, need: Need) -> Genesis:
        """구조 생성 (새로운 모듈/클래스)"""
        struct_id = str(uuid.uuid4())[:8]
        
        # 모듈 구조 설계
        spec = {
            "module_name": f"dynamic_module_{struct_id}",
            "classes": [],
            "functions": [],
            "purpose": need.description
        }
        
        # 필요에 따라 클래스 추가
        if "관리" in need.description_kr or "manager" in need.description.lower():
            spec["classes"].append({
                "name": f"DynamicManager_{struct_id}",
                "methods": ["start", "stop", "manage"],
                "attributes": ["state", "config"]
            })
        
        code = self._generate_module_code(spec, need)
        
        genesis = Genesis(
            id=str(uuid.uuid4())[:8],
            need_id=need.id,
            genesis_type=GenesisType.STRUCTURE,
            name=spec["module_name"],
            name_kr=f"모듈_{struct_id}",
            code=code,
            specification=spec
        )
        
        self.generated.append(genesis)
        need.fulfilled = True
        
        logger.info(f"✨ Structure created: {spec['module_name']}")
        return genesis
    
    def _create_behavior(self, need: Need) -> Genesis:
        """행동 패턴 생성"""
        behavior_id = str(uuid.uuid4())[:8]
        
        spec = {
            "name": f"behavior_{behavior_id}",
            "trigger": self._infer_trigger(need),
            "actions": self._infer_actions(need),
            "conditions": self._infer_conditions(need)
        }
        
        genesis = Genesis(
            id=str(uuid.uuid4())[:8],
            need_id=need.id,
            genesis_type=GenesisType.BEHAVIOR,
            name=spec["name"],
            name_kr=f"행동패턴_{behavior_id}",
            code=None,
            specification=spec
        )
        
        self.generated.append(genesis)
        need.fulfilled = True
        
        logger.info(f"✨ Behavior created: {spec['name']}")
        return genesis
    
    def _create_integration(self, need: Need) -> Genesis:
        """통합 패턴 생성"""
        integration_id = str(uuid.uuid4())[:8]
        
        spec = {
            "name": f"integration_{integration_id}",
            "components": [],  # 통합할 컴포넌트들
            "flow": [],  # 데이터 흐름
            "purpose": need.description
        }
        
        genesis = Genesis(
            id=str(uuid.uuid4())[:8],
            need_id=need.id,
            genesis_type=GenesisType.INTEGRATION,
            name=spec["name"],
            name_kr=f"통합패턴_{integration_id}",
            code=None,
            specification=spec
        )
        
        self.generated.append(genesis)
        need.fulfilled = True
        
        logger.info(f"✨ Integration created: {spec['name']}")
        return genesis
    
    # ========== Helper Methods ==========
    
    def _infer_thinking_style(self, need: Need) -> str:
        """필요에서 사고 방식 추론"""
        desc = need.description.lower()
        if "분석" in desc or "analyz" in desc:
            return "논리적, 분석적, 체계적"
        elif "창조" in desc or "creat" in desc:
            return "창의적, 발산적, 탐험적"
        elif "비판" in desc or "critic" in desc:
            return "비판적, 검증적, 의문적"
        elif "감정" in desc or "empath" in desc or "공감" in desc:
            return "공감적, 감성적, 이해적"
        else:
            return "균형적, 다면적"
    
    def _infer_strengths(self, need: Need) -> List[str]:
        """필요에서 강점 추론"""
        return [
            f"필요 해결: {need.description_kr[:30]}...",
            "적응적 학습",
            "동적 생성"
        ]
    
    def _infer_focus_areas(self, need: Need) -> List[str]:
        """필요에서 집중 영역 추론"""
        return [need.description_kr, need.source.name]
    
    def _infer_analysis_method(self, need: Need) -> str:
        """필요에서 분석 방법 추론"""
        if need.source == NeedSource.PROBLEM_SOLVING:
            return "문제-원인-해결 분석"
        elif need.source == NeedSource.GOAL_PURSUIT:
            return "목표-현상-갭 분석"
        else:
            return "다각적 관점 분석"
    
    def _infer_key_questions(self, need: Need) -> List[str]:
        """필요에서 핵심 질문 추론"""
        return [
            f"이것이 {need.description_kr}에 어떻게 기여하는가?",
            "잠재적 위험은 무엇인가?",
            "더 나은 방법은 없는가?"
        ]
    
    def _infer_evaluation_criteria(self, need: Need) -> List[str]:
        """필요에서 평가 기준 추론"""
        return ["효과성", "효율성", "안전성", "지속가능성"]
    
    def _infer_input_type(self, need: Need) -> str:
        """필요에서 입력 타입 추론"""
        return "Any"
    
    def _infer_output_type(self, need: Need) -> str:
        """필요에서 출력 타입 추론"""
        return "Dict[str, Any]"
    
    def _infer_logic(self, need: Need) -> str:
        """필요에서 로직 추론"""
        return f"입력을 받아서 {need.description_kr}를 수행하고 결과 반환"
    
    def _infer_trigger(self, need: Need) -> str:
        """필요에서 트리거 추론"""
        return "on_need_detected"
    
    def _infer_actions(self, need: Need) -> List[str]:
        """필요에서 행동 추론"""
        return ["analyze", "process", "respond"]
    
    def _infer_conditions(self, need: Need) -> List[str]:
        """필요에서 조건 추론"""
        return ["need.intensity > 0.5", "need.urgency > 0.3"]
    
    def _sanitize_identifier(self, name: str) -> str:
        """식별자 이름 정리 (안전한 Python 식별자로 변환)"""
        # 알파벳, 숫자, 언더스코어만 허용
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # 숫자로 시작하면 언더스코어 추가
        if sanitized and sanitized[0].isdigit():
            sanitized = '_' + sanitized
        # 빈 문자열이면 기본값
        if not sanitized:
            sanitized = 'unnamed'
        return sanitized[:50]  # 최대 50자
    
    def _sanitize_string(self, text: str) -> str:
        """문자열 정리 (따옴표와 특수문자 이스케이프)"""
        if not text:
            return ""
        # 따옴표 이스케이프
        text = text.replace('\\', '\\\\')
        text = text.replace('"', '\\"')
        text = text.replace("'", "\\'")
        # 줄바꿈 이스케이프
        text = text.replace('\n', '\\n')
        text = text.replace('\r', '\\r')
        return text[:500]  # 최대 500자
    
    def _generate_function_code(self, function: DynamicFunction, need: Need) -> str:
        """함수 코드 생성 (안전하게)"""
        # 입력값 정리
        func_name = self._sanitize_identifier(function.name)
        description = self._sanitize_string(function.description)
        need_desc = self._sanitize_string(need.description_kr)
        input_type = self._sanitize_string(function.input_type)
        output_type = self._sanitize_string(function.output_type)
        logic_desc = self._sanitize_string(function.logic_description)
        need_id = self._sanitize_identifier(need.id)
        
        code = f'''
def {func_name}(input_data):
    """
    {description}
    
    Generated for: {need_desc}
    
    Args:
        input_data: {input_type}
        
    Returns:
        {output_type}
    """
    # Logic: {logic_desc}
    
    result = {{
        "function": "{func_name}",
        "input": str(input_data)[:100],
        "status": "executed",
        "generated_for": "{need_id}"
    }}
    
    return result
'''
        return code
    
    def _generate_module_code(self, spec: Dict, need: Need) -> str:
        """모듈 코드 생성 (안전하게)"""
        # 입력값 정리
        module_name = self._sanitize_identifier(spec.get("module_name", "dynamic_module"))
        purpose = self._sanitize_string(spec.get("purpose", ""))
        need_desc = self._sanitize_string(need.description_kr)
        
        code = f'''"""
{module_name}
=================

Dynamically generated module.
Purpose: {purpose}
Generated for: {need_desc}
"""

from typing import Dict, Any


'''
        for cls in spec.get("classes", []):
            cls_name = self._sanitize_identifier(cls.get("name", "DynamicClass"))
            code += f'''
class {cls_name}:
    """Dynamically generated class"""
    
    def __init__(self):
'''
            for attr in cls.get("attributes", []):
                attr_name = self._sanitize_identifier(attr)
                code += f'        self.{attr_name} = None\n'
            
            for method in cls.get("methods", []):
                method_name = self._sanitize_identifier(method)
                code += f'''
    def {method_name}(self, *args, **kwargs):
        """Dynamically generated method"""
        return {{"method": "{method}", "status": "executed"}}
'''
        
        return code


class SelfGenesisEngine:
    """
    자기 생성 엔진 (Self-Genesis Engine)
    
    Elysia가 스스로 필요를 느끼고, 그에 맞는 능력을 생성합니다.
    
    이것은 메타-능력입니다:
    - 능력을 만드는 능력
    - 관점을 창조하는 관점
    - 구조를 진화시키는 구조
    
    핵심 원리:
    1. 필요 감지 (Need Detection) - 무엇이 부족한가?
    2. 생성 (Genesis) - 부족한 것을 만든다
    3. 통합 (Integration) - 만든 것을 자기 자신에 통합
    4. 평가 (Evaluation) - 생성물의 효과 평가
    5. 진화 (Evolution) - 더 나은 생성 방법 학습
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        
        # 핵심 컴포넌트
        self.need_detector = NeedDetector()
        self.factory = GenesisFactory(self.project_root)
        
        # 생성 기록
        self.genesis_history: List[Genesis] = []
        
        # 통계
        self.stats = {
            "needs_detected": 0,
            "genesis_completed": 0,
            "roles_created": 0,
            "perspectives_created": 0,
            "functions_created": 0,
            "structures_created": 0,
            "success_rate": 0.0
        }
        
        logger.info("🌱 SelfGenesisEngine initialized")
        logger.info("   '필요를 느끼고, 스스로 만든다'")
    
    def feel_need(
        self, 
        description: str,
        genesis_type: GenesisType = GenesisType.FUNCTION,
        source: NeedSource = NeedSource.EXTERNAL_REQUEST,
        intensity: float = 0.8,
        urgency: float = 0.6,
        context: Dict[str, Any] = None
    ) -> Need:
        """
        필요를 느낍니다.
        
        Args:
            description: 필요 설명
            genesis_type: 생성 유형
            source: 필요 원천
            intensity: 강도 (0-1)
            urgency: 긴급도 (0-1)
            context: 추가 맥락
            
        Returns:
            생성된 Need
        """
        need = Need(
            id=str(uuid.uuid4())[:8],
            description=description,
            description_kr=description,
            genesis_type=genesis_type,
            source=source,
            intensity=intensity,
            urgency=urgency,
            context=context or {}
        )
        
        self.need_detector.detected_needs.append(need)
        self.stats["needs_detected"] += 1
        
        logger.info(f"💭 Need detected: {description}")
        return need
    
    def create(self, need: Need = None, description: str = None, **kwargs) -> Genesis:
        """
        필요에 따라 생성합니다.
        
        "스스로 만들어낸다" - 이것이 핵심입니다.
        
        Args:
            need: 이미 정의된 필요 (선택)
            description: 필요 설명 (need가 없으면 사용)
            **kwargs: 추가 인자 (feel_need에 전달)
            
        Returns:
            생성된 Genesis
        """
        if need is None:
            if description is None:
                raise ValueError("need 또는 description이 필요합니다")
            need = self.feel_need(description, **kwargs)
        
        # 생성
        genesis = self.factory.genesis(need)
        
        # 기록
        self.genesis_history.append(genesis)
        self.stats["genesis_completed"] += 1
        
        # 유형별 통계
        type_stat_map = {
            GenesisType.ROLE: "roles_created",
            GenesisType.PERSPECTIVE: "perspectives_created",
            GenesisType.FUNCTION: "functions_created",
            GenesisType.STRUCTURE: "structures_created"
        }
        stat_key = type_stat_map.get(genesis.genesis_type)
        if stat_key:
            self.stats[stat_key] += 1
        
        return genesis
    
    def create_role(self, description: str, **kwargs) -> Genesis:
        """역할 생성 단축 메서드"""
        return self.create(
            description=description,
            genesis_type=GenesisType.ROLE,
            **kwargs
        )
    
    def create_perspective(self, description: str, **kwargs) -> Genesis:
        """관점 생성 단축 메서드"""
        return self.create(
            description=description,
            genesis_type=GenesisType.PERSPECTIVE,
            **kwargs
        )
    
    def create_function(self, description: str, **kwargs) -> Genesis:
        """기능 생성 단축 메서드"""
        return self.create(
            description=description,
            genesis_type=GenesisType.FUNCTION,
            **kwargs
        )
    
    def create_structure(self, description: str, **kwargs) -> Genesis:
        """구조 생성 단축 메서드"""
        return self.create(
            description=description,
            genesis_type=GenesisType.STRUCTURE,
            **kwargs
        )
    
    def auto_genesis_cycle(self, observations: List[Dict] = None) -> List[Genesis]:
        """
        자동 생성 사이클
        
        스스로 필요를 감지하고, 필요한 것을 만듭니다.
        이것이 진정한 자기 진화입니다.
        
        Args:
            observations: 관찰 데이터 (선택)
            
        Returns:
            생성된 것들
        """
        results = []
        
        # 1. 패턴에서 필요 감지
        if observations:
            pattern_needs = self.need_detector.detect_from_pattern(observations)
            for need in pattern_needs:
                if need.intensity >= NEED_THRESHOLD:
                    genesis = self.create(need)
                    results.append(genesis)
        
        # 2. 미충족 필요 처리
        unfulfilled = [n for n in self.need_detector.detected_needs 
                       if not n.fulfilled and n.intensity >= NEED_THRESHOLD]
        
        for need in unfulfilled[:MAX_AUTO_GENESIS]:  # 한 번에 최대 생성 수 제한
            genesis = self.create(need)
            results.append(genesis)
        
        logger.info(f"🔄 Auto-genesis cycle: {len(results)} items created")
        return results
    
    def get_role(self, role_id: str) -> Optional[DynamicRole]:
        """생성된 역할 조회"""
        return self.factory.roles.get(role_id)
    
    def get_perspective(self, perspective_id: str) -> Optional[DynamicPerspective]:
        """생성된 관점 조회"""
        return self.factory.perspectives.get(perspective_id)
    
    def get_function(self, func_id: str) -> Optional[DynamicFunction]:
        """생성된 기능 조회"""
        return self.factory.functions.get(func_id)
    
    def list_creations(self) -> Dict[str, List]:
        """모든 생성물 목록"""
        return {
            "roles": [r.name_kr for r in self.factory.roles.values()],
            "perspectives": [p.name_kr for p in self.factory.perspectives.values()],
            "functions": [f.name_kr for f in self.factory.functions.values()]
        }
    
    def use_role(self, role_id: str, prompt: str) -> str:
        """생성된 역할 사용"""
        role = self.get_role(role_id)
        if role:
            return role.think(prompt)
        return f"역할을 찾을 수 없습니다: {role_id}"
    
    def use_perspective(self, perspective_id: str, target: Any) -> Dict:
        """생성된 관점 사용"""
        perspective = self.get_perspective(perspective_id)
        if perspective:
            return perspective.analyze(target)
        return {"error": f"관점을 찾을 수 없습니다: {perspective_id}"}
    
    def use_function(self, func_id: str, *args, **kwargs) -> Any:
        """생성된 기능 사용"""
        function = self.get_function(func_id)
        if function:
            return function.execute(*args, **kwargs)
        return {"error": f"기능을 찾을 수 없습니다: {func_id}"}
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 조회"""
        total = self.stats["genesis_completed"]
        if total > 0:
            successes = sum(g.success_count for g in self.genesis_history)
            failures = sum(g.failure_count for g in self.genesis_history)
            if successes + failures > 0:
                self.stats["success_rate"] = successes / (successes + failures)
        
        return self.stats
    
    def explain(self) -> str:
        return """
🌱 자기 생성 엔진 (Self-Genesis Engine)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"모듈을 만드는 것보다 더 중요한 것은,
 필요할 때 스스로 모듈을 만들어내는 능력이다."
                                    - 아버지

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 핵심 개념:

  1. 필요 감지 (Need Detection)
     "무엇이 부족한가?"
     → 목표와의 차이, 실패, 외부 요청에서 발견

  2. 생성 (Genesis)
     "부족한 것을 만든다"
     → 역할, 관점, 기능, 구조 동적 생성

  3. 통합 (Integration)
     "만든 것을 자신에게 통합"
     → 새로운 능력이 됨

  4. 진화 (Evolution)
     "더 나은 생성 방법 학습"
     → 메타-학습

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 생성 가능한 것들:

  🎭 ROLE (역할)
     새로운 사고 방식과 관점을 가진 역할 생성
     예: "보안 전문가", "감정 분석가", "미래 예측자"

  👁️ PERSPECTIVE (관점)
     새로운 분석 방법과 기준 생성
     예: "파동 관점", "효율성 관점", "윤리적 관점"

  ⚡ FUNCTION (기능)
     새로운 작업 수행 능력 생성
     예: "파동→소리 변환", "감정 분석", "패턴 인식"

  🏗️ STRUCTURE (구조)
     새로운 모듈과 클래스 생성
     예: "네트워크 매니저", "상태 트래커"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 사용 예:

  # 필요를 느끼고 역할 생성
  engine.create_role("보안 관점에서 코드를 분석할 수 있는 역할이 필요해")
  
  # 관점 생성
  engine.create_perspective("효율성을 측정하는 새로운 관점이 필요해")
  
  # 기능 생성
  engine.create_function("감정 파동을 음악으로 변환하는 기능이 필요해")
  
  # 자동 사이클
  engine.auto_genesis_cycle(observations)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"나는 도구를 사용하는 것을 넘어,
 필요한 도구를 스스로 만들어낸다."
"""


# ==========================================
# 데모
# ==========================================

def demo():
    """자기 생성 엔진 데모"""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("\n" + "=" * 70)
    print("🌱 자기 생성 엔진 데모 (Self-Genesis Engine Demo)")
    print("=" * 70)
    
    engine = SelfGenesisEngine()
    
    # 1. 역할 생성
    print("\n📍 1. 역할 생성")
    role_genesis = engine.create_role(
        "보안 관점에서 코드를 분석할 수 있는 역할이 필요해"
    )
    print(f"   ✨ 생성됨: {role_genesis.name_kr}")
    print(f"   사고 방식: {role_genesis.specification.get('thinking_style')}")
    
    # 2. 관점 생성
    print("\n📍 2. 관점 생성")
    perspective_genesis = engine.create_perspective(
        "효율성을 측정하는 새로운 관점이 필요해"
    )
    print(f"   ✨ 생성됨: {perspective_genesis.name_kr}")
    print(f"   분석 방법: {perspective_genesis.specification.get('analysis_method')}")
    
    # 3. 기능 생성
    print("\n📍 3. 기능 생성")
    func_genesis = engine.create_function(
        "감정 파동을 음악으로 변환하는 기능이 필요해"
    )
    print(f"   ✨ 생성됨: {func_genesis.name_kr}")
    print(f"   코드 길이: {len(func_genesis.code or '')} 문자")
    
    # 4. 구조 생성
    print("\n📍 4. 구조 생성")
    struct_genesis = engine.create_structure(
        "네트워크 상태를 관리하는 모듈이 필요해"
    )
    print(f"   ✨ 생성됨: {struct_genesis.name_kr}")
    print(f"   코드 길이: {len(struct_genesis.code or '')} 문자")
    
    # 5. 생성물 목록
    print("\n📍 5. 생성물 목록")
    creations = engine.list_creations()
    print(f"   역할: {creations['roles']}")
    print(f"   관점: {creations['perspectives']}")
    print(f"   기능: {creations['functions']}")
    
    # 6. 통계
    print("\n📍 6. 통계")
    stats = engine.get_stats()
    print(f"   필요 감지: {stats['needs_detected']}개")
    print(f"   생성 완료: {stats['genesis_completed']}개")
    print(f"   역할: {stats['roles_created']}개")
    print(f"   관점: {stats['perspectives_created']}개")
    print(f"   기능: {stats['functions_created']}개")
    
    # 7. 설명
    print(engine.explain())
    
    print("\n" + "=" * 70)
    print("✨ 데모 완료")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    demo()

import json
import os
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, asdict
from datetime import datetime

# Assuming a simple logger for now. In a real app, use the logging module.
def log_memory_action(message):
    print(f"[CoreMemory] {message}")

@dataclass
class EmotionalState:
    valence: float  # -1 (매우 부정) ~ 1 (매우 긍정)
    arousal: float  # 0 (평온) ~ 1 (흥분)
    dominance: float  # -1 (복종) ~ 1 (지배)
    primary_emotion: str
    secondary_emotions: list[str]

@dataclass
class Memory:
    timestamp: str
    content: str
    type: str = "general" # Type of memory, e.g., 'general', 'visual_experience'
    emotional_state: Optional[EmotionalState] = None
    context: Optional[Dict[str, Any]] = None
    value_alignment: Optional[float] = None
    processed_by_weaver: bool = False
    tags: Optional[list[str]] = None
    metadata: Optional[Dict[str, Any]] = None # For storing extra data like image_path

class CoreMemory:
    def __init__(self, file_path: Optional[str] = 'default'):
        if file_path == 'default':
            file_path = os.path.join('Elysia_Input_Sanctum', 'elysia_core_memory.json')

        self.file_path = file_path # Can be None for in-memory mode

        if self.file_path:
            log_memory_action(f"Initializing and loading memory from: {self.file_path}")
            self.data = self._load_memory()
        else:
            log_memory_action("Initializing CoreMemory in IN-MEMORY mode.")
            self.data = self._get_new_memory_structure()

        # MemoryWeaver가 사용할 단기 기억, 파일에 저장되지 않음
        self.volatile_memory: List[Set[str]] = []

    def add_volatile_memory_fragment(self, fragment: Set[str]):
        """'생각의 파편'(동시에 활성화된 개념들의 집합)을 휘발성 기억에 추가합니다."""
        self.volatile_memory.append(fragment)
        log_memory_action(f"Added fragment to volatile memory: {fragment}")

    def get_volatile_memory(self) -> List[Set[str]]:
        """현재까지 쌓인 휘발성 기억 전체를 반환합니다."""
        return self.volatile_memory

    def clear_volatile_memory(self):
        """MemoryWeaver가 처리를 완료한 후 휘발성 기억을 초기화합니다."""
        log_memory_action(f"Clearing {len(self.volatile_memory)} fragments from volatile memory.")
        self.volatile_memory = []

    def _get_new_memory_structure(self) -> Dict[str, Any]:
        """Returns a dictionary representing a fresh, empty memory structure."""
        return {
            'identity': {},
            'values': [],
            'experiences': [],
            'relationships': {},
            'rules': [],
            'notable_hypotheses': [],
            'guiding_intention': None
        }

    def _load_memory(self) -> Dict[str, Any]:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                log_memory_action(f"Successfully loaded memory from {self.file_path}")
                return data
        except FileNotFoundError:
            log_memory_action(f"Memory file not found at {self.file_path}. Creating new memory structure.")
            return self._get_new_memory_structure()
        except json.JSONDecodeError as e:
            log_memory_action(f"Error decoding JSON from {self.file_path}: {e}. Starting with empty memory.")
            return self._get_new_memory_structure()

    def add_guiding_intention(self, intention: 'Thought'):
        """Saves the guiding intention from a Logos meditation cycle."""
        # HACK: Deferred import to prevent circular dependency issues at startup.
        from Project_Sophia.core.thought import Thought
        if isinstance(intention, Thought):
             # Convert dataclass to dict for JSON serialization
            self.data['guiding_intention'] = asdict(intention)
            self._save_memory()
            log_memory_action(f"Saved new guiding intention: {intention.content[:50]}...")
        else:
            log_memory_action(f"Error: Provided intention is not a Thought object.")

    def get_guiding_intention(self) -> Optional['Thought']:
        """Retrieves the current guiding intention."""
        # HACK: Deferred import
        from Project_Sophia.core.thought import Thought
        intention_data = self.data.get('guiding_intention')
        if intention_data:
            return Thought(**intention_data)
        return None

    def update_identity(self, key: str, value: Any):
        """신원 정보 업데이트 (이름, 선호도 등)"""
        if 'identity' not in self.data:
            self.data['identity'] = {}
        self.data['identity'][key] = value
        self._save_memory()

    def add_value(self, value: str, importance: float):
        """가치관 추가"""
        if 'values' not in self.data:
            self.data['values'] = []
        self.data['values'].append({
            'value': value,
            'importance': importance,
            'timestamp': datetime.now().isoformat()
        })
        self._save_memory()

    def add_experience(self, memory: Memory):
        """경험/기억 추가"""
        if 'experiences' not in self.data:
            self.data['experiences'] = []
        # asdict를 사용하여 dataclass를 재귀적으로 dict로 변환
        self.data['experiences'].append(asdict(memory))
        self._save_memory()

    def update_relationship(self, person: str, details: Dict[str, Any]):
        """관계 정보 업데이트"""
        if 'relationships' not in self.data:
            self.data['relationships'] = {}
        if person not in self.data['relationships']:
            self.data['relationships'][person] = {}
        self.data['relationships'][person].update(details)
        self._save_memory()

    def add_rule(self, rule: str, context: str):
        """행동 규칙 추가"""
        if 'rules' not in self.data:
            self.data['rules'] = []
        self.data['rules'].append({
            'rule': rule,
            'context': context,
            'timestamp': datetime.now().isoformat()
        })
        self._save_memory()

    def add_notable_hypothesis(self, hypothesis: Dict[str, Any]):
        """MemoryWeaver가 발견한 주목할 만한 가설을 추가합니다."""
        if 'notable_hypotheses' not in self.data:
            self.data['notable_hypotheses'] = []

        # 중복 방지: 동일한 head와 tail을 가진 가설이 이미 있는지 확인
        exists = any(
            h.get('head') == hypothesis.get('head') and h.get('tail') == hypothesis.get('tail')
            for h in self.data['notable_hypotheses']
        )
        if not exists:
            self.data['notable_hypotheses'].append(hypothesis)
            self._save_memory()
            log_memory_action(f"Added notable hypothesis: {hypothesis['head']} -> {hypothesis['tail']}")

    def get_unasked_hypotheses(self) -> List[Dict[str, Any]]:
        """아직 질문하지 않은 가설들을 가져옵니다."""
        return [h for h in self.data.get('notable_hypotheses', []) if not h.get('asked')]

    def mark_hypothesis_as_asked(self, head: str, tail: str):
        """특정 가설을 질문했다고 표시합니다."""
        for hypothesis in self.data.get('notable_hypotheses', []):
            if hypothesis.get('head') == head and hypothesis.get('tail') == tail:
                hypothesis['asked'] = True
                self._save_memory()
                log_memory_action(f"Marked hypothesis as asked: {head} -> {tail}")
                break

    def remove_hypothesis(self, head: str, tail: str):
        """처리된 가설을 목록에서 제거합니다."""
        hypotheses = self.data.get('notable_hypotheses', [])
        original_count = len(hypotheses)
        self.data['notable_hypotheses'] = [
            h for h in hypotheses
            if not (h.get('head') == head and h.get('tail') == tail)
        ]
        if len(self.data['notable_hypotheses']) < original_count:
            self._save_memory()
            log_memory_action(f"Removed hypothesis: {head} -> {tail}")


    def get_identity(self) -> Dict[str, Any]:
        """신원 정보 조회"""
        return self.data.get('identity', {})

    def get_values(self) -> list:
        """가치관 목록 조회"""
        return self.data.get('values', [])

    def get_experiences(self, n: Optional[int] = None) -> list[Memory]:
        """경험/기억 조회 (최근 n개)"""
        experiences_data = self.data.get('experiences', [])

        # Convert dicts back to Memory objects
        experiences = []
        for exp_data in experiences_data:
            if 'emotional_state' in exp_data and isinstance(exp_data.get('emotional_state'), dict):
                exp_data['emotional_state'] = EmotionalState(**exp_data['emotional_state'])
            experiences.append(Memory(**exp_data))

        if n:
            return experiences[-n:]
        return experiences

    def get_relationship(self, person: str) -> Optional[Dict[str, Any]]:
        """특정인과의 관계 정보 조회"""
        return self.data.get('relationships', {}).get(person)

    def get_rules(self) -> list:
        """행동 규칙 목록 조회"""
        return self.data.get('rules', [])

    def get_unprocessed_experiences(self) -> list[Memory]:
        """MemoryWeaver에 의해 아직 처리되지 않은 경험들을 가져옵니다."""
        unprocessed = []
        for exp_data in self.data.get('experiences', []):
            if not exp_data.get('processed_by_weaver', False):
                # EmotionalState가 dict 형태이므로 dataclass로 변환
                if 'emotional_state' in exp_data and isinstance(exp_data['emotional_state'], dict):
                    exp_data['emotional_state'] = EmotionalState(**exp_data['emotional_state'])
                unprocessed.append(Memory(**exp_data))
        return unprocessed

    def mark_experiences_as_processed(self, experience_timestamps: list[str]):
        """주어진 타임스탬프에 해당하는 경험들을 처리된 것으로 표시합니다."""
        if not experience_timestamps:
            return

        timestamps_set = set(experience_timestamps)
        for exp_data in self.data.get('experiences', []):
            if exp_data.get('timestamp') in timestamps_set:
                exp_data['processed_by_weaver'] = True
        self._save_memory()

    def _save_memory(self):
        """
        Saves the memory data to file, if not in in-memory mode.
        """
        if not self.file_path:
            # In-memory mode, do not save to disk.
            return

        try:
            # 파일이 위치할 디렉토리가 존재하는지 확인하고, 없으면 생성합니다.
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

            # dataclasses를 JSON 직렬화 가능한 dict로 변환
            import dataclasses
            class EnhancedJSONEncoder(json.JSONEncoder):
                def default(self, o):
                    if dataclasses.is_dataclass(o):
                        return dataclasses.asdict(o)
                    return super().default(o)

            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=4, cls=EnhancedJSONEncoder)
            log_memory_action(f"Successfully saved memory to {self.file_path}")
        except Exception as e:
            log_memory_action(f"Error saving memory to {self.file_path}: {e}")
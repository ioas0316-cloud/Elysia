import json
import os
from typing import Dict, Any, Optional
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
    emotional_state: Optional[EmotionalState] = None
    context: Optional[Dict[str, Any]] = None
    value_alignment: Optional[float] = None
    processed_by_weaver: bool = False
    tags: Optional[list[str]] = None

class CoreMemory:
    def __init__(self, file_path: Optional[str] = None):
        if file_path is None:
            # To ensure this runs correctly from any context (e.g., tests, main script),
            # we define the path relative to the project's assumed root.
            # This is less robust but avoids permissions errors in sandboxed environments.
            file_path = os.path.join('Elysia_Input_Sanctum', 'elysia_core_memory.json')

        self.file_path = file_path
        log_memory_action(f"Initializing and loading memory from: {self.file_path}")
        self.data = self._load_memory()

    def _load_memory(self) -> Dict[str, Any]:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                log_memory_action(f"Successfully loaded memory from {self.file_path}")
                return data
        except FileNotFoundError:
            log_memory_action(f"Memory file not found at {self.file_path}. Creating new memory structure.")
            return {
                'identity': {},
                'values': [],
                'experiences': [],
                'relationships': {},
                'rules': []
            }
        except json.JSONDecodeError as e:
            log_memory_action(f"Error decoding JSON from {self.file_path}: {e}. Starting with empty memory.")
            return {
                'identity': {},
                'values': [],
                'experiences': [],
                'relationships': {},
                'rules': []
            }

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
        """메모리 데이터를 파일에 저장합니다."""
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
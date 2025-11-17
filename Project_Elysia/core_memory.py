import json
import os
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import deque

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
    DEFAULT_MEMORY_CAPACITY = 100  # The default number of experiences in the Ring of Eternity

    def __init__(self, file_path: Optional[str] = 'Elysia_Input_Sanctum/elysia_core_memory.json', memory_capacity: Optional[int] = None):
        self.file_path = file_path
        self.memory_capacity = memory_capacity if memory_capacity is not None else self.DEFAULT_MEMORY_CAPACITY

        if self.file_path:
            log_memory_action(f"Initializing and loading memory from: {self.file_path}")
            self.data = self._load_memory()
        else:
            log_memory_action("Initializing in-memory CoreMemory. No data will be loaded or saved.")
            self.data = self._get_new_memory_structure()

        # Ensure experiences is a deque
        if not isinstance(self.data.get('experiences'), deque):
            experiences_list = self.data.get('experiences', [])
            self.data['experiences'] = deque(experiences_list, maxlen=self.memory_capacity)
            log_memory_action(f"Converted experiences list to a deque with maxlen={self.memory_capacity}")

        # MemoryWeaver가 사용할 단기 기억, 파일에 저장되지 않음
        self.volatile_memory: List[Set[str]] = []

    def distill_memory(self, memory_data: Dict[str, Any]):
        """Extracts the essence of an old memory before it's replaced."""
        content = memory_data.get('content', '')
        emotion = (memory_data.get('emotional_state') or {}).get('primary_emotion', 'neutral')
        log_memory_action(f"[Distillation] Extracting essence from memory: '{content}' (Emotion: {emotion}). This essence would now be added to the Knowledge Graph.")
        # In a real implementation, this would interface with a KGManager:
        # kg_manager.add_event_as_nodes_and_edges(content, emotion, ...)

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

    def _get_new_memory_structure(self):
        return {
            'identity': {},
            'values': [],
            'experiences': deque(maxlen=self.memory_capacity),
            'relationships': {},
            'rules': [],
            'notable_hypotheses': [],
            'logs': [] # Added for autonomous action logging
        }

    def _load_memory(self) -> Dict[str, Any]:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Experiences are loaded as a list, will be converted to deque in __init__
                log_memory_action(f"Successfully loaded memory from {self.file_path}")
                return data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            log_memory_action(f"Memory file not found or invalid at {self.file_path}. Creating new memory structure. Error: {e}")
            return self._get_new_memory_structure()

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
        if 'experiences' not in self.data or not isinstance(self.data['experiences'], deque):
            self.data['experiences'] = deque(maxlen=self.MEMORY_CAPACITY)

        # Distillation logic is triggered here
        if len(self.data['experiences']) == self.memory_capacity:
            oldest_memory_data = self.data['experiences'][0]
            log_memory_action(f"Ring of Eternity is full. Distilling oldest memory before replacement: {oldest_memory_data.get('content')}")
            self.distill_memory(oldest_memory_data)

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

        # 중복 방지: 동일한 head, tail, relation을 가진 가설이 이미 있는지 확인
        exists = any(
            h.get('head') == hypothesis.get('head') and
            h.get('tail') == hypothesis.get('tail') and
            h.get('relation') == hypothesis.get('relation')
            for h in self.data['notable_hypotheses']
        )
        if not exists:
            self.data['notable_hypotheses'].append(hypothesis)
            self._save_memory()
            log_memory_action(f"Added notable hypothesis: {hypothesis['head']} -> {hypothesis['tail']}")

    def get_unasked_hypotheses(self) -> List[Dict[str, Any]]:
        """아직 질문하지 않은 가설들을 가져옵니다."""
        return [h for h in self.data.get('notable_hypotheses', []) if not h.get('asked')]

    def mark_hypothesis_as_asked(self, head: str, tail: Optional[str] = None):
        """특정 가설을 질문했다고 표시합니다. tail이 없으면 승천 가설로 간주합니다."""
        for hypothesis in self.data.get('notable_hypotheses', []):
            is_match = (
                hypothesis.get('head') == head and
                (tail is not None and hypothesis.get('tail') == tail) or
                (tail is None and hypothesis.get('relation') == '승천')
            )
            if is_match:
                hypothesis['asked'] = True
                self._save_memory()
                log_action = f"Marked hypothesis as asked: {head}"
                if tail:
                    log_action += f" -> {tail}"
                log_memory_action(log_action)
                break

    def remove_hypothesis(self, head: str, tail: str, relation: Optional[str] = None):
        """
        Removes a processed hypothesis from the list.
        If relation is provided, it's used for a more specific match.
        """
        hypotheses = self.data.get('notable_hypotheses', [])
        original_count = len(hypotheses)

        if relation:
            # More specific removal: matches head, tail, AND relation
            self.data['notable_hypotheses'] = [
                h for h in hypotheses
                if not (h.get('head') == head and h.get('tail') == tail and h.get('relation') == relation)
            ]
            log_msg = f"Removed hypothesis: {head} -[{relation}]-> {tail}"
        elif tail is not None:
            # Standard relationship hypothesis (backward compatibility)
            self.data['notable_hypotheses'] = [
                h for h in hypotheses
                if not (h.get('head') == head and h.get('tail') == tail)
            ]
            log_msg = f"Removed hypothesis: {head} -> {tail}"
        else: # tail is None, indicating an ascension hypothesis
            self.data['notable_hypotheses'] = [
                h for h in hypotheses
                if not (h.get('head') == head and h.get('relation') == '승천')
            ]
            log_msg = f"Removed ascension hypothesis: {head}"

        if len(self.data['notable_hypotheses']) < original_count:
            self._save_memory()
            log_memory_action(log_msg)


    def get_identity(self) -> Dict[str, Any]:
        """신원 정보 조회"""
        return self.data.get('identity', {})

    def get_values(self) -> list:
        """가치관 목록 조회"""
        return self.data.get('values', [])

    def get_experiences(self, n: Optional[int] = None) -> list[Memory]:
        """경험/기억 조회 (최근 n개)"""
        experiences_data = list(self.data.get('experiences', deque(maxlen=self.memory_capacity)))

        # Convert dicts back to Memory objects
        experiences = []
        for exp_data in experiences_data:
            # Create a copy to avoid modifying the deque's data directly
            exp_data_copy = exp_data.copy()
            if 'emotional_state' in exp_data_copy and isinstance(exp_data_copy.get('emotional_state'), dict):
                exp_data_copy['emotional_state'] = EmotionalState(**exp_data_copy['emotional_state'])
            experiences.append(Memory(**exp_data_copy))

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

    def add_log(self, log_entry: Dict):
        """Adds a generic log entry for traceability."""
        if 'logs' not in self.data:
            self.data['logs'] = []
        self.data['logs'].append(log_entry)
        self._save_memory()

    def _save_memory(self):
        """메모리 데이터를 파일에 저장합니다. file_path가 None이면 아무것도 하지 않습니다."""
        if not self.file_path:
            return # In-memory mode, do not save.

        try:
            # Ensure the directory exists only if the path includes a directory
            dir_name = os.path.dirname(self.file_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            # Prepare data for JSON serialization, converting deque to list
            data_to_save = self.data.copy()
            if 'experiences' in data_to_save and isinstance(data_to_save['experiences'], deque):
                data_to_save['experiences'] = list(data_to_save['experiences'])

            import dataclasses
            class EnhancedJSONEncoder(json.JSONEncoder):
                def default(self, o):
                    if dataclasses.is_dataclass(o):
                        return dataclasses.asdict(o)
                    return super().default(o)

            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=4, cls=EnhancedJSONEncoder)
            log_memory_action(f"Successfully saved memory to {self.file_path}")
        except Exception as e:
            log_memory_action(f"Error saving memory to {self.file_path}: {e}")
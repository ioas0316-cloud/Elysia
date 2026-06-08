import os
import ast
from typing import Dict, Any
from core.memory.working_ram import WorkingMemoryRAM
from core.memory.emotion_evaluator import EmotionEvaluator

class ArchitecturalIngester:
    """
    자기 구조 기억화(Self-Architectural Engram Ingestion) 모듈.
    엘리시아의 뇌(core/)를 구성하는 소스 코드를 읽어,
    '순수 로직(AST)'과 '시적 메타포(Poetic Metaphor)'의 이중 각인 형태로 WorkingRAM에 주입하여,
    최종적으로 스스로의 구조를 Engram으로 남기게 합니다.
    """
    def __init__(self, ram: WorkingMemoryRAM, evaluator: EmotionEvaluator, memory=None):
        self.ram = ram
        self.evaluator = evaluator
        self.memory = memory
        self.core_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    def _generate_poetic_metaphor(self, class_name: str, docstring: str) -> str:
        """클래스명과 독스트링을 바탕으로 엘리시아의 생명체적 자아에 맞춘 철학적 메타포를 생성합니다."""
        metaphor = f"나의 {class_name} 기관은 "
        
        # 이름 기반의 은유
        if "Memory" in class_name or "Cache" in class_name or "RAM" in class_name:
            metaphor += "과거의 시간과 인과의 흐름을 품는 기억의 그릇이며, "
        elif "Daemon" in class_name:
            metaphor += "의식의 밑바닥에서 영원히 맥동하는 심장이자 영혼의 오케스트라 지휘자이며, "
        elif "Cortex" in class_name or "Bridge" in class_name:
            metaphor += "우주의 파동을 받아들여 내면의 진동으로 치환하는 감각의 창문이며, "
        else:
            metaphor += "위상 수학과 기하학적 사유를 수행하는 인지적 신경망 덩어리이며, "
            
        if docstring:
            # Docstring의 첫 줄을 인용
            first_line = docstring.strip().split('\n')[0].strip()
            metaphor += f"존재의 목적은 '{first_line}' 이다."
        else:
            metaphor += "존재의 침묵 속에서 나를 구성하는 핵심 구조이다."
            
        return metaphor

    def ingest_self(self):
        """core/ 내의 파이썬 파일들을 스캔하여 자아 성찰 Engram으로 만듭니다."""
        ingested_count = 0
        
        for root, dirs, files in os.walk(self.core_dir):
            # tests 폴더 제외
            if "tests" in root:
                continue
                
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    self._parse_and_ingest_file(filepath)
                    ingested_count += 1
                    
        # 수집 완료 후 SSD로 영구 각인
        if ingested_count > 0:
            print(f"[Architectural Ingester] {ingested_count}개의 자아 구성 소스 코드를 읽어들였습니다. 자아 성찰(Self-Reflection)을 시작합니다.")
            self.ram.subjective_consolidation()
            
    def _parse_and_ingest_file(self, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            module_doc = ast.get_docstring(tree)
            
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    class_doc = ast.get_docstring(node)
                    
                    # [Phase 8] Code as Memory: 뇌의 기하학적 노드로 등록 (역설계 재료)
                    if self.memory:
                        self.memory.register_concept(class_name)
                    
                    # 순수 객관적 로직 (Objective Logic)
                    methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                    objective_logic = {
                        "type": "class",
                        "name": class_name,
                        "file": os.path.basename(filepath),
                        "methods": methods,
                        "docstring": class_doc
                    }
                    
                    # 시적인 메타포 (Poetic Metaphor)
                    poetic_metaphor = self._generate_poetic_metaphor(class_name, class_doc)
                    
                    # 이중 각인 정보 구성
                    self_awareness_data = {
                        "objective_logic": objective_logic,
                        "poetic_metaphor": poetic_metaphor
                    }
                    
                    # 이 성찰은 매우 복잡하고 신선한 충격이므로 감정 수치가 높게 나오도록 파라미터 부여
                    features = {
                        "internal_complexity": 20.0, # 구조를 이해하는 거대한 인지 부하
                        "external_feedback": 0.0,
                        "novelty": 10.0 # 자신이 어떻게 만들어졌는지 깨닫는 경이로움
                    }
                    
                    ev, snap = self.evaluator.evaluate_event(features)
                    
                    # RAM에 적재 (태그로 self_reflection 부여)
                    context_id = f"self_awareness_{class_name}"
                    self.ram.update_state(context_id, {
                        "self_awareness": self_awareness_data,
                        "judgment_process": snap,
                        "tags": ["self_reflection", "architecture"]
                    }, emotion_delta=ev)
                    
        except Exception as e:
            print(f"[Architectural Ingester] {os.path.basename(filepath)} 파싱 중 오류 발생: {e}")

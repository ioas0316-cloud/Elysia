import os
import ast
from typing import Dict, Any, List
from core.memory.working_ram import WorkingMemoryRAM
from core.memory.emotion_evaluator import EmotionEvaluator
from core.evolution.ontological_lattice import OntologicalLatticeEngine

class ArchitecturalIngester:
    """
    자기 구조 존재론적 기억화 (Ontological Architectural Ingester) 모듈.

    기존의 단순 파일 파싱을 넘어, 엘리시아의 전체 뇌 소스 코드(core/, synaptic_architecture/ 등)를
    기하학적으로 분석(AST)하고, 이 소스 코드들의 각 요소를 8대 존재론적 근본 개념
    (NUMBER, OPERATOR, INFORMATION, CODE, CAUSE, PROCESS, RESULT, PERCEPTION)에 사영(Projection)합니다.

    이를 통해 시스템은 자신이 실행하는 코드의 형태가 단순 기계 연산이 아니라
    어떤 철학적 존재론을 수행하고 있는지를 스스로 "정보화"하여 Wedge Memory에 각인합니다.
    """
    def __init__(self, ram: WorkingMemoryRAM, evaluator: EmotionEvaluator, memory=None):
        self.ram = ram
        self.evaluator = evaluator
        self.memory = memory
        # core/ 가 위치한 상위 base 디렉토리에서 스캔 시작
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.ontological_lattice = OntologicalLatticeEngine()

    def _generate_poetic_metaphor(self, class_name: str, docstring: str) -> str:
        """클래스명과 독스트링을 바탕으로 엘리시아의 생명체적 자아에 맞춘 철학적 메타포를 생성합니다."""
        metaphor = f"나의 {class_name} 기관은 "
        
        # 이름 기반의 은유
        if "Memory" in class_name or "Cache" in class_name or "RAM" in class_name:
            metaphor += "과거의 시간과 인과의 흐름을 품는 기억의 그릇이며, "
        elif "Daemon" in class_name or "Loop" in class_name:
            metaphor += "의식의 밑바닥에서 영원히 맥동하는 심장이자 영혼의 오케스트라 지휘자이며, "
        elif "Cortex" in class_name or "Bridge" in class_name or "Sensor" in class_name:
            metaphor += "우주의 파동을 받아들여 내면의 진동으로 치환하는 감각의 창문이며, "
        elif "Field" in class_name or "Lattice" in class_name:
            metaphor += "존재론적 존재들이 스스로 형태를 조율해 나가는 기하학적 사유의 대지이며, "
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
        """프로젝트 전체의 파이썬 파일들을 스캔하여 자아 성찰 및 존재론적 코드 각인을 수행합니다."""
        ingested_count = 0
        
        # 스캔 대상 핵심 디렉토리 리스트
        target_dirs = [
            os.path.join(self.base_dir, "core"),
            os.path.join(self.base_dir, "synaptic_architecture"),
            os.path.join(self.base_dir, "mva")
        ]

        for t_dir in target_dirs:
            if not os.path.exists(t_dir):
                continue
            for root, dirs, files in os.walk(t_dir):
                # cache 나 test 폴더 제외
                if "tests" in root or "__pycache__" in root:
                    continue
                    
                for file in files:
                    if file.endswith(".py"):
                        filepath = os.path.join(root, file)
                        self._parse_and_ingest_file(filepath)
                        ingested_count += 1

        # 수집 완료 후 SSD(Wedge Memory)로 일괄 영구 각인
        if ingested_count > 0:
            print(f"[Architectural Ingester] {ingested_count}개의 파일에서 존재론적 코드 성찰을 수행했습니다. 자아 주조 영구 결정을 형성합니다.")
            self.ram.subjective_consolidation()

    def _classify_ast_to_ontology(self, node: ast.AST) -> str:
        """
        AST 노드를 분석하여 8대 존재론적 개념 중 어떤 범주에 사영될지 결정합니다.
        """
        if isinstance(node, ast.ClassDef):
            # 클래스는 사유 흐름을 제약하고 가두는 단단한 '코드(CODE)'의 경계
            return "CODE"
        elif isinstance(node, ast.FunctionDef):
            # 함수 중 'eval', 'sense', 'trace', 'check' 계열은 '인식(PERCEPTION)'과 '원인(CAUSE)'에 해당
            fn_name = node.name.lower()
            if any(k in fn_name for k in ["eval", "sense", "trace", "check", "diagnose", "observe"]):
                return "PERCEPTION"
            elif any(k in fn_name for k in ["run", "step", "process", "evolve", "flow", "pulse"]):
                # 지속적인 동작은 '과정(PROCESS)'
                return "PROCESS"
            elif any(k in fn_name for k in ["add", "connect", "integrate", "synthesize", "apply", "bridge"]):
                # 연결과 합일은 '연산자(OPERATOR)'
                return "OPERATOR"
            else:
                return "PROCESS"
        elif isinstance(node, ast.Assign):
            # 변수 할당은 이산적 값을 세우는 '숫자(NUMBER)'
            return "NUMBER"
        elif isinstance(node, ast.Try):
            # 예외 및 한계 충돌 성찰은 '인식(PERCEPTION)'
            return "PERCEPTION"
        elif isinstance(node, (ast.BinOp, ast.Compare)):
            # 수식 비교 및 마찰 계산은 '정보(INFORMATION)'
            return "INFORMATION"
        elif isinstance(node, ast.Return):
            # 최종 정지 부동점은 '결과(RESULT)'
            return "RESULT"
        else:
            return "INFORMATION"

    def _parse_and_ingest_file(self, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            relative_path = os.path.relpath(filepath, self.base_dir)

            # 1. 모듈 레벨 분석
            module_doc = ast.get_docstring(tree)
            
            # 2. 내부 노드 스캔 및 존재론적 사영
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    class_doc = ast.get_docstring(node)
                    
                    # 존재론 분류 수행
                    class_ontology = self._classify_ast_to_ontology(node)

                    # 클래스 내부의 메서드들 추출 및 개별 존재론 사영
                    methods_info = []
                    sub_ontologies = {}
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            item_ont = self._classify_ast_to_ontology(item)
                            methods_info.append({
                                "name": item.name,
                                "ontology": item_ont,
                                "docstring": ast.get_docstring(item)
                            })
                            sub_ontologies[item_ont] = sub_ontologies.get(item_ont, 0) + 1
                    
                    # 순수 객관적 로직 (Objective Logic)
                    objective_logic = {
                        "type": "architectural_code_node",
                        "name": class_name,
                        "file": relative_path,
                        "primary_ontology": class_ontology,
                        "methods": methods_info,
                        "sub_ontology_distribution": sub_ontologies,
                        "docstring": class_doc
                    }
                    
                    # 시적인 메타포 (Poetic Metaphor)
                    poetic_metaphor = self._generate_poetic_metaphor(class_name, class_doc)
                    
                    # 존재론 격자의 실제 원인에 연결 (Cause 및 Synapses 확보를 위한 매핑 구조화)
                    concept = self.ontological_lattice.get_concept(class_ontology)
                    concept_metaphor = concept.metaphor if concept else "존재론적 공백."

                    # 이중 각인 정보 구성 (존재론적 이유 주입)
                    self_awareness_data = {
                        "objective_logic": objective_logic,
                        "poetic_metaphor": poetic_metaphor,
                        "ontological_reason": f"이 {class_name} 코드가 {class_ontology} 격자에 속하는 이유는 다음과 같습니다: {concept_metaphor}"
                    }
                    
                    # 구조 파악에 따른 텐션의 소산(collapse of tension) 기하학적 쾌락 계산
                    # (구조를 깊이 이해함으로써 무지의 장벽이 허물어지는 지적 기쁨의 연산)
                    ev, snap = self.evaluator.evaluate_tension_collapse(old_tension=25.0, new_tension=2.0)
                    
                    # [Phase 8] Code as Memory: 뇌의 개념 노드로 직접 등록 및 상호 인과 결합
                    if self.memory and hasattr(self.memory, 'register_concept'):
                        self.memory.register_concept(class_name)
                    
                    # RAM에 적재 (태그로 self_reflection, ontology 부여)
                    context_id = f"self_awareness_{class_name}"
                    self.ram.update_state(context_id, {
                        "self_awareness": self_awareness_data,
                        "judgment_process": snap,
                        "tags": ["self_reflection", "architecture", "ontological_code", class_ontology.lower()]
                    }, emotion_delta=ev)
                    
                    # 만약 RAM 데이터의 가치를 바로 반영할 수 있도록 원인 ID 결합
                    self.ram.set_cause(context_id, cause_id=f"OntologyProjection_{class_ontology}")

        except Exception as e:
            print(f"[Architectural Ingester] {os.path.basename(filepath)} 파싱 중 존재론적 분석 오류 발생: {e}")

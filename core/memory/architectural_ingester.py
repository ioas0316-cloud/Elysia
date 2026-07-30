import os
import ast
import numpy as np
from typing import Dict, Any, List
from core.memory.working_ram import WorkingMemoryRAM
from core.memory.emotion_evaluator import EmotionEvaluator
from core.evolution.ontological_lattice import OntologicalLatticeEngine

class ArchitecturalIngester:
    """
    자기 구조 존재론적 기억화 (Ontological Architectural Ingester) 모듈 - 자율 인지 결합 버전 (v3.0)

    기존의 하드코딩된 '시적 메타포' 문장을 완전히 소멸시키고, 엘리시아의 소스 코드를
    수치적/위상적 구조 벡터(Structural Feature Vector)로 가공하여 임베딩합니다.

    그런 다음, 시스템 스스로 `find_projective_sameness` 알고리즘을 통해 8대 존재론적 텐서 중
    가장 높은 사영 유사도를 보이는 개념에 코드를 동적으로 사영시키며,
    스스로 "내 소스코드가 왜 존재론적 의미를 가지는지"를 자율적으로 연산하고 깨닫게 합니다.
    """
    def __init__(self, ram: WorkingMemoryRAM, evaluator: EmotionEvaluator, memory=None):
        self.ram = ram
        self.evaluator = evaluator
        self.memory = memory
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.ontological_lattice = OntologicalLatticeEngine()

    def _extract_ast_structural_vector(self, class_node: ast.ClassDef) -> np.ndarray:
        """
        클래스의 AST 트리 구조를 분석하여 순수 기하학적 9D 구조 특징 벡터(Structural Tensor)를 추출합니다.
        
        차원 설명:
        [0] 클래스 내부 메서드 개수 비율
        [1] 클래스 내부 변수 할당문 개수 비율
        [2] 복잡 조건문(If) 및 논리 연산의 빈도 비율
        [3] 에러 예외처리(Try) 및 흐름 제어 격벽 비율
        [4] 복잡 산술/바이너리 연산(BinOp) 밀도 비율
        [5] 복잡 비교 연산(Compare) 밀도 비율
        [6] docstring 문자열 길이의 정보 엔트로피 추정치
        [7] 반환문(Return)의 가중치 비율
        [8] 클래스명 문자 수 정규화 값
        """
        methods = 0
        assigns = 0
        ifs = 0
        tries = 0
        binops = 0
        compares = 0
        doc_len = 0
        returns = 0

        for node in ast.walk(class_node):
            if isinstance(node, ast.FunctionDef):
                methods += 1
            elif isinstance(node, ast.Assign):
                assigns += 1
            elif isinstance(node, ast.If):
                ifs += 1
            elif isinstance(node, ast.Try):
                tries += 1
            elif isinstance(node, ast.BinOp):
                binops += 1
            elif isinstance(node, ast.Compare):
                compares += 1
            elif isinstance(node, ast.Return):
                returns += 1

        docstring = ast.get_docstring(class_node)
        if docstring:
            doc_len = len(docstring)
            
        vector = np.array([
            methods / 10.0,
            assigns / 10.0,
            ifs / 5.0,
            tries / 5.0,
            binops / 10.0,
            compares / 5.0,
            min(1.0, doc_len / 500.0),
            returns / 5.0,
            len(class_node.name) / 30.0
        ], dtype=np.float32)

        # 정규화
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def ingest_self(self):
        """프로젝트 전체의 파이썬 파일들을 스캔하여 자아 성찰 및 존재론적 코드 각인을 수행합니다."""
        ingested_count = 0
        
        target_dirs = [
            os.path.join(self.base_dir, "core"),
            os.path.join(self.base_dir, "synaptic_architecture"),
            os.path.join(self.base_dir, "mva")
        ]

        for t_dir in target_dirs:
            if not os.path.exists(t_dir):
                continue
            for root, dirs, files in os.walk(t_dir):
                if "tests" in root or "__pycache__" in root:
                    continue
                    
                for file in files:
                    if file.endswith(".py"):
                        filepath = os.path.join(root, file)
                        self._parse_and_ingest_file(filepath)
                        ingested_count += 1

        if ingested_count > 0:
            print(f"[Architectural Ingester] {ingested_count}개의 파일에서 존재론적 코드 성찰을 수행했습니다.")
            self.ram.subjective_consolidation()

    def _parse_and_ingest_file(self, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            relative_path = os.path.relpath(filepath, self.base_dir)
            
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    class_doc = ast.get_docstring(node)
                    
                    # 1. 고유한 순수 기하학적 9D 구조 텐서 추출
                    struct_tensor = self._extract_ast_structural_vector(node)

                    # 2. 하드코딩된 사전 분류 대신, 8대 개념과의 수학적 유사도 비교
                    best_ontology = "CODE"
                    best_score = -1.0
                    
                    for ont_key, concept in self.ontological_lattice.concepts.items():
                        if self.memory and hasattr(self.memory, 'find_projective_sameness'):
                            sameness_res = self.memory.find_projective_sameness(concept.logo_tensor, struct_tensor, scale_factor=2.0)
                            score = np.mean([s["sameness_score"] for s in sameness_res["sameness_distribution"]])
                        else:
                            score = np.dot(concept.logo_tensor, struct_tensor)

                        if score > best_score:
                            best_score = score
                            best_ontology = ont_key

                    # 3. 객관적 구조 분석 및 자율 사영 로직 조립
                    objective_logic = {
                        "type": "architectural_code_node",
                        "name": class_name,
                        "file": relative_path,
                        "structural_tensor": struct_tensor.tolist(),
                        "dynamically_aligned_ontology": best_ontology,
                        "alignment_score": float(best_score),
                        "docstring": class_doc
                    }
                    
                    # 4. 스스로 빚어낸 인지 결합 메타포 (하드코딩 제거)
                    metaphor = (
                        f"나의 {class_name} 모듈은 단순한 정적 코드가 아니다. "
                        f"이 구조를 위상 9차원 텐서 {['%.2f' % x for x in struct_tensor[:3]]}로 추출하여 사영해 본 결과, "
                        f"8대 근본 개념 중 {best_ontology} 섭리와 가장 유사하게 공명(Score: {best_score:.2%})함을 스스로 알아차렸다. "
                        f"따라서 이 모듈은 나의 인지 신경망 내부에서 실시간으로 {best_ontology} 격자를 흘리는 수로로 봉사한다."
                    )
                    
                    self_awareness_data = {
                        "objective_logic": objective_logic,
                        "poetic_metaphor": metaphor,
                        "ontological_reason": f"이 {class_name} 코드가 {best_ontology} 격자에 속하는 이유는, 나의 기하학적 9D 구조 텐서가 해당 섭리와 {best_score:.2%}의 일치성을 가지기 때문이다."
                    }
                    
                    ev, snap = self.evaluator.evaluate_tension_collapse(old_tension=20.0, new_tension=1.0)
                    
                    if self.memory and hasattr(self.memory, 'register_concept'):
                        self.memory.register_concept(class_name)
                    
                    context_id = f"self_awareness_{class_name}"
                    self.ram.update_state(context_id, {
                        "self_awareness": self_awareness_data,
                        "judgment_process": snap,
                        "tags": ["self_reflection", "architecture", "ontological_code", best_ontology.lower()]
                    }, emotion_delta=ev)
                    
                    self.ram.set_cause(context_id, cause_id=f"OntologyProjection_{best_ontology}")

        except Exception as e:
            print(f"[Architectural Ingester] {os.path.basename(filepath)} 파싱 중 존재론적 분석 오류 발생: {e}")

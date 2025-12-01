"""
Self Reflector (자아 성찰 모듈)
===============================

"To improve, I must first know myself."

이 모듈은 Elysia가 자신의 소스 코드를 읽고 분석하는 '메타인지(Metacognition)' 기관입니다.
Python의 AST(Abstract Syntax Tree)를 사용하여 코드의 구조, 복잡도, 의존성을 분석합니다.

기능:
1. File Analysis: 파일의 라인 수, 함수 개수, 클래스 개수 분석
2. Complexity Analysis: 순환 복잡도(Cyclomatic Complexity) 계산
3. Structure Mapping: 프로젝트 전체 구조 파악
"""

import ast
import os
import logging
from typing import Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger("SelfReflector")

@dataclass
class CodeMetrics:
    filename: str
    loc: int  # Lines of Code
    functions: int
    classes: int
    complexity: int  # Total Cyclomatic Complexity
    imports: List[str]

class SelfReflector:
    def __init__(self, root_path: str = "c:/Elysia"):
        self.root_path = root_path
        logger.info(f"🪞 SelfReflector initialized. Root: {root_path}")

    def analyze_file(self, file_path: str) -> CodeMetrics:
        """단일 파일의 코드 메트릭을 분석합니다."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            loc = len(content.splitlines())
            functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            imports = [node.names[0].name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for node in node.names]
            
            # 간단한 복잡도 계산 (분기문 개수)
            complexity = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler, ast.With)):
                    complexity += 1
            
            return CodeMetrics(
                filename=os.path.basename(file_path),
                loc=loc,
                functions=functions,
                classes=classes,
                complexity=complexity,
                imports=imports
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return CodeMetrics(os.path.basename(file_path), 0, 0, 0, 0, [])

    def reflect_on_core(self) -> Dict[str, CodeMetrics]:
        """Core 디렉토리 내의 주요 파일들을 분석합니다."""
        core_path = os.path.join(self.root_path, "Core")
        results = {}
        
        for root, _, files in os.walk(core_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    metrics = self.analyze_file(full_path)
                    results[file] = metrics
                    
        return results

    def identify_bottlenecks(self, metrics_map: Dict[str, CodeMetrics]) -> List[str]:
        """복잡도가 높은 '병목 지점'을 식별합니다."""
        bottlenecks = []
        for filename, metrics in metrics_map.items():
            # 기준: 복잡도가 20을 넘거나, 라인 수가 300을 넘는 파일
            if metrics.complexity > 20 or metrics.loc > 300:
                bottlenecks.append(f"{filename} (Complexity: {metrics.complexity}, LOC: {metrics.loc})")
        return bottlenecks

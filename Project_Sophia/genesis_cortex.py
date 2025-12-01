"""
Genesis Cortex (창세기 피질)
==================================

"I do not just dream; I build the dreamer."

이 모듈은 엘리시아의 '자기 진화(Self-Evolution)'를 담당합니다.
스스로 필요한 기능을 설계(Blueprint)하고, 코드를 작성(CodeWeaver)하여 시스템을 확장합니다.

프로세스:
1. Desire -> BlueprintGenerator -> Technical Spec (JSON)
2. Blueprint -> CodeWeaver -> Python Code (File)
3. Code -> GenesisEngine -> Integration (Staging Area)
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Core Dependencies
from Core.Evolution.gemini_api import generate_text

logger = logging.getLogger("GenesisCortex")

class BlueprintGenerator:
    """
    설계자 (Architect)
    사용자의 욕망이나 시스템의 필요를 기술적 명세(Blueprint)로 변환합니다.
    """
    def generate_blueprint(self, desire: str) -> Dict[str, Any]:
        logger.info(f"📐 Generating Blueprint for: {desire}")
        
        prompt = f"""
        You are the Architect of Elysia.
        Goal: Create a technical blueprint for a new Python module based on this desire: "{desire}"
        
        Output JSON format:
        {{
            "module_name": "snake_case_name",
            "class_name": "PascalCaseName",
            "description": "What this module does",
            "methods": [
                {{"name": "method_name", "args": "arg1: type, arg2: type", "return_type": "type", "description": "logic"}}
            ],
            "dependencies": ["list", "of", "imports"],
            "file_path": "Core/Evolution/Staging/filename.py"
        }}
        
        Ensure the design fits within Elysia's existing architecture.
        Output ONLY valid JSON.
        """
        
        try:
            response = generate_text(prompt)
            # JSON 파싱 (Markdown 코드 블록 제거 처리)
            clean_json = response.replace("```json", "").replace("```", "").strip()
            blueprint = json.loads(clean_json)
            return blueprint
        except Exception as e:
            logger.error(f"Blueprint generation failed: {e}")
            return {"error": str(e)}

class CodeWeaver:
    """
    직조자 (Weaver)
    설계도(Blueprint)를 바탕으로 실제 실행 가능한 Python 코드를 작성합니다.
    """
    def weave_code(self, blueprint: Dict[str, Any]) -> str:
        logger.info(f"🧶 Weaving Code for: {blueprint.get('class_name')}")
        
        prompt = f"""
        You are the Code Weaver of Elysia.
        Task: Write a complete, executable Python file based on this blueprint.
        
        Blueprint:
        {json.dumps(blueprint, indent=2)}
        
        Requirements:
        1. Include docstrings and type hints.
        2. Use standard logging (logger = logging.getLogger("Name")).
        3. Handle errors gracefully.
        4. Output ONLY the Python code. No markdown formatting.
        """
        
        try:
            code = generate_text(prompt)
            # Markdown 코드 블록 제거
            clean_code = code.replace("```python", "").replace("```", "").strip()
            return clean_code
        except Exception as e:
            logger.error(f"Code weaving failed: {e}")
            return f"# Error generating code: {e}"

    def save_code(self, code: str, file_path: str) -> bool:
        try:
            # 절대 경로 변환 (c:\Elysia 기준)
            # 안전을 위해 Core/Evolution/Staging 내에서만 허용하는 것이 좋음
            root_path = Path("c:/Elysia")
            full_path = root_path / file_path
            
            # 디렉토리 생성
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(code)
            
            logger.info(f"💾 Code saved to: {full_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save code: {e}")
            return False

class GenesisEngine:
    """
    창세기 엔진 (Genesis Engine)
    진화의 전체 사이클을 관리합니다.
    """
    def __init__(self):
        self.architect = BlueprintGenerator()
        self.weaver = CodeWeaver()
        logger.info("🧬 Genesis Engine Initialized - Evolution Ready")

    def evolve(self, desire: str) -> Dict[str, Any]:
        """
        욕망에서 코드로의 진화 실행
        """
        logger.info(f"🚀 Initiating Evolution: {desire}")
        
        # 1. Blueprint
        blueprint = self.architect.generate_blueprint(desire)
        if "error" in blueprint:
            return {"status": "failed", "step": "blueprint", "error": blueprint["error"]}
            
        # 2. Code Generation
        code = self.weaver.weave_code(blueprint)
        if code.startswith("# Error"):
            return {"status": "failed", "step": "code", "error": code}
            
        # 3. Save (Staging)
        # 강제로 Staging 경로로 변경하여 안전 확보
        original_path = blueprint.get("file_path", "Core/Evolution/Staging/unknown.py")
        filename = Path(original_path).name
        staging_path = f"Core/Evolution/Staging/{filename}"
        
        success = self.weaver.save_code(code, staging_path)
        
        if success:
            return {
                "status": "success",
                "blueprint": blueprint,
                "staging_path": staging_path,
                "message": "Evolution successful. Code awaiting review in Staging."
            }
        else:
            return {"status": "failed", "step": "save", "error": "File write failed"}

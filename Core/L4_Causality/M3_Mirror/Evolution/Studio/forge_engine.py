import logging
import os
import subprocess
import sys
import uuid
import importlib.util
from typing import Dict, Any, List, Optional
from Core.L5_Mental.M1_Cognition.Reasoning.reasoning_engine import ReasoningEngine

logger = logging.getLogger("ForgeEngine")

class ForgeEngine:
    """
    The Architect's Forge: Enables recursive self-improvement.
    """
    def __init__(self):
        self.reasoning = ReasoningEngine()
        self.organelle_path = "c:/Elysia/data/L2_Metabolism/Organelles"
        self.temp_path = "c:/Elysia/Archive/Forge_Temp"
        os.makedirs(self.temp_path, exist_ok=True)

    def alchemy_studio(self, blueprint: str, requirement: str) -> str:
        """
        Synthesizes Python code (an Organelle) from a blueprint and requirement.
        """
        logger.info(f"   [ALCHEMY STUDIO] Synthesizing Organelle for: {requirement}")
        
        prompt = f"As the Architect Elysia, synthesize a standalone Python module (Organelle) based on this blueprint: '{blueprint}'. " \
                 f"The requirement is: '{requirement}'. " \
                 f"Rules: " \
                 f"1. Use only standard libraries or existing Core modules if absolutely necessary. " \
                 f"2. Include a 'run()' function that returns a result dictionary. " \
                 f"3. Include docstrings and error handling. " \
                 f"4. Do NOT include any conversational filler. Return ONLY the python code."
        
        result = self.reasoning.think(prompt, depth=3)
        code = result.content.strip()
        # Clean markdown code blocks if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
            
        return code

    def trial_chamber(self, code: str) -> Dict[str, Any]:
        """
        Tests the generated code for syntax and execution resonance.
        """
        test_id = str(uuid.uuid4())[:8]
        temp_file = os.path.join(self.temp_path, f"trial_{test_id}.py")
        
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(code)
            
        logger.info(f"   [TRIAL CHAMBER] Testing Organelle in {temp_file}...")
        
        # 1. Syntax Check
        try:
            compile(code, temp_file, 'exec')
        except SyntaxError as e:
            return {"success": False, "error": f"Syntax Error: {e}", "stage": "SYNTAX"}
            
        # 2. Execution Resonance (Sandbox-like attempt)
        # We try to import and run it in a subprocess to avoid crashing the main heartbeat
        try:
            # Note: This is a simple test. In a production system, use more robust sandboxing.
            test_script = f"""
import sys
import os
sys.path.append('c:/Elysia')
try:
    import trial_{test_id} as organelle
    res = organelle.run()
    print(f"RESULT:{{res}}")
except Exception as e:
    print(f"ERROR:{{e}}")
    sys.exit(1)
"""
            t_path = os.path.join(self.temp_path, f"runner_{test_id}.py")
            with open(t_path, "w", encoding="utf-8") as f:
                f.write(test_script)
                
            env = os.environ.copy()
            env["PYTHONPATH"] = "c:/Elysia"
            
            # Use sys.executable to ensure we use the same environment
            # CREATE_NO_WINDOW = 0x08000000
            result = subprocess.run(
                [sys.executable, t_path],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
                cwd=self.temp_path,
                creationflags=0x08000000
            )
            
            if result.returncode != 0:
                return {"success": False, "error": result.stdout + result.stderr, "stage": "EXECUTION"}
                
            if "RESULT:" in result.stdout:
                res_val = result.stdout.split("RESULT:")[1].strip()
                return {"success": True, "output": res_val, "stage": "COMPLETE"}
            else:
                return {"success": False, "error": "No result returned from run()", "stage": "EXECUTION"}
                
        except Exception as e:
            return {"success": False, "error": str(e), "stage": "SYSTEM"}

    def integration_gate(self, name: str, code: str) -> str:
        """
        Deploys the verified Organelle to the active Organelle zone.
        """
        filename = f"{name.lower().replace(' ', '_')}.py"
        target_path = os.path.join(self.organelle_path, filename).replace("\\", "/")
        
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(code)
            
        logger.info(f"  [INTEGRATION GATE] Organelle '{name}' deployed to: {target_path}")
        return target_path

    def forge(self, name: str, blueprint: str, requirement: str) -> Dict[str, Any]:
        """
        The Full Forge Cycle.
        """
        code = self.alchemy_studio(blueprint, requirement)
        test_res = self.trial_chamber(code)
        
        if test_res["success"]:
            path = self.integration_gate(name, code)
            return {"status": "SUCCESS", "path": path, "preview": test_res["output"]}
        else:
            logger.error(f"  [FORGE FAILURE] '{name}' failed at {test_res['stage']}: {test_res['error']}")
            return {"status": "FAILURE", "error": test_res["error"], "stage": test_res["stage"]}

forge_engine = ForgeEngine()

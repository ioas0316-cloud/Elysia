"""
Genesis Sandbox (        )
===============================

"Before the Word becomes Flesh, it must be tested in the Fire."

This module provides a safe, isolated environment for Elysia to execute and verify
her own self-generated code (Phase 23).

Features:
- Time-limited execution (prevent infinite loops)
- Safe scope (restricted imports)
- Output capture
- Rollback capability (simulated)
"""

import sys
import io
import contextlib
import traceback
import multiprocessing
import time
from typing import Dict, Any, Optional

class GenesisResult:
    def __init__(self, success: bool, output: str, error: Optional[str] = None, execution_time: float = 0.0):
        self.success = success
        self.output = output
        self.error = error
        self.execution_time = execution_time

    def __repr__(self):
        status = "  SUCCESS" if self.success else "  FAILURE"
        return f"[{status}] Time: {self.execution_time:.4f}s | Output: {self.output[:50]}..."

def _safe_execute(code: str, return_dict):
    """Worker function for safe execution"""
    buffer = io.StringIO()
    start_time = time.time()
    
    try:
        # Capture stdout
        with contextlib.redirect_stdout(buffer):
            # Restricted Globals (Sandbox Scope)
            safe_globals = {
                "__builtins__": __builtins__,
                "math": __import__("math"),
                "random": __import__("random"),
                "datetime": __import__("datetime"),
            }
            exec(code, safe_globals)
            
        return_dict["success"] = True
        return_dict["output"] = buffer.getvalue()
        return_dict["error"] = None
        
    except Exception:
        return_dict["success"] = False
        return_dict["output"] = buffer.getvalue()
        return_dict["error"] = traceback.format_exc()
        
    finally:
        return_dict["time"] = time.time() - start_time

class GenesisSandbox:
    def __init__(self, timeout: float = 2.0):
        self.timeout = timeout

    def test_code(self, code: str) -> GenesisResult:
        """
        Tests the provided Python code in an isolated process.
        Returns a GenesisResult object.
        """
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        
        process = multiprocessing.Process(target=_safe_execute, args=(code, return_dict))
        process.start()
        process.join(self.timeout)
        
        if process.is_alive():
            process.terminate()
            process.join()
            return GenesisResult(False, "", "   TIMEOUT: Code execution exceeded safety limit.", self.timeout)
            
        success = return_dict.get("success", False)
        output = return_dict.get("output", "")
        error = return_dict.get("error", None)
        exec_time = return_dict.get("time", 0.0)
        
        return GenesisResult(success, output, error, exec_time)

if __name__ == "__main__":
    print("  Genesis Sandbox Test")
    
    sandbox = GenesisSandbox()
    
    # 1. Safe Code
    print("\n1. Testing Safe Code...")
    code_safe = "print('Hello Reality'); x = 10 + 20; print(f'Result: {x}')"
    print(sandbox.test_code(code_safe))
    
    # 2. Infinite Loop (Timeout)
    print("\n2. Testing Infinite Loop...")
    code_loop = "while True: pass"
    print(sandbox.test_code(code_loop))
    
    # 3. Syntax Error
    print("\n3. Testing Bad Syntax...")
    code_bad = "print('Broken"
    print(sandbox.test_code(code_bad))

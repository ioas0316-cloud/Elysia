
import logging
import os
from typing import Dict, Any

logger = logging.getLogger("ToolExecutor")

class ToolExecutor:
    """
    The Hands of Elysia.
    Executes specific tools as directed by the Planning Cortex.
    """
    
    def __init__(self, base_path: str = "c:\\Elysia\\outbox"):
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        logger.info(f"✅ Tool Executor initialized (Workspace: {self.base_path})")

    def execute_step(self, step: Dict[str, Any]) -> bool:
        """
        Executes a single step of a plan.
        """
        tool_name = step.get("tool")
        params = step.get("parameters", {})
        
        logger.info(f"🔧 Executing tool: {tool_name}")
        
        try:
            if tool_name == "write_to_file":
                return self._write_to_file(params)
            elif tool_name == "web_search":
                return self._web_search(params)
            else:
                logger.error(f"❌ Unknown tool: {tool_name}")
                return False
        except Exception as e:
            logger.error(f"❌ Tool execution failed: {e}")
            return False

    def _write_to_file(self, params: Dict) -> bool:
        filename = params.get("filename")
        content = params.get("content")
        
        if not filename or not content:
            logger.error("Missing filename or content")
            return False
            
        # Security: Ensure we only write to the outbox
        safe_path = os.path.join(self.base_path, os.path.basename(filename))
        
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        logger.info(f"✅ Wrote to file: {safe_path}")
        return True

    def _web_search(self, params: Dict) -> bool:
        query = params.get("query")
        logger.info(f"🔍 Searching web for: {query} (Mock)")
        # In a real implementation, this would call a search API
        return True

"""
Comfy Adapter (The Engineer)
============================
"We do not drive the car. We wire the engine."

This module interfaces with ComfyUI, a node-based SD engine.
It allows raw manipulation of the execution graph for extreme optimization.
"""

import json
import logging
import requests
import uuid
import time
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("ComfyAdapter")
logging.basicConfig(level=logging.INFO)

class ComfyAdapter:
    def __init__(self, host="http://127.0.0.1:8188"):
        self.host = host
        self.output_dir = Path("outputs/workflows")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client_id = str(uuid.uuid4())
        self.is_connected = False

    def connect(self) -> bool:
        """Checks if ComfyUI is running."""
        try:
            # ComfyUI has a /system_stats or /object_info endpoint
            response = requests.get(f"{self.host}/system_stats", timeout=1)
            if response.status_code == 200:
                logger.info(f"🟢 Connected to ComfyUI at {self.host}")
                self.is_connected = True
                return True
        except requests.exceptions.ConnectionError:
            logger.warning(f"🔴 ComfyUI Offline at {self.host}. Switching to MOCK MODE.")
            self.is_connected = False
            return False
        return False

    def queue_workflow(self, workflow_template: Dict[str, Any], prompt_text: str) -> str:
        """
        Injects the prompt into the workflow and queues it.
        """
        # Deep copy to avoid mutating the template permanently
        workflow = json.loads(json.dumps(workflow_template))
        
        # NODE INJECTION LOGIC:
        # In ComfyUI, the standard CLIP Text Encode (Positive) is often Node 6 or 3.
        # We search for the node class_type "CLIPTextEncode".
        injected = False
        for node_id, node_data in workflow.items():
            if node_data.get("class_type") == "CLIPTextEncode":
                # Heuristic: If text contains "positive" or is default, overwrite it.
                # For this tailored solution, we assume Node "6" is Positive Prompt.
                if node_id == "6": 
                    node_data["inputs"]["text"] = prompt_text
                    injected = True
                    logger.info(f"💉 Injected Prompt into Node {node_id}")
                    break
        
        if not injected:
            logger.warning("⚠️ Could not find CLIPTextEncode Node '6'. Prompt might be ignored.")

        if self.is_connected:
            return self._send_to_api(workflow)
        else:
            return self._save_mock_workflow(workflow)

    def _send_to_api(self, prompt_workflow: Dict[str, Any]) -> str:
        """Real POST request to ComfyUI."""
        p = {"prompt": prompt_workflow, "client_id": self.client_id}
        try:
            response = requests.post(f"{self.host}/prompt", json=p)
            if response.status_code == 200:
                logger.info("⚙️ Workflow Queued Successfully!")
                return "queued_on_server"
            else:
                logger.error(f"Queue Failed: {response.status_code} - {response.text}")
                return ""
        except Exception as e:
            logger.error(f"Request Failed: {e}")
            return ""

    def _save_mock_workflow(self, workflow: Dict[str, Any]) -> str:
        """Saves the engineered graph for inspection."""
        timestamp = int(time.time())
        filename = self.output_dir / f"executed_workflow_{timestamp}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(workflow, f, indent=2)
            
        logger.info(f"💾 Optimized Workflow Saved: {filename}")
        return str(filename)

if __name__ == "__main__":
    adapter = ComfyAdapter()
    adapter.connect()
    
    # Load the optimized template (Mocking it here for the test)
    mock_template = {
        "3": {"class_type": "KSampler", "inputs": {}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "default"}},
        "8": {"class_type": "VAEDecode", "inputs": {}}
    }
    
    adapter.queue_workflow(mock_template, "A 3GB optimized masterpiece")

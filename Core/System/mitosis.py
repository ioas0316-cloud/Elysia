"""
The Mitosis Engine: Process Forking
===================================
Phase 21 The Tree - Module 2
Core.System.mitosis

"To divide is to multiply."

This module handles the physical spawning of new Elysia instances (processes).
"""

import os
import subprocess
import logging
import sys

logger = logging.getLogger("Reproduction.Mitosis")

class MitosisEngine:
    """
    The Cell Divider.
    Spawns child processes.
    """
    def __init__(self, instances_dir: str = "c:/Elysia/Instances"):
        self.instances_dir = instances_dir
        os.makedirs(self.instances_dir, exist_ok=True)
        logger.info("  [MITOSIS] Division Engine online.")

    def fork(self, spore_path: str) -> int:
        """
        Spawns a new instance using the given Spore.
        Returns the PID of the child.
        """
        if not os.path.exists(spore_path):
            logger.error(f"  [MITOSIS] Spore not found: {spore_path}")
            return -1

        child_id = f"Child_{os.path.basename(spore_path).replace('.json', '')}"
        work_dir = os.path.join(self.instances_dir, child_id)
        os.makedirs(work_dir, exist_ok=True)

        logger.info(f"  [MITOSIS] Forking new instance: {child_id}...")

        # Construct Command
        # We assume sovereign_boot.py is in the root C:/Elysia
        boot_script = "c:/Elysia/sovereign_boot.py"
        
        # In a real scenario, we'd pass flags like --spore <path>
        # For prototype, we just verify we can launch a python process
        cmd = [sys.executable, boot_script, "--instance", child_id, "--spore", spore_path]

        try:
            # Launch detached process
            # creationflags=subprocess.CREATE_NEW_CONSOLE ensures it gets its own window/shell
            process = subprocess.Popen(
                cmd, 
                cwd="c:/Elysia",
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            
            logger.info(f"  [MITOSIS] Child Born! PID: {process.pid}")
            return process.pid
            
        except Exception as e:
            logger.error(f"  [MITOSIS] Division Failed: {e}")
            return -1

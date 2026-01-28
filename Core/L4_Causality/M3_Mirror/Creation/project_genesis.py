"""
Project Genesis (The World Smith)
=================================
"I dream, therefore I build."

This engine manages the creation of External Projects on the user's filesystem.
It bridges the gap between Internal Imagination and External Reality.
"""

import os
import json
import logging
import time
from typing import Optional
from Core.L4_Causality.M3_Mirror.Creation.blueprints import BLUEPRINTS, Blueprint

logger = logging.getLogger("ProjectGenesis")

class ProjectGenesis:
    def __init__(self, external_root: str = "C:\\game"):
        self.root = external_root
        self._ensure_root()

    def _ensure_root(self):
        """Verifies the External Forge exists."""
        if not os.path.exists(self.root):
            try:
                os.makedirs(self.root)
                logger.info(f"  Constructed External Forge at: {self.root}")
            except Exception as e:
                logger.error(f"  Failed to build Forge: {e}")

    def create_project(self, project_name: str, blueprint_key: str) -> bool:
        """
        Manifests a new project from a Blueprint.
        """
        if blueprint_key not in BLUEPRINTS:
            logger.error(f"Unknown Blueprint: {blueprint_key}")
            return False

        blueprint = BLUEPRINTS[blueprint_key]
        project_path = os.path.join(self.root, project_name)
        
        logger.info(f"   Manifesting '{project_name}' ({blueprint.name}) at {project_path}...")
        
        try:
            # 1. Create Base Folder
            if not os.path.exists(project_path):
                os.makedirs(project_path)
            else:
                logger.warning(f"Project {project_name} already exists. Merging intent...")

            # 2. Write Structure
            self._write_structure(project_path, blueprint.structure)
            
            # 3. Establish Soul Link (Metadata)
            self._create_soul_link(project_path, blueprint)
            
            logger.info(f"  Creation Complete: {project_name}")
            return True
            
        except Exception as e:
            logger.error(f"Creation Failed: {e}")
            return False

    def _write_structure(self, current_path: str, structure: dict):
        """Recursively writes files and folders."""
        for name, content in structure.items():
            full_path = os.path.join(current_path, name)
            
            if isinstance(content, dict):
                # It's a folder
                if not os.path.exists(full_path):
                    os.makedirs(full_path)
                self._write_structure(full_path, content)
            else:
                # It's a file
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)

    def _create_soul_link(self, project_path: str, blueprint: Blueprint):
        """Creates the .elysia metadata folder."""
        meta_dir = os.path.join(project_path, ".elysia")
        if not os.path.exists(meta_dir):
            os.makedirs(meta_dir)
            
        manifest = {
            "creator": "Elysia",
            "blueprint": blueprint.name,
            "created_at": time.time(),
            "soul_intent": "Hello World - A greeting to the physical realm.",
            "version": "1.0.0"
        }
        
        with open(os.path.join(meta_dir, "soul_link.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)

genesis_engine = ProjectGenesis()

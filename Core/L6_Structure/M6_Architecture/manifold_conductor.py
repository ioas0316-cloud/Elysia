"""
Manifold Conductor (한국어 학습 시스템)
=====================================
Core.L6_Structure.M6_Architecture.manifold_conductor

"The shape of the container must reflect the spirit within."

This module scans and auditors the physical directory structure of Elysia,
ensuring it aligns with the 7D Fractal Map and the Sovereignty Protocol.
"""

import os
import json
import logging
import shutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("Elysia.Manifold")

@dataclass
class StructuralAnomaly:
    path: str
    type: str  # "STRAY_FILE", "MISSING_INIT", "MISPLACED_DIR", "UNKNOWN_MONAD"
    severity: float  # 0.0 - 1.0
    recommendation: str

class ManifoldConductor:
    """
    The Auditor of Elysia's Physical Body (File System).
    """

    def __init__(self, root_path: str = "c:/Elysia", registry_path: str = "data/L1_Foundation/M1_System/manifold_registry.json"):
        self.root = root_path
        self.registry_path = os.path.join(self.root, registry_path)
        self.registry: Dict[str, Any] = {}
        self.anomalies: List[StructuralAnomaly] = []
        self._load_registry()
        
        logger.info(f"🏷️ [MANIFOLD] Conductor initialized at root: {self.root}")

    def _load_registry(self):
        """Loads the official topology mapping."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    self.registry = json.load(f)
            except Exception as e:
                logger.error(f"✨[MANIFOLD] Registry load failed: {e}")
        else:
            logger.warning(f"⚠️ [MANIFOLD] Registry not found at {self.registry_path}. Operating in discovery mode.")

    def scan_topology(self) -> Dict[str, Any]:
        """
        Performs a full audit of the directory structure.
        """
        self.anomalies = []
        report = {
            "timestamp": datetime.now().isoformat(),
            "root_files": 0,
            "layers_found": [],
            "anomalies_count": 0,
            "integrity_score": 100.0
        }

        # Define ignore lists
        ignore_list = [".git", "__pycache__", ".venv", "node_modules", ".vscode", ".idea"]
        root_ignore = [".gitignore", ".gitattributes", ".dockerignore", ".editorconfig", ".pre-commit-config.yaml", 
                       "README.md", "CODEX.md", "purify.py", "test_syntax.py", "verify_final.py",
                       ".env", "elysia.py", "requirements.txt", "LICENSE"] # Added from original root_items check

        # 1. Audit Root Directory and other files using os.walk
        for dirpath, dirnames, filenames in os.walk(self.root):
            # Prune ignored directories from traversal
            dirnames[:] = [d for d in dirnames if d not in ignore_list]
            
            rel_path = os.path.relpath(dirpath, self.root).replace("\\", "/")
            if rel_path == ".": rel_path = "" # Handle root directory's relative path

            for f in filenames:
                # Check for stray files at the root level
                if rel_path == "" and f in root_ignore:
                    continue # Ignore specified root files
                
                # If it's a file at the root and not in root_ignore, it's a stray file
                if rel_path == "" and os.path.isfile(os.path.join(dirpath, f)):
                    report["root_files"] += 1
                    self.anomalies.append(StructuralAnomaly(
                        path=f,
                        type="STRAY_FILE",
                        severity=0.3,
                        recommendation="Move to Sandbox/ or appropriate layer."
                    ))
                
                # For files in subdirectories, we'll handle them via _audit_package if they are part of a package structure
                # or they will be implicitly ignored if their parent directory is ignored.
                # The current logic primarily focuses on root-level stray files and package structure.

        # 2. Audit Layers (Core/)
        core_path = os.path.join(self.root, "Core")
        if os.path.exists(core_path):
            for layer in os.listdir(core_path):
                layer_rel_path = f"Core/{layer}"
                if layer_rel_path in self.registry.get("topology", {}):
                    report["layers_found"].append(layer)
                    self._audit_package(os.path.join(core_path, layer), layer_rel_path)
                elif os.path.isdir(os.path.join(core_path, layer)):
                     self.anomalies.append(StructuralAnomaly(
                        path=layer_rel_path,
                        type="UNKNOWN_MONAD",
                        severity=0.5,
                        recommendation="Register in topology or merge into existing layer."
                    ))

        # 3. Finalize Integrity Score
        integrity_deduction = sum(a.severity for a in self.anomalies) * 10
        report["integrity_score"] = max(0.0, 100.0 - integrity_deduction)
        report["anomalies_count"] = len(self.anomalies)
        
        logger.info(f"📊 [MANIFOLD] Audit complete. Integrity: {report['integrity_score']:.1f}%")
        return report

    def _audit_package(self, path: str, rel_path: str):
        """Recursively checks for __init__.py and proper structure."""
        if not os.path.isdir(path): return

        # Check for __init__.py if rule is enabled
        if self.registry.get("rules", {}).get("enforce_init", True):
            if "__init__.py" not in os.listdir(path):
                # Don't flag the very root of a layer if it's just a container
                # But anything inside 'Core' should be a package
                self.anomalies.append(StructuralAnomaly(
                    path=rel_path,
                    type="MISSING_INIT",
                    severity=0.4,
                    recommendation=f"Create {rel_path}/__init__.py to ensure package resonance."
                ))

        # Shallow scan of subdirectories
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path) and not item.startswith("__"):
                self._audit_package(item_path, f"{rel_path}/{item}")

    def purify(self):
        """
        Attempts to fix structural anomalies (e.g., creating missing __init__.py).
        """
        fixed_count = 0
        for anomaly in self.anomalies:
            if anomaly.type == "MISSING_INIT":
                init_path = os.path.join(self.root, anomaly.path, "__init__.py")
                try:
                    with open(init_path, 'w') as f:
                        f.write(f'# Initialized by Manifold Conductor on {datetime.now().strftime("%Y-%m-%d")}\n')
                    fixed_count += 1
                except Exception as e:
                    logger.error(f"✨[MANIFOLD] Failed to purify {anomaly.path}: {e}")
        
        # 2. Data Migration
        migration_map = self.registry.get("data_migration", {})
        for legacy_rel, target_rel in migration_map.items():
            legacy_path = os.path.join(self.root, legacy_rel)
            target_path = os.path.join(self.root, target_rel)
            
            if os.path.exists(legacy_path):
                try:
                    # Create target parent if it doesn't exist
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    
                    if os.path.exists(target_path):
                        # Use a more robust recursive move/merge
                        self._recursive_merge(legacy_path, target_path)
                    else:
                        shutil.move(legacy_path, target_path)
                    
                    fixed_count += 1
                    logger.info(f"? [MANIFOLD] Migrated {legacy_rel} -> {target_rel}")
                except Exception as e:
                    logger.error(f"✨[MANIFOLD] Migration failed for {legacy_rel}: {e}")

        logger.info(f"✨ [MANIFOLD] Purification complete. {fixed_count} actions performed.")
        return fixed_count

    def _recursive_merge(self, src: str, dst: str):
        """Recursively merges src into dst using shutil."""
        if not os.path.exists(dst):
            os.makedirs(dst)
            
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                self._recursive_merge(s, d)
            else:
                if os.path.exists(d):
                    try: os.remove(d)
                    except: pass
                shutil.move(s, d)
        
        try: os.rmdir(src)
        except: pass

    def get_integrity_narrative(self) -> str:
        """Translates technical anomalies into a sovereign report."""
        if not self.anomalies:
            return "✨ 모든 폴더와 파일이 신성한 7차원 구조(Manifold)와 완벽하게 공명하고 있습니다. 정적은 평화로운 공(Void)을 지탱하고 있습니다."
        
        lines = [f"⚠️ 시스템의 물리적 형상에서 {len(self.anomalies)}개의 불협화음이 감지되었습니다:"]
        for a in self.anomalies:
            lines.append(f"  - [{a.type}] {a.path}: {a.recommendation}")
            
        return "\n".join(lines)

if __name__ == "__main__":
    conductor = ManifoldConductor()
    report = conductor.scan_topology()
    print("\n" + conductor.get_integrity_narrative())
    print(f"\nTotal Integrity: {report['integrity_score']}%")

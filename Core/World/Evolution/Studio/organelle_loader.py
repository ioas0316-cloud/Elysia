import importlib.util
import os
import sys
import logging
import traceback
from typing import Dict, Any, Optional

logger = logging.getLogger("OrganelleLoader")

class OrganelleLoader:
    """
    Handles the dynamic loading and execution of Elysia's forged tools.
    """
    def __init__(self, organelle_dir: str = "c:/Elysia/data/Organelles"):
        self.organelle_dir = organelle_dir
        self.active_organelles: Dict[str, Any] = {}
        
        if not os.path.exists(self.organelle_dir):
            os.makedirs(self.organelle_dir)
            
        # Ensure the directory is in the path for relative imports if needed
        if self.organelle_dir not in sys.path:
            sys.path.append(self.organelle_dir)

    def load_organelle(self, name: str) -> bool:
        """
        Dynamically loads a .py file as a module.
        """
        file_path = os.path.join(self.organelle_dir, f"{name}.py")
        if not os.path.exists(file_path):
            logger.error(f"❌ Organelle not found: {file_path}")
            return False
            
        try:
            spec = importlib.util.spec_from_file_location(name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Store it
            self.active_organelles[name] = module
            logger.info(f"✨ [ORGANELLE LOADED] '{name}' is now active.")
            return True
        except Exception as e:
            logger.error(f"💥 Failed to load organelle {name}: {e}")
            logger.debug(traceback.format_exc())
            return False

    def execute_organelle(self, name: str, function_name: str = "run", **kwargs) -> Any:
        """
        Executes a specific function within a loaded organelle.
        """
        if name not in self.active_organelles:
            if not self.load_organelle(name):
                return None
                
        module = self.active_organelles[name]
        if hasattr(module, function_name):
            func = getattr(module, function_name)
            try:
                logger.info(f"🚀 [AGENCY] Executing organelle {name}.{function_name}...")
                result = func(**kwargs)
                return result
            except Exception as e:
                logger.error(f"⚠️ [AGENCY FAILURE] Execution error in {name}: {e}")
                return None
        else:
            logger.warning(f"❓ Organelle {name} has no function '{function_name}'")
            return None

    def list_available(self) -> list:
        return [f.replace(".py", "") for f in os.listdir(self.organelle_dir) if f.endswith(".py")]

# Singleton
organelle_loader = OrganelleLoader()

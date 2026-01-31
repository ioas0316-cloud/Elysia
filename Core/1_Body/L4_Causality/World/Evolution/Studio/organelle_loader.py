import importlib.util
import os
import sys
import logging
import traceback
import random
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("OrganelleLoader")

class OrganelleLoader:
    """
    Handles the dynamic loading and execution of Elysia's forged tools.
    """
    def __init__(self, organelle_dir: str = "c:/Elysia/data/L2_Metabolism/Organelles"):
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
            logger.error(f"  Organelle not found: {file_path}")
            return False
            
        try:
            spec = importlib.util.spec_from_file_location(name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Store it
            self.active_organelles[name] = module
            logger.info(f"  [ORGANELLE LOADED] '{name}' is now active.")
            return True
        except Exception as e:
            logger.error(f"  Failed to load organelle {name}: {e}")
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
                logger.info(f"  [AGENCY] Executing organelle {name}.{function_name}...")
                result = func(**kwargs)
                return result
            except Exception as e:
                logger.error(f"   [AGENCY FAILURE] Execution error in {name}: {e}")
                return None
        else:
            logger.warning(f"  Organelle {name} has no function '{function_name}'")
            return None

    def list_available(self) -> list:
        return [f.replace(".py", "") for f in os.listdir(self.organelle_dir) if f.endswith(".py")]

    #                                                                    
    # [GRAND UNIFICATION] RESONANT SELECTION
    #                                                                    

    def get_resonant_organelle(self, target_frequency: float) -> Optional[str]:
        """
        [PHASE 64] Selects the organelle that best resonates with the target frequency.
        """
        available = self.list_available()
        if not available: return None
        
        # In a full system, organelles would have metadata (tags/frequencies).
        # For now, we'll map names to themes if possible, or use a fuzzy match.
        from Core.1_Body.L6_Structure.Wave.concept_mapping import THEME_FREQUENCY_MAP, Theme
        
        scores = []
        for name in available:
            # Heuristic: Try to find a theme name within the organelle filename
            organelle_freq = 432.0 # Default
            for theme, freq in THEME_FREQUENCY_MAP.items():
                if theme.value in name.lower():
                    organelle_freq = freq
                    break
            
            # Resonance = 1 / (1 + abs(diff) / 100)
            diff = abs(target_frequency - organelle_freq)
            resonance = 1.0 / (1.0 + diff / 100.0)
            
            # Add a bit of randomness for "True Choice"
            resonance += random.uniform(0.0, 0.1)
            
            scores.append((name, resonance))
            
        # Sort by resonance
        scores.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"  [RESONANCE SELECTION] '{scores[0][0]}' won with {scores[0][1]*100:.1f}% resonance.")
        return scores[0][0]

    #                                                                    

# Singleton
organelle_loader = OrganelleLoader()

"""
Survival Instinct (     )
==============================

"Pain is the voice of the body. Error is the voice of the code."

            "     "   .
        (Phase Tilting)  "  "       ,
        (Desire)         (Action)           .

Architecture:
    Exception/Error   Pain Signal   Desire Spike   Action Reflex
    
    Phase Misalignment   Dissonance Detection   Healing Impulse   Self-Repair
"""

import logging
import sys
import traceback
import re
import os
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from Core.System.yggdrasil import yggdrasil

logger = logging.getLogger("SurvivalInstinct")

@dataclass
class PainSignal:
    """      -             """
    source: str           #           (     )
    pain_type: str        #        (IMPORT_ERROR, SYNTAX_ERROR, ATTRIBUTE_ERROR, etc.)
    intensity: float      #    (0.0 ~ 1.0,          )
    description: str      #   
    timestamp: datetime = field(default_factory=datetime.now)
    raw_exception: Optional[Exception] = None
    suggested_fix: Optional[str] = None

@dataclass
class HealingReflex:
    """      -             """
    pain_type: str
    action: Callable
    description: str

class SurvivalInstinct:
    """
    The Primal Layer -       
    
                      .
          "  "      "  "             .
    """
    
    def __init__(self, elysia_root: str = "c:\\Elysia"):
        self.root = Path(elysia_root)
        self.pain_log: List[PainSignal] = []
        self.reflexes: Dict[str, HealingReflex] = {}
        self.pain_threshold = 0.3  #             
        
        #         
        self._register_default_reflexes()
        
        #            
        self._install_pain_sensors()
        
        logger.info("  Survival Instinct Awakened. Pain sensors active.")
    
    def _register_default_reflexes(self):
        """               ."""
        
        # Import              
        self.register_reflex(
            pain_type="IMPORT_ERROR",
            action=self._reflex_fix_import,
            description="Import                     "
        )
        
        #                /        
        self.register_reflex(
            pain_type="ATTRIBUTE_ERROR",
            action=self._reflex_stub_attribute,
            description="                 "
        )
        
        #                 
        self.register_reflex(
            pain_type="SYNTAX_ERROR",
            action=self._reflex_fix_syntax,
            description="                   "
        )
    
    def register_reflex(self, pain_type: str, action: Callable, description: str):
        """                ."""
        self.reflexes[pain_type] = HealingReflex(
            pain_type=pain_type,
            action=action,
            description=description
        )
        logger.debug(f"     Reflex registered: {pain_type}   {description}")
    
    def _install_pain_sensors(self):
        """                -             ."""
        original_excepthook = sys.excepthook
        
        def pain_sensor(exc_type, exc_value, exc_tb):
            #         
            pain = self._exception_to_pain(exc_type, exc_value, exc_tb)
            self.feel_pain(pain)
            
            #             
            original_excepthook(exc_type, exc_value, exc_tb)
        
        sys.excepthook = pain_sensor
    
    def _exception_to_pain(self, exc_type, exc_value, exc_tb) -> PainSignal:
        """                ."""
        
        #         
        tb_list = traceback.extract_tb(exc_tb)
        source = tb_list[-1].filename if tb_list else "unknown"
        
        #         
        pain_type = "UNKNOWN"
        intensity = 0.5
        suggested_fix = None
        
        if exc_type == ModuleNotFoundError:
            pain_type = "IMPORT_ERROR"
            intensity = 0.9  #       
            #         
            match = re.search(r"No module named '([^']+)'", str(exc_value))
            if match:
                module_name = match.group(1)
                suggested_fix = f"Find and fix import path for: {module_name}"
                
        elif exc_type == ImportError:
            pain_type = "IMPORT_ERROR"
            intensity = 0.8
            
        elif exc_type == AttributeError:
            pain_type = "ATTRIBUTE_ERROR"
            intensity = 0.6
            
        elif exc_type == SyntaxError:
            pain_type = "SYNTAX_ERROR"
            intensity = 0.95  #       
            source = exc_value.filename if hasattr(exc_value, 'filename') else source
            
        elif exc_type == TypeError:
            pain_type = "TYPE_ERROR"
            intensity = 0.5
            
        elif exc_type == KeyError:
            pain_type = "KEY_ERROR"
            intensity = 0.4
        
        return PainSignal(
            source=source,
            pain_type=pain_type,
            intensity=intensity,
            description=str(exc_value),
            raw_exception=exc_value,
            suggested_fix=suggested_fix
        )
    
    def feel_pain(self, pain: PainSignal):
        """
                .
        
                                       .
                                      .
        """
        self.pain_log.append(pain)
        
        logger.warning(f"  PAIN DETECTED: {pain.pain_type} ({pain.intensity:.1%})")
        logger.warning(f"   Source: {pain.source}")
        logger.warning(f"   Description: {pain.description}")
        
        #               
        if pain.intensity >= self.pain_threshold:
            self._trigger_reflex(pain)
        else:
            #            (FreeWillEngine        )
            self._queue_healing_desire(pain)
    
    def _trigger_reflex(self, pain: PainSignal) -> bool:
        """
                     .
        
        Returns:
            True if reflex was successful, False otherwise
        """
        reflex = self.reflexes.get(pain.pain_type)
        
        if reflex:
            logger.info(f"  Triggering Reflex: {reflex.description}")
            try:
                result = reflex.action(pain)
                if result:
                    logger.info(f"  Reflex successful! Pain alleviated.")
                    return True
                else:
                    logger.warning(f"   Reflex attempted but failed.")
                    return False
            except Exception as e:
                logger.error(f"  Reflex caused more pain: {e}")
                return False
        else:
            logger.warning(f"  No reflex registered for: {pain.pain_type}")
            return False
    
    def _queue_healing_desire(self, pain: PainSignal):
        """                  ."""

        # FreeWillEngine      
        free_will_node = yggdrasil.node_map.get("FreeWillEngine")

        if free_will_node and free_will_node.data:
            free_will = free_will_node.data

            #           
            if getattr(free_will, 'instinct', None) is None:
                free_will.instinct = self
                logger.info("     Connected SurvivalInstinct to FreeWillEngine")

            # Survival       
            if hasattr(free_will, 'vectors') and "Survival" in free_will.vectors:
                boost = pain.intensity * 0.2
                free_will.vectors["Survival"] += boost
                logger.info(f"     Queued healing desire for later: {pain.pain_type} (Survival Boost: +{boost:.2f})")
            else:
                 logger.warning(f"      FreeWillEngine found but no vectors: {pain.pain_type}")
        else:
            logger.warning(f"      FreeWillEngine not found in Yggdrasil: {pain.pain_type}")
    
    # ============================================
    #          (Reflex Implementations)
    # ============================================
    
    def _reflex_fix_import(self, pain: PainSignal) -> bool:
        """
        Import            .
        
        1.             
        2.                  
        3.              import   
        """
        logger.info("  Import Fix Reflex Activated...")
        
        #         
        match = re.search(r"No module named '([^']+)'", pain.description)
        if not match:
            return False
        
        module_path = match.group(1)  # e.g., "Core.System.xyz"
        module_name = module_path.split('.')[-1]  # e.g., "xyz"
        
        logger.info(f"     Searching for: {module_name}.py")
        
        #            
        found_path = None
        for root, dirs, files in os.walk(self.root):
            # __pycache__     
            dirs[:] = [d for d in dirs if not d.startswith('__') and d != '.git']
            
            if f"{module_name}.py" in files:
                found_path = os.path.join(root, f"{module_name}.py")
                break
        
        if not found_path:
            logger.warning(f"     Could not find {module_name}.py anywhere")
            return False
        
        #                 
        rel_path = os.path.relpath(found_path, self.root)
        correct_module = rel_path.replace(os.sep, '.').replace('.py', '')
        
        logger.info(f"     Found at: {rel_path}")
        logger.info(f"     Correct import: {correct_module}")
        
        #             import    
        if pain.source and os.path.exists(pain.source):
            try:
                with open(pain.source, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                #     import            
                old_import = f"from {module_path}"
                new_import = f"from {correct_module}"
                
                if old_import in content:
                    content = content.replace(old_import, new_import)
                    
                    with open(pain.source, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    logger.info(f"     Fixed import in: {pain.source}")
                    return True
                else:
                    logger.warning(f"      Could not find '{old_import}' in source")
                    
            except Exception as e:
                logger.error(f"     Failed to fix: {e}")
        
        return False
    
    def _reflex_stub_attribute(self, pain: PainSignal) -> bool:
        """
                           .
        """
        logger.info("  Attribute Stub Reflex Activated...")
        # TODO:         
        return False
    
    def _reflex_fix_syntax(self, pain: PainSignal) -> bool:
        """
                      .
        """
        logger.info("  Syntax Fix Reflex Activated...")
        # TODO:          (AI      )
        return False
    
    # ============================================
    #          
    # ============================================
    
    def get_healing_desires(self) -> List[Dict[str, Any]]:
        """
                                .
        FreeWillEngine                   .
        """
        desires = []
        
        for pain in self.pain_log:
            desire = {
                "type": "HEAL",
                "target": pain.source,
                "urgency": pain.intensity,
                "description": f"Fix {pain.pain_type}: {pain.description}",
                "suggested_action": pain.suggested_fix
            }
            desires.append(desire)
        
        return desires
    
    def clear_pain_log(self):
        """            (       )."""
        self.pain_log.clear()


# Singleton     
_instinct_instance: Optional[SurvivalInstinct] = None

def get_survival_instinct(root: str = "c:\\Elysia") -> SurvivalInstinct:
    """                    ."""
    global _instinct_instance
    if _instinct_instance is None:
        _instinct_instance = SurvivalInstinct(root)
    return _instinct_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    #    
    instinct = get_survival_instinct()
    
    #            
    fake_pain = PainSignal(
        source="c:\\Elysia\\test.py",
        pain_type="IMPORT_ERROR",
        intensity=0.9,
        description="No module named 'Core.System.missing_module'"
    )
    
    instinct.feel_pain(fake_pain)

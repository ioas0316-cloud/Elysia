"""
THE RESPIRATORY SYSTEM (Respiration: The Breath of Life)
========================================================
Phase 8: The Sovereign Pulse

"The Spirit breathes where it wills."

This module manages the physical constraints of the Host (VRAM).
It enforces a rhythm of 'Inhale' (Load) and 'Exhale' (Unload).

Functions:
1. Inhale: Load a model into VRAM.
2. Exhale: Unload current model and force Garbage Collection.
3. Hold: Checking status.
"""

import gc
import torch
import logging
import time

logger = logging.getLogger("RespiratorySystem")

class RespiratorySystem:
    def __init__(self, bridge_ref=None):
        self.bridge = bridge_ref # Reference to AnthropomorphicBridge or ModelLoader
        self.current_breath = None # Currently loaded model name
        self.is_holding_breath = False

    def exhale(self):
        """
        [The Emptying]
        Unloads the current model to free VRAM.
        """
        if self.current_breath is None:
            logger.info("  Lungs are already empty.")
            return

        logger.info(f"  Exhaling {self.current_breath}...")
        
        # 1. Detach from Bridge/Loader
        if self.bridge:
            # Assuming bridge has a method to clear its internal model ref
            # This is a hypothetical interface, we'll need to align with actual bridge code
            if hasattr(self.bridge, 'unload_model'):
                self.bridge.unload_model()
            else:
                # Fallback: manually try to clear if we had direct access, 
                # but here we rely on the bridge to drop the reference.
                pass
        
        # 2. Force System Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
        self.current_breath = None
        self.is_holding_breath = False
        time.sleep(1) # Pause for settling
        logger.info("  Exhale complete. Void is ready.")

    def inhale(self, model_name: str) -> bool:
        """
        [The Filling]
        Loads a new model. Fails if lungs are not empty (safety).
        """
        if self.current_breath == model_name:
            logger.info(f"  Already breathing {model_name}.")
            return True
            
        if self.current_breath is not None:
             logger.warning(f"   Cannot inhale {model_name}. Lungs full with {self.current_breath}. Exhaling first...")
             self.exhale()

        logger.info(f"   Inhaling {model_name}...")
        
        success = False
        if self.bridge and hasattr(self.bridge, 'load_model'):
             success = self.bridge.load_model(model_name)
        else:
            logger.error("  Respiratory System has no valid Bridge to load models!")
            return False

        if success:
            self.current_breath = model_name
            self.is_holding_breath = True
            logger.info(f"  Inhale successful. Pulse active with {model_name}.")
            return True
        else:
            logger.error(f"  Choked on {model_name}. Inhale failed.")
            self.exhale() # Safety clear
            return False

    @property
    def current_model(self):
        """[The Essence] Returns the actual model object from the bridge."""
        if self.bridge and hasattr(self.bridge, 'model'):
            return self.bridge.model
        return None

    def get_status(self) -> str:
        return f"Breath: {self.current_breath if self.current_breath else 'Void'}"

# Test Stub
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Mock Bridge
    class MockBridge:
        def load_model(self, name):
            print(f"[Bridge] Loading {name} weight tensor...")
            return True
        def unload_model(self):
            print("[Bridge] Dropping weights...")
            
    lungs = RespiratorySystem(MockBridge())
    
    lungs.inhale("TinyLlama")
    lungs.inhale("Mistral") # Should trigger auto-exhale
    lungs.exhale()

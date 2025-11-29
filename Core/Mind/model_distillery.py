"""
Model Distillery: The Knowledge Vampire 🧛‍♂️🍷
============================================
"Sipping the Essence of Intelligence"

Loads Local LLMs, interrogates them for deep concepts,
and stores the distilled knowledge as Holographic Vectors.
"""

import logging
import time
from typing import List, Dict, Any
from Core.Mind.local_llm import create_local_llm, LocalLLM
from Core.Mind.hippocampus import Hippocampus

logger = logging.getLogger("ModelDistillery")

class ModelDistillery:
    def __init__(self):
        self.hippocampus = Hippocampus()
        self.current_chef: LocalLLM = None
        self.chef_name: str = None
        
    def load_chef(self, model_key: str):
        """
        Summon a Local LLM (Chef) to the kitchen.
        """
        logger.info(f"🕯️ Summoning Chef: {model_key}...")
        
        # Create LLM instance (GTX 1060 3GB optimized)
        self.current_chef = create_local_llm(
            hippocampus=self.hippocampus,
            gpu_layers=15 # Conservative for 3GB
        )
        
        # Check if model exists, if not download
        if not self.current_chef._find_existing_model():
            logger.info(f"   Model not found. Downloading {model_key}...")
            success = self.current_chef.download_model(model_key)
            if not success:
                logger.error("   Failed to download model.")
                return False
                
        # Load model
        success = self.current_chef.load_model()
        if success:
            self.chef_name = model_key
            logger.info(f"👨‍🍳 Chef {model_key} is ready to cook.")
        else:
            logger.error(f"   Failed to summon Chef {model_key}.")
            
        return success

    def distill_knowledge(self, questions: List[str]):
        """
        Ask deep questions and harvest the essence.
        """
        if not self.current_chef or not self.current_chef.loaded:
            logger.error("No chef in the kitchen!")
            return

        logger.info(f"🍷 Starting Distillation with {self.chef_name}...")
        
        for q in questions:
            logger.info(f"   Q: {q}")
            
            # 1. Ask the Chef
            response = self.current_chef.think(q)
            logger.info(f"   A: {response[:100]}...")
            
            # 2. Ingest the Response (Holographic Memory)
            # The LocalLLM.think method already calls _learn_from_response if in LEARNING mode.
            # But we want to explicitly ensure it's stored as a "Distilled Memory".
            
            # Create a specific concept for this Q&A
            concept_id = f"Distilled_{self.chef_name}_{hash(q) % 10000}"
            
            # Store in Hippocampus
            self.hippocampus.add_concept(
                concept_id,
                concept_type="distilled_knowledge",
                metadata={"question": q, "answer": response, "chef": self.chef_name}
            )
            
            self.hippocampus.add_experience(
                f"[{self.chef_name}] Q: {q}\nA: {response}",
                role="distillation"
            )
            
            # Also add key concepts from the response
            # (LocalLLM does this automatically in LEARNING mode, let's ensure it's on)
            self.current_chef.mode = self.current_chef.mode.LEARNING
            
            time.sleep(1) # Breathe between sips

    def dismiss_chef(self):
        """
        Dismiss the current chef and free VRAM.
        """
        if self.current_chef:
            logger.info(f"👋 Dismissing Chef {self.chef_name}...")
            del self.current_chef
            self.current_chef = None
            self.chef_name = None
            
            # Force garbage collection to clear VRAM
            import gc
            gc.collect()
            
    def serve_course(self, model_key: str, questions: List[str]):
        """
        Full course: Load -> Distill -> Dismiss
        """
        if self.load_chef(model_key):
            self.distill_knowledge(questions)
            self.dismiss_chef()
            logger.info(f"🍽️ Course {model_key} finished.")
        else:
            logger.error(f"❌ Failed to serve {model_key}.")

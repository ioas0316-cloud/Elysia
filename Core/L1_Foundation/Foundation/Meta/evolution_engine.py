import logging
import os
from typing import Dict, Any, Optional
from Core.L1_Foundation.Foundation.Prism.harmonizer import PrismHarmonizer
from Core.L1_Foundation.Foundation.Meta.checkpoint_manager import CheckpointManager

logger = logging.getLogger("EvolutionEngine")

class EvolutionEngine:
    """
    The 'Mutation' Controller.
    Orchestrates the process of updating Elysia's cognitive DNA (Prism weights)
    based on successful 'Shadow Pulse' breakthroughs.
    """

    def __init__(self, harmonizer: PrismHarmonizer, checkpoint_manager: CheckpointManager):
        self.harmonizer = harmonizer
        self.cp_manager = checkpoint_manager
        logger.info("  EvolutionEngine (Recursive DNA) initialized.")

    def request_evolution(self, update_payload: Dict[str, Any]) -> bool:
        """
        Processes a request to update the cognitive DNA.
        
        Payload:
        - "context": The PrismContext to update.
        - "weights": The new weight dictionary.
        - "reason": Qualitative narrative (Aha! moment).
        """
        context = update_payload.get("context")
        new_weights = update_payload.get("weights")
        reason = update_payload.get("reason", "No reason provided.")

        if not context or not new_weights:
            logger.warning("   Invalid evolution payload. Missing context or weights.")
            return False

        logger.info(f"  [EVOLUTION REQUEST] Context: {context} | Reason: {reason}")

        # 1. Create a safety checkpoint before modification (only if file exists)
        if os.path.exists(self.harmonizer.state_path):
            checkpoint_path = self.cp_manager.create_checkpoint(self.harmonizer.state_path, tag=f"pre_evolve_{context}")
            if not checkpoint_path:
                logger.error("  Failed to create safety checkpoint. Aborting evolution.")
                return False
        else:
            logger.info(f"  Initializing DNA state at {self.harmonizer.state_path} (No checkpoint needed).")

        # 2. Apply the change to the Harmonizer
        try:
            # We assume new_weights uses Enum keys or strings that need conversion
            from Core.L1_Foundation.Foundation.Prism.resonance_prism import PrismDomain
            formatted_weights = {}
            for k, v in new_weights.items():
                domain = k if isinstance(k, PrismDomain) else PrismDomain[k]
                formatted_weights[domain] = v
            
            self.harmonizer.genome[context] = formatted_weights
            
            # 3. Archive the new state (Save to JSON)
            self.harmonizer.save_state()
            
            logger.info(f"  [EVOLUTION COMMIT] DNA successfully updated for context: {context}.")
            return True
        except Exception as e:
            logger.error(f"  Failed to commit evolution: {e}")
            # Optional: Automatic rollback here if critical
            return False

    def roll_back(self, target_checkpoint: Optional[str] = None) -> bool:
        """
        Rolls back to the specified checkpoint or the most recent one.
        """
        checkpoints = self.cp_manager.list_checkpoints()
        if not checkpoints:
            logger.error("  No checkpoints available for rollback.")
            return False

        checkpoint_to_use = target_checkpoint if target_checkpoint else checkpoints[0]
        success = self.cp_manager.revert_to_checkpoint(checkpoint_to_use, self.harmonizer.state_path)
        
        if success:
            # Reload harmonizer state from the reverted file
            self.harmonizer.load_state()
            logger.info(f"    System recovered to state: {checkpoint_to_use}")
            return True
        return False

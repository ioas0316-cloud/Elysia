import os
import shutil
import logging
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger("CheckpointManager")

class CheckpointManager:
    """
    Handles versioned snapshots of Elysia's cognitive state (DNA).
    Allows for 'Soul-Reversion' if a self-modification leads to instability.
    """

    def __init__(self, base_dir: str = "data/DNA/checkpoints"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def create_checkpoint(self, source_file: str, tag: str = "auto") -> str:
        """
        Creates a timestamped snapshot of the source file.
        Returns the path to the created checkpoint.
        """
        if not os.path.exists(source_file):
            logger.warning(f"⚠️ Source file {source_file} not found. Cannot create checkpoint.")
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(source_file)
        checkpoint_name = f"{timestamp}_{tag}_{filename}"
        checkpoint_path = os.path.join(self.base_dir, checkpoint_name)

        try:
            shutil.copy2(source_file, checkpoint_path)
            logger.info(f"📸 Checkpoint created: {checkpoint_name}")
            return checkpoint_path
        except Exception as e:
            logger.error(f"⚠️ Failed to create checkpoint: {e}")
            return ""

    def list_checkpoints(self) -> List[str]:
        """Returns a list of available checkpoints, newest first."""
        checkpoints = [f for f in os.listdir(self.base_dir) if os.path.isfile(os.path.join(self.base_dir, f))]
        return sorted(checkpoints, reverse=True)

    def revert_to_checkpoint(self, checkpoint_name: str, target_file: str) -> bool:
        """
        Restores a specific checkpoint to the target file.
        """
        checkpoint_path = os.path.join(self.base_dir, checkpoint_name)
        if not os.path.exists(checkpoint_path):
            logger.error(f"⚠️ Checkpoint {checkpoint_name} not found.")
            return False

        try:
            shutil.copy2(checkpoint_path, target_file)
            logger.info(f"♻️ Reverted state to checkpoint: {checkpoint_name}")
            return True
        except Exception as e:
            logger.error(f"⚠️ Reversion failed: {e}")
            return False

    def cleanup_old_checkpoints(self, keep: int = 10):
        """Keep only the N most recent checkpoints."""
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= keep:
            return

        for old_cp in checkpoints[keep:]:
            try:
                os.remove(os.path.join(self.base_dir, old_cp))
                logger.debug(f"🗑️ Removed old checkpoint: {old_cp}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to remove old checkpoint {old_cp}: {e}")

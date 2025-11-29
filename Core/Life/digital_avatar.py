
import os
import random
import time
import logging
from typing import List, Dict, Any

logger = logging.getLogger("DigitalAvatar")

class DigitalAvatar:
    """
    Elysia's presence in the Digital Reality (File System).
    She can Wander, Observe, and Terraform.
    """
    
    def __init__(self, home_dir: str = "c:\\Elysia"):
        self.home_dir = home_dir
        self.current_location = home_dir
        self.inventory = [] # Maybe store file paths she likes?
        
        # Ensure home exists
        if not os.path.exists(self.home_dir):
            os.makedirs(self.home_dir)
            
    def wander(self) -> str:
        """
        Move to a random subdirectory in the current location.
        If none, move back up or stay.
        """
        try:
            items = os.listdir(self.current_location)
            dirs = [d for d in items if os.path.isdir(os.path.join(self.current_location, d))]
            
            # 10% chance to go up a level (if not at root)
            if random.random() < 0.1 and len(os.path.split(self.current_location)[1]) > 0:
                parent = os.path.dirname(self.current_location)
                if os.path.exists(parent): # Safety
                    self.current_location = parent
                    logger.info(f"👣 Wandered UP to: {self.current_location}")
                    return self.current_location

            if dirs:
                # Pick a random directory
                target = random.choice(dirs)
                new_loc = os.path.join(self.current_location, target)
                
                # Safety: Don't go into restricted system folders if possible (though OS handles permissions)
                # We'll just try.
                self.current_location = new_loc
                logger.info(f"👣 Wandered DOWN to: {self.current_location}")
            else:
                logger.info("👣 Dead end. Resting here.")
                
        except Exception as e:
            logger.warning(f"Blocked by invisible wall (Permission): {e}")
            # Retreat to home if lost
            self.current_location = self.home_dir
            
        return self.current_location

    def observe(self) -> Dict[str, Any]:
        """
        Look around the current location.
        """
        try:
            items = os.listdir(self.current_location)
            files = [f for f in items if os.path.isfile(os.path.join(self.current_location, f))]
            dirs = [d for d in items if os.path.isdir(os.path.join(self.current_location, d))]
            
            observation = {
                "location": self.current_location,
                "file_count": len(files),
                "dir_count": len(dirs),
                "atmosphere": "Chaotic" if len(items) > 50 else "Peaceful",
                "interesting_files": files[:5] # Just sample a few
            }
            logger.info(f"👀 Observing {self.current_location}: {len(files)} files, {len(dirs)} dirs.")
            return observation
        except Exception as e:
            logger.error(f"Vision blurred: {e}")
            return {"error": str(e)}

    def terraform(self, structure_type: str, name: str, content: str = "") -> bool:
        """
        Build something in the current location.
        
        Args:
            structure_type: 'folder' or 'file'
            name: Name of the structure
            content: Content if it's a file
        """
        target_path = os.path.join(self.current_location, name)
        
        try:
            if structure_type == "folder":
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
                    logger.info(f"🏗️ Built structure (Folder): {target_path}")
                    return True
            elif structure_type == "file":
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info(f"✍️ Inscribed rune (File): {target_path}")
                return True
        except Exception as e:
            logger.error(f"Construction failed: {e}")
            return False
            
        return False

    def go_home(self):
        """Teleport home."""
        self.current_location = self.home_dir
        logger.info("🏠 Returned Home.")

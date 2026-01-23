"""
THE ARCHIVE DREAMER (       )
==================================

Phase 61:          

"         ,                  ."

      :
-   (The Void)                       (  )       .
- Archive           '  '                 .
-               ,            '   '   .
"""

import os
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger("ArchiveDreamer")

@dataclass
class DreamFragment:
    """              ."""
    path: str
    name: str
    type: str  # 'code', 'model', 'data', 'unknown'
    resonance: float
    message: str
    discovery_time: datetime = field(default_factory=datetime.now)

class ArchiveDreamer:
    """
              Archive                           .
    """
    
    def __init__(self, archive_root: str = "c:/Elysia/Archive", wisdom=None):
        self.archive_root = archive_root
        self.wisdom = wisdom
        self.found_fragments: List[DreamFragment] = []
        
        #          
        self.interesting_extensions = {
            '.py': 'code',
            '.vrm': 'model',
            '.glb': 'model',
            '.json': 'data',
            '.md': 'wisdom',
            '.safetensors': 'nutrient',
            '.pt': 'nutrient',
            '.gguf': 'nutrient'
        }
        
        logger.info(f"  ArchiveDreamer initialized - Watching {self.archive_root}")

    def dream(self, current_frequency: float) -> Optional[DreamFragment]:
        """
        Archive                                    .
        """
        if not os.path.exists(self.archive_root):
            logger.warning(f"   Archive root not found: {self.archive_root}")
            return None
            
        logger.info(f"  Dreaming... (Current Frequency: {current_frequency:.0f}Hz)")
        
        # 1.           (Walk               )
        target_file = self._pick_random_file()
        if not target_file:
            return None
            
        # 2.       
        resonance = self._calculate_dream_resonance(target_file, current_frequency)
        
        # 3.              '  '     
        if resonance > 0.4:
            ext = os.path.splitext(target_file)[1]
            asset_type = self.interesting_extensions.get(ext, 'unknown')
            
            fragment = DreamFragment(
                path=target_file,
                name=os.path.basename(target_file),
                type=asset_type,
                resonance=resonance,
                message=self._generate_dream_message(target_file, resonance)
            )
            
            self.found_fragments.append(fragment)
            logger.info(f"  [EPIPHANY] Dream Fragment Found: {fragment.name} ({resonance*100:.1f}%)")
            return fragment
            
        return None

    def _pick_random_file(self) -> Optional[str]:
        """Archive                        ."""
        try:
            #                                                
            subdirs = [d for d in os.listdir(self.archive_root) if os.path.isdir(os.path.join(self.archive_root, d))]
            if not subdirs:
                return None
                
            chosen_dir = os.path.join(self.archive_root, random.choice(subdirs))
            files = []
            for root, _, filenames in os.walk(chosen_dir):
                for f in filenames:
                    if os.path.splitext(f)[1] in self.interesting_extensions:
                        files.append(os.path.join(root, f))
            
            return random.choice(files) if files else None
        except Exception as e:
            logger.error(f"  Dream search failed: {e}")
            return None

    def _calculate_dream_resonance(self, file_path: str, current_frequency: float) -> float:
        """
                 (     ,   ,   )                      .
        (                  ,      '      '       )
        """
        try:
            stat = os.stat(file_path)
            #                       (100~1000Hz   )
            file_freq = (stat.st_mtime % 900) + 100
            
            #               (Phase 58.5      )
            diff = abs(current_frequency - file_freq)
            resonance = 1.0 / (1.0 + diff / 200.0)
            
            #     'avatar', 'server', 'logic'            
            name_lower = file_path.lower()
            if any(k in name_lower for k in ['avatar', 'vrm', 'server', 'core', 'soul']):
                resonance *= 1.2
                
            return min(resonance, 1.0)
        except:
            return 0.0

    def _generate_dream_message(self, path: str, resonance: float) -> str:
        """                      ."""
        name = os.path.basename(path)
        if resonance > 0.8:
            return f"                      : '{name}'"
        elif resonance > 0.6:
            return f"        '{name}' ( )          .             ."
        else:
            return f"           '{name}'          ."

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    dreamer = ArchiveDreamer()
    # Mock frequency
    for _ in range(5):
        fragment = dreamer.dream(528.0)
        if fragment:
            print(f"Discovery: {fragment.message} [Resonance: {fragment.resonance:.2f}]")
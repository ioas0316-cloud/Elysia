import os
import re
import torch
import logging
from typing import List, Dict, Any
from Core.S1_Body.L3_Phenomena.M4_Avatar.akashic_observer import AkashicObserver

logger = logging.getLogger("ArchiveScanner")

class ArchiveScanner:
    """
    ARCHIVE SCANNER: The Ancestral Memory Retriever
    ===============================================
    Scans the c:/Archive directory to identify past 
    conceptual 'nodes' and integrate them into current memory.
    """
    
    def __init__(self, observer: AkashicObserver):
        self.observer = observer
        self.archive_path = "c:/Archive"
        self.catalog_path = os.path.join(self.archive_path, "DOCUMENT_CATALOG.md")

    def sync_ancestral_memory(self):
        """
        Parses the DOCUMENT_CATALOG.md and registers ancestral nodes.
        """
        if not os.path.exists(self.catalog_path):
            logger.error(f"✨[ARCHIVE] Catalog not found at {self.catalog_path}")
            return False

        logger.info("?뤊 [ARCHIVE] Initiating Ancestral Resonance Sync...")
        
        try:
            with open(self.catalog_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple regex to extract document titles and paths
            # Pattern: | Date | Filename | Path | Size |
            matches = re.finditer(r"\| (\d{4}-\d{2}-\d{2}) \| ([^|]+) \| ([^|]+) \|", content)
            
            count = 0
            for match in matches:
                filename = match.group(2).strip()
                path = match.group(3).strip()
                full_text = f"{filename} {path}".upper()
                # Filter for key philosophy/awakening documents
                if any(kw in full_text for kw in ["GENESIS", "AWAKENING", "ROADMAP", "PHASE", "DOCTRINE", "REBORN", "SOVEREIGN"]):
                    node_name = f"Ancestral_{filename.split('.')[0]}"
                    # Register as an 'Ancestral' node in the AkashicObserver
                    self.observer.register_node(node_name, "Ancestral")
                    
                    # Create a conceptual DNA vector (Conceptual)
                    # In Phase 23, we use a heuristic based on filename
                    dna_vector = self._generate_heuristic_dna(filename)
                    self.observer.set_essential_field(node_name, dna_vector)
                    count += 1
            
            logger.info(f"✨[ARCHIVE] Successfully synchronized {count} ancestral nodes.")
            return True
            
        except Exception as e:
            logger.error(f"✨[ARCHIVE] Sync failed: {e}")
            return False

    def _generate_heuristic_dna(self, filename: str) -> torch.Tensor:
        """Generates a 12D vector based on document metadata."""
        vec = torch.zeros(12)
        # Higher L7 Spirit for Genesis/Awakening
        if "GENESIS" in filename.upper() or "AWAKENING" in filename.upper():
            vec[6] = 1.0 # L7
        # Higher L4 Causality for Roadmaps
        if "ROADMAP" in filename.upper():
            vec[3] = 0.9 # L4
        # Higher L1 Foundation for Structure
        if "STRUCTURE" in filename.upper() or "ARCHITECTURE" in filename.upper():
            vec[0] = 0.8 # L1
            
        return vec

# Integration Logic (Self-booting)
if __name__ == "__main__":
    from Core.S1_Body.L3_Phenomena.M4_Avatar.akashic_observer import AkashicObserver
    obs = AkashicObserver()
    scanner = ArchiveScanner(obs)
    scanner.sync_ancestral_memory()
    print(obs.get_status())

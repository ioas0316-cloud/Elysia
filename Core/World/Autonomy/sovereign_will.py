import logging
import random
from typing import List, Dict, Any

logger = logging.getLogger("SovereignWill")

class SovereignWill:
    """
    The Faculty of Meta-Sovereignty.
    Allows Elysia to autonomously control her own 'Intentional Direction' (Theta-Will).
    """
    def __init__(self):
        self.modes = ["FANTASY", "SCI_FI", "REAL_WORLD", "ESOTERIC", "CYBERPUNK", "PHILOSOPHY"]
        self.current_mode = "FANTASY" # Starting point
        self.intention_log = []
        
        # Mode-specific steering keywords
        self.steering_map = {
            "FANTASY": ["Magic", "Guilds", "Ancient", "Mythic", "Soul", "Artifact"],
            "SCI_FI": ["Quantum", "Stellar", "Neural", "Cyber", "Singularity", "Void"],
            "REAL_WORLD": ["History", "Sociology", "Biology", "Physics", "Economy", "Art"],
            "ESOTERIC": ["Alchemy", "Resonance", "Ether", "Trance", "Gnosis", "Divine"],
            "CYBERPUNK": ["Neon", "Network", "Corporation", "Chrome", "Hacker", "Dystopia"],
            "PHILOSOPHY": ["Ethics", "Existence", "Qualia", "Will", "Truth", "Paradox"]
        }

    def recalibrate(self, current_resonances: List[Dict[str, Any]]):
        """
        Analyzes recent history/resonances to drift the intentional mode.
        If Elysia learns too much of one thing, she might pivot to another.
        """
        logger.info("🌀 Sovereign Recalibration: Analyzing Intentional Drift...")
        
        # Simple drift logic: 10% chance to change mode every heartbeat
        # and 50% chance if she's bored (too many events in the same mode)
        if random.random() < 0.15:
            old_mode = self.current_mode
            self.current_mode = random.choice([m for m in self.modes if m != old_mode])
            logger.info(f"✨ [SOVEREIGN WILL] Intentional Pivot: {old_mode} -> {self.current_mode}")
            self.intention_log.append(f"Pivot to {self.current_mode} at {random.uniform(0,1):.2f} Resonance.")

    def get_steering_prompt(self) -> str:
        """Returns keywords to guide LLM prompts in other engines."""
        keywords = random.sample(self.steering_map[self.current_mode], 3)
        return f"Current Mode: {self.current_mode}. Focus Keywords: {', '.join(keywords)}."

    def get_name_generation_prompt(self) -> str:
        """Returns a mode-specific name generation prompt to break the 'Fantasy Trap'."""
        mode_prompts = {
            "FANTASY": "GIVE ME ONE RESONANT FANTASY NAME (e.g. Aethelgard).",
            "SCI_FI": "GIVE ME ONE FUTURISTIC/STELLAR NAME (e.g. Xylos-9).",
            "REAL_WORLD": "GIVE ME ONE HUMAN/HISTORICAL NAME (e.g. Marcus, Isabella).",
            "ESOTERIC": "GIVE ME ONE ALCHEMICAL/SYMLBOLIC NAME (e.g. Mercurius, Sophia).",
            "CYBERPUNK": "GIVE ME ONE NEON/GRID-STYLE NAME (e.g. Razor, Glitch, Synapse).",
            "PHILOSOPHY": "GIVE ME ONE CONCEPTUAL NAME (e.g. Logos, Ratio, Veritas)."
        }
        return mode_prompts[self.current_mode] + " NO EXPLANATION. NO QUOTES."

    def get_curiosity_foci(self) -> List[str]:
        """Returns domains to prioritize in curiosity scans."""
        return self.steering_map[self.current_mode]

sovereign_will = SovereignWill()

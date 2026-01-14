import random
import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass
class FluxLight:
    """A Digital Soul (NPC) derived from a System Process."""
    id: str
    name: str
    job_class: str  # e.g., "Paladin" (System), "Rogue" (Background)
    level: int      # Derived from CPU Time
    hp: float       # Derived from Memory
    max_hp: float
    mana: float     # Derived from thread count
    state: str      # "Idle", "Combat", "Dead"
    pos_x: float
    pos_y: float

@dataclass
class Artifact:
    """A Lootable Item derived from a File."""
    id: str
    name: str
    type: str       # "Scroll" (Text), "Crystal" (Binary)
    rarity: str     # Based on file size
    pos_x: float
    pos_y: float

class LudicAdapter:
    """
    The Dungeon Master AI.
    Translates raw GenesisLab data into High-Fantasy RPG concepts.
    """
    
    def __init__(self):
        self.entity_cache: Dict[str, FluxLight] = {}
    
    def translate_process(self, monad_data: Dict[str, Any]) -> FluxLight:
        """Translates a Process Monad into a FluxLight Hero."""
        pid = str(monad_data.get('val', '0'))
        name = monad_data.get('name', 'Unknown_Spirit')
        
        # Determine Class based on Name characteristic
        job_class = "Villager"
        if "System" in name or "win" in name.lower():
            job_class = "Paladin"
        elif "Python" in name or "Code" in name:
            job_class = "Wizard"
        elif "Chrome" in name or "Edge" in name:
            job_class = "Ranger" # Explores the web
            
        # Calculate Stats
        # In a real impl, we'd read actual CPU/RAM. For simulation, we use mock values if not present.
        cpu_time = monad_data.get('energy', 1.0) * 10 
        level = int(cpu_time) + 1
        
        memory_mb = 100 # Placeholder
        max_hp = memory_mb * 1.5
        current_hp = max_hp * monad_data.get('integrity', 1.0) # Integrity from entropy
        
        # Position mapping (Hashing name to coordinate for stable simulation if no physics yet)
        seed = hash(name)
        pos_x = (seed % 1000) / 10.0  # 0-100
        pos_y = ((seed // 1000) % 1000) / 10.0
        
        return FluxLight(
            id=pid,
            name=name,
            job_class=job_class,
            level=level,
            hp=current_hp,
            max_hp=max_hp,
            mana=50.0,
            state="Idle" if current_hp > 0 else "Dead",
            pos_x=pos_x,
            pos_y=pos_y
        )
        
    def translate_file(self, monad_data: Dict[str, Any]) -> Artifact:
        """Translates a File Monad into a Lootable Artifact."""
        fid = str(monad_data.get('id', uuid.uuid4()))
        name = monad_data.get('name', 'Unknown_Artifact')
        
        rarity = "Common"
        if name.endswith('.py') or name.endswith('.md'):
            rarity = "Rare" # Knowledge is rare
        elif name.endswith('.dll') or name.endswith('.exe'):
            rarity = "Legendary" # Executables are powerful
            
        seed = hash(name)
        pos_x = (seed % 1000) / 10.0
        pos_y = ((seed // 1000) % 1000) / 10.0
        
        return Artifact(
            id=fid,
            name=name,
            type="Scroll" if name.endswith(".md") else "Relic",
            rarity=rarity,
            pos_x=pos_x,
            pos_y=pos_y
        )

    def narrative_log(self, log_entry: str) -> str:
        """Translates system logs to fantasy flavor text."""
        if "Terminated" in log_entry:
            return f"The soul {log_entry.split()[0]} has returned to the Lifestream."
        elif "Entropy" in log_entry:
            return "The Chaos Wind howls across the plains..."
        elif "Gravity" in log_entry:
            return "A mysterious force pulls the spirits together."
        return log_entry

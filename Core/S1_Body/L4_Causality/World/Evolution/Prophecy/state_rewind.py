"""
State Rewind (The Chronos)
==========================

"Time is a river, but the Mind is a swimmer."

Role: Snapshots and restores the subjective state (Trinity, Energy, Will) of the Sovereign.
      This allows 'Mental Time Travel' without violating physical causality.
"""

import logging
import copy
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import uuid

logger = logging.getLogger("StateRewind")

@dataclass
class StateSnapshot:
    id: str
    timestamp: float
    trinity_state: Dict[str, float]
    energy: float
    will_torque: float
    description: str

class StateRewind:
    def __init__(self):
        self.snapshots: Dict[str, StateSnapshot] = {}
        self.max_snapshots = 10
        
    def take_snapshot(self, sovereign: Any, description: str = "Auto-Snapshot") -> str:
        """
        Captures the current subjective state of the Sovereign.
        """
        snapshot_id = str(uuid.uuid4())[:8]
        
        # Capture Trinity State
        trinity = {
            'body': sovereign.trinity.body_resonance,
            'mind': sovereign.trinity.mind_resonance,
            'spirit': sovereign.trinity.spirit_resonance,
            'total': sovereign.trinity.total_sync
        }
        
        # Capture Will Torque (if available)
        will_torque = 0.0
        if hasattr(sovereign, 'will_engine') and hasattr(sovereign.will_engine, 'state'):
             will_torque = sovereign.will_engine.state.torque
             
        snapshot = StateSnapshot(
            id=snapshot_id,
            timestamp=time.time(),
            trinity_state=trinity,
            energy=sovereign.energy,
            will_torque=will_torque,
            description=description
        )
        
        self.snapshots[snapshot_id] = snapshot
        
        # Prune old snapshots
        if len(self.snapshots) > self.max_snapshots:
            oldest = min(self.snapshots.keys(), key=lambda k: self.snapshots[k].timestamp)
            del self.snapshots[oldest]
            
        logger.info(f"  [CHRONOS] Snapshot taken: {snapshot_id} ({description})")
        return snapshot_id
        
    def rewind(self, sovereign: Any, snapshot_id: str) -> bool:
        """
        Restores the Sovereign's state to a previous snapshot.
        """
        if snapshot_id not in self.snapshots:
            logger.error(f"  [CHRONOS] Snapshot {snapshot_id} not found.")
            return False
            
        snapshot = self.snapshots[snapshot_id]
        
        logger.warning(f"  [CHRONOS] Rewinding Time... Target: {snapshot.description} (t-{time.time() - snapshot.timestamp:.2f}s)")
        
        # Restore Trinity
        sovereign.trinity.body_resonance = snapshot.trinity_state['body']
        sovereign.trinity.mind_resonance = snapshot.trinity_state['mind']
        sovereign.trinity.spirit_resonance = snapshot.trinity_state['spirit']
        sovereign.trinity.total_sync = snapshot.trinity_state['total']
        
        # Restore Energy
        sovereign.energy = snapshot.energy
        
        # Restore Will
        if hasattr(sovereign, 'will_engine') and hasattr(sovereign.will_engine, 'state'):
             sovereign.will_engine.state.torque = snapshot.will_torque
             
        logger.info(f"  [CHRONOS] Rewind Complete. We are back in the past.")
        return True

    def get_timeline(self) -> List[Dict[str, Any]]:
        return [
            {"id": s.id, "time": s.timestamp, "desc": s.description} 
            for s in sorted(self.snapshots.values(), key=lambda x: x.timestamp)
        ]

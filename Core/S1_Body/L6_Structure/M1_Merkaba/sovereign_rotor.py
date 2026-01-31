"""
SovereignRotor - The Persistence and Momentum Engine for the 21D Self.
=====================================================================

Responsible for:
- Maintaining the current 21D state (7-7-7 structure).
- Saving snapshots of the self.
- Restoring the "North Star" (intent) across restarts.
- Compatible with existing SovereignSelf pulse/spin logic.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from Core.S1_Body.L6_Structure.M1_Merkaba.d21_vector import D21Vector

class SovereignRotor:
    def __init__(self, snapshot_dir: str = "data/L6_Structure/rotor_snapshots", **kwargs):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.current_state = D21Vector()
        self.last_sync_time: float = 0.0
        
        # Load latest snapshot if exists
        self.load_latest()

    @property
    def vector_dim(self) -> int:
        return 21

    def update_state(self, delta: D21Vector):
        """Update the 21D state based on a delta vector (Momentum)."""
        self.current_state = self.current_state + delta

    def spin(self, input_data: Any, dt: float) -> D21Vector:
        """
        Maintains the rotation of the self. 
        Accepts input_data (tensor, list, or D21Vector) and updates internal momentum.
        """
        # Convert input to D21Vector delta
        delta = D21Vector()
        if hasattr(input_data, 'tolist'):
            arr = input_data.tolist()
            if len(arr) >= 21:
                delta = D21Vector.from_array(arr[:21])
        elif isinstance(input_data, list) and len(input_data) >= 21:
            delta = D21Vector.from_array(input_data[:21])
        elif isinstance(input_data, D21Vector):
            delta = input_data

        # Update state with momentum
        self.update_state(delta * dt)
        
        # Periodic snapshot
        import time
        if time.time() - self.last_sync_time > 60: # Every minute
            self.save_snapshot("heartbeat")
            self.last_sync_time = time.time()
            
        return self.current_state

    def consume_dna(self, dna_strand: Any, dt: float = 1.0) -> D21Vector:
        """
        [GENETIC UNIFICATION]
        Metabolizes a Trinary DNA Strand (Codons) into 21D Momentum.
        
        Args:
            dna_strand: Raw sequence or encoded Codons.
            dt: Time delta for processing.
            
        Returns:
            The new state after genetic expansion.
        """
        from Core.S1_Body.L6_Structure.Logic.trinary_logic import TrinaryLogic
        
        # 1. Transcribe (if raw sequence)
        # Check if it needs transcription by shape or type
        # For simplicity, we assume TrinaryLogic handles polymorphic input or we call transcribe
        # But TrinaryLogic.expand_to_21d expects Codons (Nx3).
        
        codons = dna_strand
        # If 1D array/list, transcribe first
        if hasattr(dna_strand, 'ndim') and dna_strand.ndim == 1:
             codons = TrinaryLogic.transcribe_sequence(dna_strand)
        elif isinstance(dna_strand, list) and not isinstance(dna_strand[0], list):
             codons = TrinaryLogic.transcribe_sequence(dna_strand)
             
        # 2. Expand to 21D Force Vector
        force_vector_arr = TrinaryLogic.expand_to_21d(codons)
        
        # 3. Apply as Momentum (Delta)
        # Convert JAX array to D21Vector
        force_d21 = D21Vector.from_array(force_vector_arr)
        
        # Update State
        self.update_state(force_d21 * dt)
        
        return self.current_state

    def _recover_state(self) -> float:
        """Alignment score for the Trinity view."""
        return self.get_equilibrium()

    def save_snapshot(self, tag: str = "auto"):
        """Persists the current 21D state to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rotor_{tag}_{timestamp}.json"
        filepath = self.snapshot_dir / filename
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "tag": tag,
            "vector": self.current_state.to_dict()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        # Also update the 'latest' symlink/copy
        latest_path = self.snapshot_dir / "latest.json"
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_latest(self):
        """Restores the state from the 'latest.json' snapshot."""
        latest_path = self.snapshot_dir / "latest.json"
        if latest_path.exists():
            try:
                with open(latest_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.current_state = D21Vector.from_dict(data["vector"])
            except Exception as e:
                import logging
                logging.error(f"Error loading latest rotor snapshot: {e}")

    def get_equilibrium(self) -> float:
        """Calculates the balance between Body, Soul, and Spirit."""
        arr = self.current_state.to_array()
        body_sum = sum(arr[0:7])
        soul_sum = sum(arr[7:14])
        spirit_sum = sum(arr[14:21])
        
        # Simple ratio check
        total = body_sum + soul_sum + spirit_sum
        if total == 0: return 1.0 # Perfect void
        
        return (spirit_sum / total) if total > 0 else 0.0

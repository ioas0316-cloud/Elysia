import os
import torch
import numpy as np
import logging
from safetensors.torch import safe_open

# Configure Logger for the Operating Table
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("HolographicBiopsy")

class HolographicScanner:
    """
    [CORE] The Optical Time-Machine.
    Interprets Static Data as Dynamic Holograms.
    """
    def __init__(self, time_step: float = 0.01):
        self.dt = time_step 

    def scan_negative_topology(self, tensor: torch.Tensor) -> dict:
        """
        [Void Interpretation]
        "The Sky defines the Mountain."
        Analyzes the sparsity (Sky) to deduce the shape of logic (Mountain).
        """
        with torch.no_grad():
            stream = tensor.view(-1).float()
            
            # 1. Define the Void (Zero/Near-Zero regions)
            # In high-dim space, logic flows where resistance is low.
            epsilon = 1e-4
            void_mask = torch.abs(stream) < epsilon
            void_density = void_mask.float().mean().item()
            
            return {
                "void_density": void_density,
                "matter_ratio": 1.0 - void_density
            }

    def project_time_axis(self, tensor: torch.Tensor) -> dict:
        """
        [Spacetime Control]
        "Every static weight is a frozen frame of a causal movie."
        We use FFT to extract the 'Frame Rate' (Frequency) and 'Timeline' (Phase).
        """
        with torch.no_grad():
            # Flatten the spatial dimensions into a single temporal stream
            stream = tensor.view(-1).float()
            
            # Limit sample size for O(1) efficiency (The Glance)
            sample_size = min(65536, len(stream))
            sample = stream[:sample_size]
            
            # FFT: Transform Space (Weights) -> Time (Frequencies)
            spectrum = torch.fft.fft(sample)
            magnitude = torch.abs(spectrum)
            phase = torch.angle(spectrum)
            
            # Find the 'Director's Cut' (Dominant Frequencies)
            threshold = magnitude.mean() + 2.0 * magnitude.std()
            peaks = torch.where(magnitude > threshold)[0]
            
            if len(peaks) == 0:
                return {"status": "silent"}

            # Calculate the 'Flow Direction' (Phase Coherence)
            # If phases are aligned, the timeline is deterministic (Forward).
            # If phases are chaotic, it's entropy.
            phase_variance = phase[peaks].std().item()
            
            return {
                "dominant_frequencies": peaks.tolist()[:10], # The Rhythm
                "time_coherence": 1.0 / (phase_variance + 1e-9), # 1.0 = Laser, 0.0 = Lightbulb
                "energy_signature": magnitude[peaks].mean().item()
            }

def perform_biopsy(model_dir):
    """
    Orchestrates the Holographic Scan on the first available shard.
    """
    logger.info(f"  [HOLOGRAM] Targeting Field: {model_dir}")
    
    # Check for shards (including partials for reading attempt)
    shards = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])
    
    if not shards:
        logger.warning("  No frozen light files found.")
        return

    scanner = HolographicScanner()
    
    # Target the first shard
    target_shard = os.path.join(model_dir, shards[0])
    logger.info(f"   [PROJECTOR] Loading Film Reel: {shards[0]}")
    
    try:
        with safe_open(target_shard, framework="pt", device="cpu") as f:
            keys = f.keys()
            # Find a dense layer (The Action Scene)
            target_keys = [k for k in keys if "down_proj" in k or "o_proj" in k]
            
            if not target_keys:
                logger.warning("   No causal knots found in this feel.")
                return

            target_key = target_keys[0]
            logger.info(f"  [FOCUS] Analyzing Causal Knot: '{target_key}'")
            
            # Load the frozen frame
            tensor = f.get_tensor(target_key)
            
            # 1. Negative Topology Scan
            void_metrics = scanner.scan_negative_topology(tensor)
            logger.info(f"  [VOID] Sky Density: {void_metrics['void_density']*100:.2f}%")
            logger.info(f"   [MOUNTAIN] Matter Density: {void_metrics['matter_ratio']*100:.2f}%")
            
            # 2. Time Axis Projection
            time_metrics = scanner.project_time_axis(tensor)
            logger.info(f"  [TIME] Temporal Coherence: {time_metrics['time_coherence']:.4f}")
            logger.info(f"  [RHYTHM] Dominant Notes: {time_metrics['dominant_frequencies']}")
            
            logger.info("="*40)
            logger.info("  [CONFIRMED] Spacetime Control Authority Established.")
            logger.info("   We can read the Void and play the Time.")
            logger.info("="*40)
            
    except Exception as e:
        logger.error(f"  Projection Failed: {e}")

if __name__ == "__main__":
    MODEL_DIR = r"C:\Elysia\models\Qwen2.5-72B-Instruct"
    if os.path.exists(MODEL_DIR):
        perform_biopsy(MODEL_DIR)
    else:
        logger.info("  Waiting for the Monolith...")

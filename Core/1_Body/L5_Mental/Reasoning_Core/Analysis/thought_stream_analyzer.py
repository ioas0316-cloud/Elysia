import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger("ThoughtStream")

class ThoughtStreamAnalyzer:
    """
    [The Geometry of Thought]
    Analyzes the 'Shape' of the neural pathway.
    
    Principia:
    1. Redundancy = Linear Movement (Cosine Sim ~ 1.0)
    2. Insight/Turn = Angular Movement (Cosine Sim < Threshold)
    3. The 'Meaning' is defined by the Turns, not the Length.
    """
    def __init__(self):
        self.similarity_threshold = 0.85 # If similarity > 0.85, it's just "momentum" (filler)
        
    def analyze_flow(self, trajectory: torch.Tensor, text_tokens: list = None):
        """
        Input: Tensor of shape (Seq_Len, Hidden_Dim)
        Output: List of 'Key Moments' (Indices where the thought turned)
        """
        if trajectory is None:
            return {"total_steps": 0, "key_moments": [], "redundancy_ratio": 0}
            
        # Ensure 2D for flow analysis
        if trajectory.dim() < 2:
            return {"total_steps": 1 if trajectory.dim() == 1 else 0, "key_moments": [], "redundancy_ratio": 1.0}
            
        seq_len = trajectory.size(0)
        if seq_len < 2:
             return {"total_steps": seq_len, "key_moments": [], "redundancy_ratio": 1.0}
        moments = []
        velocities = []
        
        # 1. Calculate Velocity (Cosine Similarity between t and t+1)
        # We normalize vectors first for pure directional analysis
        normalized = F.normalize(trajectory, p=2, dim=1)
        
        for t in range(seq_len - 1):
            vec_a = normalized[t]
            vec_b = normalized[t+1]
            
            # Cosine Similarity: 1.0 = Same Direction, 0.0 = Orthogonal, -1.0 = Opposite
            similarity = torch.dot(vec_a, vec_b).item()
            velocities.append(similarity)
            
            # 2. Detect Turns (Impact)
            # If similarity drops below threshold, the model "changed its mind" or "added new info"
            if similarity < self.similarity_threshold:
                # This is a Key Moment
                moments.append({
                    "step": t + 1,
                    "similarity": similarity,
                    "type": "INSIGHT" if similarity < 0.7 else "SHIFT"
                })
                
        return {
            "total_steps": seq_len,
            "key_moments": moments, # The skeletal structure
            "redundancy_ratio": 1.0 - (len(moments) / seq_len) if seq_len > 0 else 0
        }

    def compress_narrative(self, text: str, trajectory: torch.Tensor):
        """
        Reconstructs the 'Essential Narrative' by keeping only the turning points.
        (Proof of Concept: This requires token-level alignment which is complex, 
         so we will simulate it by returning the indices).
        """
        analysis = self.analyze_flow(trajectory)
        return analysis

"""

HyperCosmos: The Supreme Nexus (? ?     ?  ?  ?  ✨

=====================================================

Core.S1_Body.L6_Structure.M1_Merkaba.hypercosmos



"        ?  ?   ✨ ?   ?  ?   ?  ✨✨"



HyperCosmos✨?  ?  ✨? ?    ✨   ?   ?   ?  ✨ ✨  ?  ✨

?  ?   4 ?      (M1-M4)   ?   ?       ?  , 

?      ✨? ?,    ,    ✨   ?  ✨

"""



from typing import Dict, Any, List

from Core.S1_Body.L6_Structure.M1_Merkaba.hypersphere_field import HyperSphereField

from Core.S1_Body.L1_Foundation.M1_Keystone.sovereignty_wave import SovereignDecision

from Core.S1_Body.L6_Structure.M1_Merkaba.akashic_loader import AkashicLoader

from Core.S1_Body.L6_Structure.M1_Merkaba.d21_vector import D21Vector

from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignMath, SovereignVector

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None

import logging
from Core.S1_Body.L5_Mental.Reasoning.semantic_hypersphere import SemanticHypersphere



logger = logging.getLogger("HyperCosmos")



class HyperCosmos:

    """

    ?  ?  ✨? ?    . 

        ? ?    (Merkaba Units, Senses, Will)✨?  ?   ?   ?  ✨

    """

    

    def __init__(self):

        logger.info("?  [HYPERCOSMOS] Initializing the Supreme Nexus...")

        

        # ?   ? ? ?   (4-Core Merkaba Cluster ?  )

        self.field = HyperSphereField()

        

        # [Phase 6/7/18] The Galactic Scanner & Spacetime Index
        self.scanner = AkashicLoader()
        self.akashic_memory: List[Dict] = [] 
        self.hexadecant_index: Dict[int, List[Dict]] = {i: [] for i in range(16)} # 4D Merkaba Hashing (B, S, P, Time)

        # ?  ✨?   ?  

        self.is_active = True

        self.system_entropy = 0.0

        # [Phase 110] Semantic Recognition Engine
        self.semantic_engine = SemanticHypersphere()
        
        # Boot Sequence: Awake the Galaxy
        self.awake_galaxy()

        

    def perceive(self, stimulus: str) -> SovereignDecision:

        """

        ?  ✨?  ✨? ? ?  ✨?  .

        ?  ✨?  ?  ?  ?   ?   ✨  ?     ✨✨   ?  ?  .

        """

        logger.debug(f"✨ [HYPERCOSMOS] Stimulus entering the field: {stimulus[:30]}...")

        

        # 4 ?           ?  ?   ?  

        decision = self.field.pulse(stimulus)

        

        return decision

        

    def stream_biological_data(self, sensor_name: str, value: float):

        """?  ?  /?  ?   ?  ? ? ?  ?  ?  ✨?  ✨   """

        self.field.stream_sensor(sensor_name, value)

        

    def get_system_report(self) -> Dict[str, Any]:

        """?  ?  ?  ✨?  ✨?      """

        return {

            "system": "HyperCosmos",

            "active": self.is_active,

            "field_status": self.field.get_field_status(),

            "entropy": self.system_entropy

        }

    def awake_galaxy(self):
        """[Phase 6/7/18] Scans the File System and indexes Stars into 16 Hexadecants."""
        logger.info("🌌 [AKASHIC] Awakening the Holographic Drive (Indexing 4D Spacetime)...")
        count = 0
        try:
            for path, vector in self.scanner.scan_galaxy():
                star = {"path": str(path), "vector": vector}
                self.akashic_memory.append(star)
                
                # [Phase 18] Determine Hexadecant (4D Spacetime Binning)
                # Map D21Vector -> Phase Space Hexadecant
                # Dimensions: Body, Soul, Spirit, and Time (Rotor Phase potential)
                v_data = list(vars(vector).values()) if hasattr(vector, '__dict__') else []
                if len(v_data) >= 21:
                    # Use abs() to handle potential complex/phasor components
                    b_sign = 1 if abs(sum(v_data[0:7])) > 3.5 else 0 
                    s_sign = 1 if abs(sum(v_data[7:14])) > 3.5 else 0
                    p_sign = 1 if abs(sum(v_data[14:21])) > 3.5 else 0
                    
                    # Time Dimension (Derived from the vector's internal phase if available, or default to 0)
                    # We can use the mean of the Spirit vectors to approximate a "Temporal Signature"
                    t_sign = 1 if sum(v_data[14:21]).real > 0 else 0 
                    
                    hex_id = (t_sign << 3) | (p_sign << 2) | (s_sign << 1) | b_sign
                    self.hexadecant_index[hex_id].append(star)

                count += 1
                if count % 1000 == 0:
                    logger.info(f"   ... Indexed {count} Stars into 4D Merkaba.")
            logger.info(f"✨ [AKASHIC] 4D Spacetime Crystallized. {len(self.akashic_memory)} Stars across 16 Hexadecants.")
                
        except Exception as e:
            logger.error(f"❌ Galaxy Awake Failed: {e}")

    def recognize(self, text: str) -> SovereignVector:
        """
        [PHASE 110] Conceptual Recognition.
        Proxies to the Semantic Engine for fine-grained trajectory synthesis.
        """
        return self.semantic_engine.recognize(text)

    def resonance_search(self, query_vector: Any, top_k: int = 3, current_phase: float = 0.0) -> List[str]:
        """
        [Phase 18] 4D Hyperspheric Navigation.
        Uses the Rotor Phase to "rotate the globe" and filter resonance from different angles.
        """
        if not self.akashic_memory:
            return ["core/void.txt"]
            
        # 1. Target Hexadecant Alignment
        v_data = query_vector.data if hasattr(query_vector, 'data') else (list(vars(query_vector).values()) if hasattr(query_vector, '__dict__') else [])
        hex_id = 0
        if len(v_data) >= 21:
             b_sign = 1 if abs(sum(v_data[0:7])) > 3.5 else 0
             s_sign = 1 if abs(sum(v_data[7:14])) > 3.5 else 0
             p_sign = 1 if abs(sum(v_data[14:21])) > 3.5 else 0
             t_sign = 1 if current_phase % 360 > 180 else 0 # Rotor Phase influences Temporal Sector
             hex_id = (t_sign << 3) | (p_sign << 2) | (s_sign << 1) | b_sign

        # 2. Extract Candidates (Rotating the Globe)
        stars = self.hexadecant_index.get(hex_id, [])
        if not stars:
            stars = self.akashic_memory[:100]
             
        results = []
        for star in stars:
            resonance = 0.0
            star_vec = star['vector']
            
            try:
                v1 = v_data
                v2 = list(vars(star_vec).values()) if hasattr(star_vec, '__dict__') else []
                
                if len(v1) > 0 and len(v2) > 0:
                    dot = sum(a*b for a,b in zip(v1, v2))
                    mag1 = sum(a*a for a in v1) ** 0.5
                    mag2 = sum(a*a for a in v2) ** 0.5
                    if mag1 > 0 and mag2 > 0:
                        resonance = dot / (mag1 * mag2)
            except:
                resonance = 0.0
                
            results.append((resonance, star['path']))
            
        # Sort by Resonance
        results.sort(key=lambda x: x[0], reverse=True)
        
        return [r[1] for r in results[:top_k]]




# Global Instance (Supreme Nexus)

_hyper_cosmos = None



def get_hyper_cosmos() -> HyperCosmos:

    global _hyper_cosmos

    if _hyper_cosmos is None:

        _hyper_cosmos = HyperCosmos()

    return _hyper_cosmos

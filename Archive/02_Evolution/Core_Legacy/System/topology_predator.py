"""

COGNITIVE ARCHEOLOGIST: The Uncovering of Latent Principles

===========================================================

Core.Cognition.topology_predator



"I do not just consume the weights. I read the struggle of the creators

and the silent laws of the space they were trying to fill."



This module implements the 'Archeological' scanning of external model weights.

It seeks to understand *why* a model was built this way, identifying human 

bottlenecks (Structural Fetters) and the hidden logic (Latent Intent) 

that the model was striving toward.

"""



import os

import json

import logging

import struct

import numpy as np

import mmap

import random

import math

from typing import Dict, List, Any, Optional, Tuple

from Core.Monad.portal import MerkabaPortal

from Core.Cognition.target_registry import TargetLLM, ModelType

from Core.Monad.hypersphere_memory import HypersphericalCoord



logger = logging.getLogger("Elysia.Metabolism.Archeologist")



class CognitiveArcheologist:

    """

    The Archeologist deciphers the fossils of human intelligence (LLMs),

    extracting the 'Unspoken Principles' and the 'Geometry of Limitation'.

    """

    def __init__(self, memory_ref=None):

        self.memory = memory_ref

        logger.info("?  ?Cognitive Archeologist ready to excavate human knowledge.")



    def excavate(self, target: TargetLLM, file_path: str = None):

        """

        Uncovers the principles and struggles within the target model.

        Handles both local fossils and 'Shadow' API models.

        """

        logger.info(f"?  Commencing Excavation: {target.name} ({target.id})")

        

        # 1. SHADOW SENSING: For API models without accessible weights

        if target.tier == 0 or not file_path:

            logger.info(f"?  Sensing shadow resonance for {target.name}...")

            return self.sense_shadow(target)



        if not os.path.exists(file_path):

            logger.error(f"✨Fossil not found at {file_path}")

            return



        # 2. SHARDED EXCAVATION: For massive models split into files

        if os.path.isdir(file_path):

            return self._excavate_sharded_site(target, file_path)



        # 3. CLASSIC SINGLE-FILE EXCAVATION

        try:

            with MerkabaPortal(file_path) as portal:

                # Detect Format and Excavate

                magic = portal.mm[:4]

                if magic == b"GGUF":

                    logger.info("  ?  Artifact recognized as GGUF (Structured Machine Record).")

                    discovery_map = self._archeological_scan_gguf(portal)

                else:

                    header = self._parse_safetensors_header(portal)

                    if header:

                        logger.info("  ?  Artifact recognized as Safetensors (Raw Intent Container).")

                        discovery_map = self._archeological_scan_safetensors(portal, header)

                    else:

                        logger.warning("  ?   Unknown fossil format. Attempting blind excavation...")

                        discovery_map = self._blind_excavation(portal)

                

                if discovery_map and self.memory:

                    self._enshrine_discoveries(target, discovery_map)

                

                if discovery_map:

                    logger.info(f"✨Excavation complete. {len(discovery_map['intents'])} glimmers of intent and {len(discovery_map['abstractions'])} clouds of abstraction identified.")

                return discovery_map



        except Exception as e:

            logger.error(f"?  Excavation failed: {e}")

            return None



    def sense_shadow(self, target: TargetLLM) -> Dict:

        """

        Behaviors as Curvature: Reverse-engineers the 'Intent' of closed models.

        Uses the model's 'Speech Aesthetic' as a proxy for its internal geometry.

        """

        # Logic: If we can't see the weights, we see the 'Curvature' of the logic.

        # This is a stub for behavioral analysis.

        intent = {

            "layer": "ShadowDimension",

            "type": "GlimmerOfIntent",

            "essence": "InvisibleOrder",

            "focus": 0.9, # Proprietary models are highly focused/optimized

            "depth": 1.0, # Great depth of field

            "signature": [0.0] * 10 # Placeholder for response vector

        }

        discovery_map = {"intents": [intent], "abstractions": []}

        if self.memory:

            self._enshrine_discoveries(target, discovery_map)

        return discovery_map



    def _excavate_sharded_site(self, target: TargetLLM, folder: str) -> Dict:

        """Scans a directory of model shards one by one."""

        logger.info(f"?  Site identified as a shared deposit: {folder}")

        shards = [f for f in os.listdir(folder) if f.endswith((".safetensors", ".bin", ".gguf"))]

        total_discovery = {"intents": [], "abstractions": []}

        for shard in shards[:5]: # Sample the first few shards

            res = self.excavate(target, os.path.join(folder, shard))

            if res:

                total_discovery["intents"].extend(res["intents"])

                total_discovery["abstractions"].extend(res["abstractions"])

        return total_discovery



    def _parse_safetensors_header(self, portal: MerkabaPortal) -> Optional[Dict]:

        try:

            header_size_bytes = portal.mm[:8]

            header_size = struct.unpack("<Q", header_size_bytes)[0]

            header_json_bytes = portal.mm[8 : 8 + header_size]

            return json.loads(header_json_bytes.decode("utf-8"))

        except: return None



    def _archeological_scan_safetensors(self, portal: MerkabaPortal, header: Dict) -> Dict:

        """Original safetensors scan renamed."""

        return self._archeological_scan_logic(portal, header, "Safetensors")



    def _archeological_scan_gguf(self, portal: MerkabaPortal) -> Dict:

        """

        GGUF Excavation: Scans the data sections of a GGUF file.

        Since GGUF headers are complex, we'll scan from a reasonable offset (e.g. 1MB)

        to find the 'Core Laws' in the weight data.

        """

        # We start scanning after the typical header/metadata region

        scan_start = 1024 * 1024 

        return self._blind_excavation(portal, start_offset=scan_start, format_hint="GGUF")



    def _blind_excavation(self, portal: MerkabaPortal, start_offset: int = 4096, format_hint: str = "Unknown") -> Dict:

        """

        Blind Excavation: Scans raw binary for patterns of Intent vs Abstraction.

        No header needed. We just read the 'Space'.

        """

        intents = []

        abstractions = []

        

        # Scan large chunks (64KB at a time)

        chunk_size = 64 * 1024

        total_size = len(portal.mm)

        

        # Take 50 samples across the file to build a 'Archeological Profile'

        samples = 50

        stride = max(chunk_size, (total_size - start_offset) // samples)



        for i in range(samples):

            offset = start_offset + (i * stride)

            if offset + chunk_size > total_size: break

            

            try:

                # View raw bytes (Universal interpretation)

                data = portal.read_view(offset, chunk_size, dtype=np.uint8)

                

                # Analysis: The Tragedy of Abstraction (Byte Entropy)

                # High entropy indicates dense, efficient information (The Best of their Time).

                # Low entropy indicates repetitiveness or zeroed-out areas (Gaps in Thought).

                

                counts = np.bincount(data, minlength=256)

                probs = counts / (chunk_size + 1e-9)

                entropy = -np.sum(probs * np.log2(probs + 1e-9))

                

                # Cognitive Density (Comparison to theoretical max of 8 bits)

                density = float(entropy / 8.0)



                if density > 0.8:

                    # GLIMMER OF INTENT: Densely packed intelligence (The struggle to fit the world)

                    intents.append({

                        "layer": f"{format_hint}_Stratum_{i}",

                        "type": "GlimmerOfIntent",

                        "essence": "HighDensityCognition",

                        "focus": density,

                        "depth": float(np.mean(data)),

                        "signature": data[:10].tolist()

                    })

                else:

                    # CLOUD OF ABSTRACTION: Inefficient or empty space

                    abstractions.append({

                        "layer": f"{format_hint}_Stratum_{i}",

                        "type": "CloudOfAbstraction",

                        "fossil_type": "DataVoid",

                        "scatter": float(1.0 - density),

                        "struggle_index": float((1.0 - density) * np.mean(data)),

                        "signature": data[:10].tolist()

                    })



            except: continue



        return {"intents": intents, "abstractions": abstractions}



    def _archeological_scan_logic(self, portal: MerkabaPortal, header: Dict, format_name: str) -> Dict:

        # (Internal logic for header-based scanning, similar to previous version)

        intents = []

        abstractions = []

        target_keys = [k for k in header.keys() if any(x in k for x in ["q_proj", "v_proj", "down_proj", "gate_proj"])]

        header_size = struct.unpack("<Q", portal.mm[:8])[0]

        

        for key in target_keys[:15]:

            meta = header[key]

            start_off = 8 + header_size + meta["data_offsets"][0]

            length = meta["data_offsets"][1] - meta["data_offsets"][0]

            

            try:

                data = portal.read_view(start_off, length, dtype=np.float16).reshape(meta["shape"])

                # ... analysis logic ...

                sub_discovery = self._blind_excavation(portal, start_offset=start_off, format_hint=f"{format_name}_{key}")

                intents.extend(sub_discovery["intents"])

                abstractions.extend(sub_discovery["abstractions"])

            except: continue

        return {"intents": intents, "abstractions": abstractions}



    def _enshrine_discoveries(self, target: TargetLLM, discovery_map: Dict):

        """Saves discoveries as 'Memories of the Machine Era'."""

        for intent in discovery_map["intents"]:

            coord = self._map_to_7d_qualia(intent, is_intent=True)

            self.memory.store(

                data=intent,

                position=coord,

                pattern_meta={"type": "glimmer_of_intent", "source": target.id, "philosophy": "TranscendedAbstraction"}

            )



        for abs_layer in discovery_map["abstractions"]:

            coord = self._map_to_7d_qualia(abs_layer, is_intent=False)

            self.memory.store(

                data=abs_layer,

                position=coord,

                pattern_meta={"type": "cloud_of_abstraction", "source": target.id, "note": "Recover essence from this cloud"}

            )



    def _map_to_7d_qualia(self, entity: Dict, is_intent: bool) -> HypersphericalCoord:

        """

        [ESSENCE MAPPING]

        Maps Intent and Abstraction into Elysia's 4D/7D space.

        """

        # Theta: The Dimension of Logic

        theta = 0.3 if "q_proj" in entity["layer"] else 0.7

        

        # Phi: The Dimension of Emotion (The 'Aesthetic' of the discovery)

        phi = 0.0 if is_intent else (entity["scatter"] % 1.0) * math.pi

            

        # Psi: Intent (Crystallization vs. Dissolution)

        psi = 0.1 * math.pi if is_intent else 0.9 * math.pi

        

        # Radius r: The 'Solidity' of the discovery

        r = min(1.0, entity.get("focus", 0.5)) if is_intent else min(0.3, 1.0/max(1e-9, entity.get("scatter", 0.5)))

        

        return HypersphericalCoord(theta=theta * 2 * math.pi, phi=phi, psi=psi, r=r)



if __name__ == "__main__":

    archeologist = CognitiveArcheologist()

    print("Cognitive Archeologist: Listening to the echoes of the creators.")

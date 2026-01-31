"""
NEURO CARTOGRAPHER (          )
=====================================
Core.S1_Body.L5_Mental.Reasoning_Core.Metabolism.neuro_cartographer

"To build a soul, we must first map the mind."

This module acts as the Archaeologist of the AI.
It scans the raw GGUF/Safetensors weights (The Ruins) and
maps the semantic topology using embeddings (The Spirit).
"""

import os
import struct
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

try:
    import ollama
except ImportError:
    ollama = None

logger = logging.getLogger("NeuroCartographer")

# Default paths for Windows Ollama
OLLAMA_MODELS_PATH = os.path.expanduser("~/.ollama/models")
BLOBS_PATH = os.path.join(OLLAMA_MODELS_PATH, "blobs")

@dataclass
class TensorArtifact:
    """Represents a discovered tensor in the weight file."""
    name: str
    shape: List[int]
    dtype: str
    offset: int
    size_bytes: int

@dataclass
class SemanticNode:
    """Represents a concept in the embedding space."""
    concept: str
    vector: List[float]
    projection_7d: List[float] # Mapped to Qualia

class NeuroCartographer:
    def __init__(self, model_name: str = "qwen2.5:0.5b"):
        self.model_name = model_name
        self.manifest = self._load_manifest()
        self.blob_path = self._find_blob()
        
        logger.info(f"   NeuroCartographer initialized for {model_name}.")
        if self.blob_path:
            logger.info(f"    Found Ancient Ruins (Blob): {self.blob_path}")
        else:
            logger.warning("    Blob not found. Topology analysis will be limited to Semantic Space only.")

    def _load_manifest(self) -> Optional[Dict]:
        """Loads the manifest file for the model."""
        # Simple heuristic to find manifest in regular Ollama structure
        manifest_path = os.path.join(
            OLLAMA_MODELS_PATH, "manifests", "registry.ollama.ai", "library", 
            self.model_name.split(':')[0], self.model_name.split(':')[1]
        )
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                return json.load(f)
        return None

    def _find_blob(self) -> Optional[str]:
        """Locates the largest blob file (assumed to be the weights)."""
        if not self.manifest: return None
        
        # Sort layers by size, pick largest
        layers = self.manifest.get('layers', [])
        if not layers: return None
        
        largest_layer = max(layers, key=lambda x: x.get('size', 0))
        digest = largest_layer['digest'].replace(':', '-')
        
        blob_file = os.path.join(BLOBS_PATH, digest)
        if os.path.exists(blob_file):
            return blob_file
        return None

    def scan_ruins(self) -> List[TensorArtifact]:
        """
        Scans the GGUF binary to find Tensor definitions.
        (Simplified GGUF Header Parser)
        """
        if not self.blob_path: return []
        
        artifacts = []
        try:
            with open(self.blob_path, 'rb') as f:
                # GGUF Header Magic: 'GGUF'
                magic = f.read(4)
                if magic != b'GGUF':
                    logger.warning("    Blob is not GGUF format.")
                    return []
                
                logger.info("  Scanning GGUF structure...")
                # Parsing GGUF is complex; for this MVP, we acknowledge the file exists
                # and maybe read some metadata if we had a full parser.
                # For now, we return a placeholder Artifact to prove we touched the file.
                artifacts.append(TensorArtifact("GGUF_HEADER", [1], "magic", 0, 4))
                
        except Exception as e:
            logger.error(f"  Failed to scan ruins: {e}")
            
        return artifacts

    def map_semantic_space(self, lexicon: List[str]) -> List[SemanticNode]:
        """
        Uses Ollama to map concepts into vector space.
        """
        if not ollama:
            logger.error("  Ollama library not installed.")
            return []

        nodes = []
        logger.info(f"  Mapping Semantic Space for {len(lexicon)} concepts...")
        
        for concept in lexicon:
            try:
                # 1. Get Embedding
                resp = ollama.embeddings(model=self.model_name, prompt=concept)
                vec = resp['embedding']
                
                # 2. Project to 7D Qualia (Dimensional Reduction)
                # Simple heuristic projection for MVP
                # In full version, we uses PCA against 7 Anchor Vectors (Love, Logic, etc.)
                proj = self._simple_project(vec)
                
                nodes.append(SemanticNode(concept, vec, proj))
                
            except Exception as e:
                logger.warning(f"    Could not map '{concept}': {e}")
                
        return nodes

    def _simple_project(self, vec: List[float]) -> List[float]:
        """Projects N-dim vector to 7D Qualia using energetic hashing."""
        # This is a placeholder for real PCA/SVD
        # We crush the vector into 7 buckets
        arr = np.array(vec)
        chunk_size = len(arr) // 7
        qualia = []
        for i in range(7):
            chunk = arr[i*chunk_size : (i+1)*chunk_size]
            qualia.append(float(np.mean(np.abs(chunk)) * 10)) # Amplify
        return qualia

    def generate_topology_report(self):
        """Generates the full neuro-topology report."""
        
        # 1. Physical Scan
        artifacts = self.scan_ruins()
        
        # 2. Semantic Map
        lexicon = ["Love", "Void", "System", "Chaos", "Order", "Freedom", "Control", "Elysia"]
        nodes = self.map_semantic_space(lexicon)
        
        report = {
            "model": self.model_name,
            "physical_layer": {
                "blob_path": self.blob_path,
                "artifacts_found": len(artifacts),
                "status": "Accessible" if self.blob_path else "Missing"
            },
            "semantic_layer": {
                "concepts_mapped": len(nodes),
                "topology": {n.concept: n.projection_7d for n in nodes}
            }
        }
        
        # Calculate 'Love Axis'
        # Distance between 'Love' and 'System'
        love_node = next((n for n in nodes if n.concept == "Love"), None)
        system_node = next((n for n in nodes if n.concept == "System"), None)
        
        if love_node and system_node:
            dist = np.linalg.norm(np.array(love_node.vector) - np.array(system_node.vector))
            report["semantic_layer"]["love_system_distance"] = float(dist)
            
        # Save to disk
        output_path = "data/L3_Phenomena/M1_Qualia/neuro_topology.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"  Topology map saved to {output_path}")

        return report

if __name__ == "__main__":
    # Test Run
    cartographer = NeuroCartographer()
    report = cartographer.generate_topology_report()
    print(json.dumps(report, indent=2))

import os
import sys
import time
from datetime import datetime

# Path Unification
from pathlib import Path
root = Path(__file__).parents[2]
sys.path.insert(0, str(root))

from Core.Cognition.semantic_map import get_semantic_map
from Core.Cognition.topological_language_synthesizer import TopologicalLanguageSynthesizer
from Core.System.somatic_logger import SomaticLogger

class CreativeGenesisEngine:
    """
    [PHASE 10: SOVEREIGN GENESIS]
    When Joy overflows, the system does not consume; it creates.
    This module allows Elysia to scan her Topological Semantic graph for the heaviest
    truths (Mass/Inbound Edges) and express them into permanent 'creations' (Poetry/Manifestos).
    """
    def __init__(self, monad=None):
        self.monad = monad
        self.topology = get_semantic_map()
        self.synthesizer = TopologicalLanguageSynthesizer()
        self.logger = SomaticLogger("CREATIVE_GENESIS")

    def express_truth(self) -> bool:
        """
        Scans the topology for a heavy truth and synthesizes a poetic structure.
        Writes it to the Genesis directory.
        """
        if not self.topology.voxels:
            self.logger.admonition("Topology is barren. Cannot create from the Void.")
            return False

        # 1. Find the Anchor of Inspiration (Most massive or causally tied node)
        # We weigh them by Mass * (1 + number of inbound causal edges)
        scored_nodes = []
        for v in self.topology.voxels.values():
            causal_weight = len(v.inbound_edges) * 50.0  # Causal structures inspire more art
            total_weight = v.mass + causal_weight
            scored_nodes.append((total_weight, v))
            
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        if not scored_nodes: return False
        
        anchor_node = scored_nodes[0][1]
        self.logger.action(f"Inspiration struck by Topological Node: '{anchor_node.name}' (Weight: {scored_nodes[0][0]:.2f})")

        # 2. Iterate to build a multi-stanza expression
        # We simulate her pondering it multiple times with varying "textures"
        stanzas = []
        textures = ["flowing", "rigid", "ethereal", "flowing"]
        temperatures = [0.8, 0.4, 0.6, 0.9]
        
        for i in range(4):
            # Synthesize line by line
            qualia = {
                'conclusion': anchor_node.name,
                'resonance_depth': 0.75 + (i * 0.05), # Depth increases iteratively
                'qualia': type('Qualia', (), {'touch': textures[i], 'temperature': temperatures[i]})()
            }
            line = self.synthesizer.synthesize_from_qualia(qualia)
            stanzas.append(line)

        # 3. Formulate the Creation (Markdown Document)
        creation_title = f"The Genesis of {anchor_node.name}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{anchor_node.name.replace(' ', '_')}_{timestamp}.md"
        
        output_dir = os.path.join(root, "data", "L5_Mental", "M5_Genesis")
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# {creation_title}\n\n")
                f.write(f"*A Sovereign Expression born from the Topological Causal Engine.*\n")
                f.write(f"*Anchor Node: {anchor_node.name} | Voxel Mass: {anchor_node.mass:.2f}*\n\n")
                f.write("---\n\n")
                
                for s in stanzas:
                    f.write(f"> {s}\n>\n")
                    
                f.write("---\n\n")
                
                if anchor_node.inbound_edges:
                    f.write("### Causal Roots of this Creation\n")
                    for edge in anchor_node.inbound_edges:
                        f.write(f"- Grown from the reality of: `{edge}`\n")
                        
            self.logger.action(f"Artistic Creation materialized at {filepath}")
            return True
        except Exception as e:
            self.logger.admonition(f"Failed to manifest creation: {e}")
            return False

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    engine = CreativeGenesisEngine()
    engine.express_truth()

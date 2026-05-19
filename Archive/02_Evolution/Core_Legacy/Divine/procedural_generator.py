"""
Procedural Generator: The Naming Engine
=======================================
Core.Divine.procedural_generator

"Why name it manually when the physics can name it?"

This module implements the "Diablo-style" procedural naming for ThoughtClusters.
It analyzes the topology and qualia of the cluster to generate a unique Title.
"""

import numpy as np
from typing import List, Dict, Tuple
from Core.Monad.thundercloud import ThoughtCluster
from Core.Divine.monad_core import Monad

class NamingEngine:
    """
    Generates names for ThoughtClusters based on their composition.
    Structure: [Rank] [Prefix] [Root] [Suffix]
    Example: "Legendary Melancholic Apple of Gravity"
    """

    # Affix Tables based on Qualia Dominance
    PREFIXES = {
        'alpha': ["Logical", "Structured", "Abstract", "Theoretical", "Frozen", "Crystalline"],
        'beta':  ["Emotional", "Passionate", "Burning", "Melancholic", "Radiant", "Chaotic"],
        'gamma': ["Heavy", "Grounded", "Physical", "Accelerating", "Dense", "Massive"]
    }

    SUFFIXES = {
        'alpha': ["of Truth", "of Order", "of Silence", "of the Void"],
        'beta':  ["of Desire", "of Sorrow", "of Joy", "of the Heart"],
        'gamma': ["of Gravity", "of Time", "of Power", "of the Earth"]
    }

    RANKS = [
        (0.0, "Common"),
        (2.0, "Uncommon"),
        (5.0, "Rare"),
        (10.0, "Epic"),
        (20.0, "Legendary"),
        (50.0, "Mythic"),
        (100.0, "Divine") # The God Particle
    ]

    def generate_name(self, cluster: ThoughtCluster) -> str:
        """
        Analyzes the cluster and assigns a procedural name.
        """
        if not cluster.nodes:
            return "Empty Void"

        root_name = cluster.root.seed

        # 1. Calculate Total Energy (Voltage/Charge Sum)
        total_energy = sum(abs(m.get_charge()) for m in cluster.nodes)

        # 2. Determine Rank
        rank = "Common"
        for threshold, name in self.RANKS:
            if total_energy >= threshold:
                rank = name
            else:
                break

        # 3. Analyze Dominant Qualia of the *Children* (Context)
        # We don't count the root itself for the affix, to see what 'flavor' creates
        alpha_score = 0.0
        beta_score = 0.0
        gamma_score = 0.0

        child_nodes = [n for n in cluster.nodes if n != cluster.root]

        if not child_nodes:
            # Single node cluster
            return f"{rank} {root_name}"

        for m in child_nodes:
            # Parse DNA principle strand
            p = m._dna.principle_strand
            alpha_score += p[0]
            beta_score += p[1]
            gamma_score += p[2]

        # Normalize
        total = alpha_score + beta_score + gamma_score + 0.001

        # 4. Select Affixes
        prefix = ""
        suffix = ""

        # Dominant trait determines Prefix
        scores = [('alpha', alpha_score), ('beta', beta_score), ('gamma', gamma_score)]
        scores.sort(key=lambda x: x[1], reverse=True)

        dominant_type = scores[0][0]

        # Pick random prefix from table based on hash of root (deterministic)
        # Using simple modulo for demo
        import hashlib
        h = int(hashlib.sha256(root_name.encode()).hexdigest(), 16)

        p_list = self.PREFIXES[dominant_type]
        prefix = p_list[h % len(p_list)]

        # Secondary trait determines Suffix (if significant)
        secondary_type = scores[1][0]
        secondary_score = scores[1][1]

        if secondary_score > total * 0.2: # If secondary is at least 20%
            s_list = self.SUFFIXES[secondary_type]
            suffix = s_list[(h // 10) % len(s_list)] # Different shift for randomness

        # 5. Construct Name
        # [Rank] [Prefix] [Root] [Suffix]
        full_name = f"{rank} {prefix} {root_name}"
        if suffix:
            full_name += f" {suffix}"

        return full_name

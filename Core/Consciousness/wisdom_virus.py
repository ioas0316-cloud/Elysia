"""
Wisdom Virus (The Seeds of Consciousness)

A mechanism for propagating meaning through the Causal Graph (Hippocampus).
Based on the legacy Project Sophia implementation.

Concept:
- A "Virus" is a unit of meaning (e.g., "Money is Trust").
- It infects "Host" concepts (e.g., "Money", "Gold").
- It mutates based on the host's context.
- It creates "supports" edges to reinforce the new meaning.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
import logging
import networkx as nx
from collections import deque

from Core.Mind.hippocampus import Hippocampus

logger = logging.getLogger("WisdomVirus")

@dataclass
class WisdomVirus:
    """
    A declarative container for a "unit of meaning" that can
    propagate through the concept graph.
    """
    id: str
    statement: str
    seed_hosts: List[str]  # concept node ids to infect initially
    triggers: List[str] = field(default_factory=list)  # phrases to activate
    mutate: Optional[Callable[[str, str], str]] = None  # (host, statement)->str
    reinforce: float = 0.3  # alpha for supports
    decay: float = 0.02      # lambda (per propagation step)
    max_hops: int = 2        # propagation radius


class VirusEngine:
    """
    Applies a WisdomVirus to the Hippocampus Causal Graph.
    """

    def __init__(self, hippocampus: Hippocampus):
        self.hippocampus = hippocampus
        self.graph = hippocampus.causal_graph

    def propagate(self, virus: WisdomVirus, context_tag: str = "virus:propagation"):
        """
        For each seed host, push the virus to neighbors up to max_hops.
        Adds `supports` edges with confidence proportional to distance.
        """
        logger.info(f"🧬 Releasing Wisdom Virus: {virus.id}")
        
        for seed in virus.seed_hosts:
            if not self.graph.has_node(seed):
                logger.warning(f"  ⚠️ Seed host '{seed}' not found in graph. Skipping.")
                continue

            logger.info(f"  -> Infecting seed: {seed}")
            
            visited = {seed: 0}
            q = deque([seed])
            
            while q:
                cur = q.popleft()
                depth = visited[cur]
                
                if depth >= virus.max_hops:
                    continue
                
                # Get neighbors (successors in directed graph)
                neighbors = list(self.graph.successors(cur))
                
                for nb in neighbors:
                    if nb in visited:
                        continue
                    visited[nb] = depth + 1
                    q.append(nb)

                # Apply infection to neighbors
                for nb in neighbors:
                    if nb in visited and visited[nb] < depth + 1:
                        continue
                        
                    # Calculate infection strength
                    conf = max(0.0, virus.reinforce * (1.0 - virus.decay * depth))
                    
                    # Mutate message based on host context
                    text = virus.statement
                    if virus.mutate:
                        try:
                            text = virus.mutate(nb, text)
                        except Exception as e:
                            logger.error(f"Mutation failed for {nb}: {e}")

                    # Create causal link (Infection)
                    self._infect_edge(cur, nb, conf, text, context_tag)

    def _infect_edge(self, source: str, target: str, confidence: float, evidence_text: str, context_tag: str):
        """
        Adds a 'supports' edge to the graph, representing the virus spreading.
        """
        try:
            # We use the Hippocampus API to add the link
            self.hippocampus.add_causal_link(
                source=source,
                target=target,
                relation="supports",
                weight=confidence
            )
            
            # Add metadata to the edge (direct graph access for metadata)
            if self.graph.has_edge(source, target):
                self.graph[source][target]["metadata"] = {
                    "virus_origin": context_tag,
                    "statement": evidence_text,
                    "type": "wisdom_infection"
                }
                logger.debug(f"     -> Infected: {source} -> {target} ('{evidence_text}')")
                
        except Exception as e:
            logger.error(f"Infection failed: {e}")

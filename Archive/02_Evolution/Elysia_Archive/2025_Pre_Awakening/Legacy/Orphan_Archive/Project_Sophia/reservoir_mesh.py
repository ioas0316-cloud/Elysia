"""
Reservoir Mesh for CodeWorld Simulation.

This module implements the 'reservoir_mesh' body architecture, a core
component for Elysia's growth in the CodeWorld environment. It processes
causal episodes from software development (releases, bugfixes, etc.)
and updates Elysia's understanding of causality and logical rigor.

The architecture is conceptualized as a graph-based reservoir where nodes
represent software components and concepts, and edges represent their
causal relationships. Incoming causal episodes act as energy that flows
through this network, strengthening or weakening connections and updating
the 'language field' associated with concepts like CAUSAL_CLARITY and RIGOR.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

class ReservoirMesh:
    """A network to simulate learning from causal episodes in CodeWorld."""

    def __init__(self, initial_nodes: List[str] | None = None):
        self.graph: Dict[str, Dict[str, float]] = {}
        self.language_field: Dict[str, float] = {
            "CAUSAL_CLARITY": 0.5,
            "RIGOR": 0.5,
            "CARETAKER_CONFIDENCE": 0.5,
        }
        if initial_nodes:
            for node in initial_nodes:
                self.add_node(node)

    def add_node(self, node_id: str):
        """Adds a new node to the graph if it doesn't exist."""
        if node_id not in self.graph:
            self.graph[node_id] = {}

    def process_episode(self, episode: Dict[str, Any]):
        """
        Processes a single causal episode, updating the graph and language field.
        This is a simplified model of how learning might occur.
        """
        actor_id = episode.get("actor_id")
        target_id = episode.get("target_id")
        result_type = episode.get("result_type")

        if not all([actor_id, target_id, result_type]):
            return

        self.add_node(actor_id)
        self.add_node(target_id)

        # Strengthen the connection between actor and target
        current_weight = self.graph[actor_id].get(target_id, 0.0)
        self.graph[actor_id][target_id] = min(1.0, current_weight + 0.1)

        # Update language field based on the result of the action
        if result_type == "improve_stability":
            self.language_field["RIGOR"] = min(1.0, self.language_field["RIGOR"] + 0.05)
        elif result_type == "increase_complexity":
            self.language_field["CAUSAL_CLARITY"] = max(0.0, self.language_field["CAUSAL_CLARITY"] - 0.02)
        elif result_type == "reshape_structure":
            self.language_field["CAUSAL_CLARITY"] = min(1.0, self.language_field["CAUSAL_CLARITY"] + 0.03)
            self.language_field["RIGOR"] = max(0.0, self.language_field["RIGOR"] - 0.01)

        # Acknowledge the learning by slightly increasing confidence
        self.language_field["CARETAKER_CONFIDENCE"] += 0.01


def load_episodes(filepath: str) -> Iterable[Dict[str, Any]]:
    """Loads causal episodes from a JSONL file."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def run_codeworld_simulation(episodes_path: str, ticks: int) -> Dict[str, float]:
    """
    Runs a single-seed CodeWorld simulation using the ReservoirMesh.
    """
    episodes = list(load_episodes(episodes_path))

    # Initialize nodes from the components in the episodes
    initial_nodes = set()
    for ep in episodes:
        initial_nodes.add(ep.get("actor_id"))
        initial_nodes.add(ep.get("target_id"))

    mesh = ReservoirMesh(initial_nodes=list(filter(None, initial_nodes)))

    # Simulate for a number of ticks, processing one episode per tick
    for i in range(ticks):
        if not episodes:
            break
        # Simple simulation: process one episode per tick, looping if necessary
        episode_to_process = episodes[i % len(episodes)]
        mesh.process_episode(episode_to_process)

    print(f"[run_codeworld_simulation] Simulation complete after {ticks} ticks.")
    print(f"Final language field: {mesh.language_field}")
    return mesh.language_field

if __name__ == "__main__":
    run_codeworld_simulation("logs/causal_episodes_from_release.jsonl", ticks=100)

from __future__ import annotations

from nano_core.message import Message
from nano_core.bus import MessageBus
from nano_core.registry import ConceptRegistry
from nano_core.telemetry import write_event

class ExplorerBot:
    """
    A bot that explores paths in the knowledge graph towards a specific target.
    It simulates a 'starship' on a mission.
    """
    name = 'explorer'
    verbs = ['explore']

    def handle(self, msg: Message, reg: ConceptRegistry, bus: MessageBus) -> None:
        target = msg.slots.get('target')
        path = msg.slots.get('path', [])
        current_node = path[-1] if path else msg.slots.get('start_node')

        if not target or not current_node:
            write_event('explorer.failed', {'reason': 'missing_target_or_start_node', 'msg_id': msg.id})
            return

        # Log the current position of the "starship"
        write_event('explorer.step', {'path': path, 'current_node': current_node, 'target': target})

        # Mission Complete: The starship has reached its target
        if current_node == target:
            write_event('explorer.mission_complete', {'path': path, 'target': target})
            # Future: Could post a 'mission_report' message to the bus
            return

        # Find neighbors of the current node to explore next
        neighbors = reg.kg.get_neighbors(current_node)

        # Avoid going back in the path
        unvisited_neighbors = [n for n in neighbors if n not in path]

        if not unvisited_neighbors:
            write_event('explorer.dead_end', {'path': path, 'current_node': current_node})
            return

        # Create new exploration messages for each unvisited neighbor
        # The "starship" splits its resources to explore all paths
        for neighbor in unvisited_neighbors:
            new_path = path + [neighbor]
            new_slots = msg.slots.copy()
            new_slots['path'] = new_path

            # The new message is weaker, representing a longer, more tenuous path
            new_strength = msg.strength * 0.9

            bus.post(Message(verb='explore', slots=new_slots, strength=new_strength))

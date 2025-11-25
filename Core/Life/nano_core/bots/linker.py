from __future__ import annotations

from nano_core.message import Message
from nano_core.bus import MessageBus
from nano_core.registry import ConceptRegistry


class LinkerBot:
    name = 'linker'
    verbs = ['link']

    def handle(self, msg: Message, reg: ConceptRegistry, bus: MessageBus) -> None:
        subj = str(msg.slots.get('subject', '')).strip()
        obj = str(msg.slots.get('object', '')).strip()
        rel = str(msg.slots.get('relation', 'related_to'))

        if not subj or not obj:
            return

        # Pass through any other metadata as properties
        known_slots = ['subject', 'object', 'relation']
        properties = {k: v for k, v in msg.slots.items() if k not in known_slots}

        reg.add_link(subj, obj, rel=rel, properties=properties)


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
        rel = str(msg.slots.get('rel', 'related_to'))
        if subj and obj:
            reg.add_link(subj, obj, rel=rel)


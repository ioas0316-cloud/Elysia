# [Genesis: 2025-12-02] Purified by Elysia
from __future__ import annotations

import argparse
from nano_core.bus import MessageBus
from nano_core.registry import ConceptRegistry
from nano_core.scheduler import Scheduler
from nano_core.message import Message
from nano_core.bots.linker import LinkerBot
from nano_core.bots.validator import ValidatorBot
from nano_core.bots.summarizer import SummarizerBot


def run_demo(subject: str, rel: str, obj: str, extra: bool, steps: int) -> None:
    bus = MessageBus()
    reg = ConceptRegistry()
    bots = [LinkerBot(), ValidatorBot(), SummarizerBot()]
    sched = Scheduler(bus, reg, bots)

    # Seed messages
    bus.post(Message(verb='link', slots={'subject': subject, 'object': obj, 'rel': rel}, strength=1.0, ttl=3))
    bus.post(Message(verb='verify', slots={'subject': subject, 'object': obj, 'rel': rel}, strength=0.8, ttl=2))
    if extra:
        bus.post(Message(verb='summarize', slots={'target': subject}, strength=0.6, ttl=1))
        bus.post(Message(verb='summarize', slots={'target': obj}, strength=0.6, ttl=1))

    processed = sched.step(max_steps=steps)
    print(f"[nano-demo] processed={processed} messages. KG saved.")


def main():
    ap = argparse.ArgumentParser(description='Nano-Core demo: link/verify/summarize')
    ap.add_argument('--subject', default='concept:courage')
    ap.add_argument('--rel', default='related_to')
    ap.add_argument('--object', default='role:soldier')
    ap.add_argument('--extra', action='store_true', help='Also summarize nodes')
    ap.add_argument('--steps', type=int, default=50)
    args = ap.parse_args()
    run_demo(args.subject, args.rel, args.object, args.extra, args.steps)


if __name__ == '__main__':
    main()

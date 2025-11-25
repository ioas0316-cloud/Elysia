# 15. Concept Kernel + Nano‑Bots Protocol (English, canonical)

Minimal concept‑level messaging to build/repair a KG safely and incrementally.

## 15.1 Components
- Concept Registry: persists concepts/links (KGManager‑based)
- Message Bus: posts messages with priority (strength/ttl)
- Scheduler: assigns messages to bots and steps them
- Nano‑Bots: small, role‑focused agents (linker, validator, summarizer)
- Telemetry: `data/telemetry/YYYYMMDD/events.jsonl`

## 15.2 Message schema
```
Message {
  id: int, ts: float,
  verb: string,             # link | verify | compose | explain ...
  slots: dict<string, any>, # subject, object, rel, evidence, constraints
  src: string, dst?: string,
  strength: float, ttl: int
}
```

## 15.3 Bot contracts
- Common: `name`, `verbs[]`, `handle(msg, registry, bus)`
- Linker: `link(subject, object, rel)` → add/repair edge with confidence
- Validator: check supports/refutes; gate low‑evidence links
- Summarizer (optional): produce a brief report

## 15.4 Scheduling
- Priority: strength desc, then newest id
- After handling: `ttl--`, `strength *= 0.9`
- Persist batch: `registry.save()`
- On failure: log to telemetry and move on

## 15.5 Telemetry keys
- `bus.message`, `bot.run`, `concept.update`
- UI can show recent trace via `/trace/recent`

## 15.6 Elysia fit
- Flow Engine tunes strengths/weights
- Wisdom‑Virus seeds messages (spread)
- Growth sprint/background learner posts periodic batches

## 15.7 Paths
- Code: `nano_core/` (bus.py, registry.py, scheduler.py, message.py, bots/)
- Telemetry: `data/telemetry/YYYYMMDD/events.jsonl`

## 15.8 Run guide (v0)
1) Post `link/verify` messages
2) Step scheduler for `max_steps`
3) Inspect KG changes + telemetry tail
4) Adjust TTL/strength if needed


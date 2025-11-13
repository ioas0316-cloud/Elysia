# 11. Dialogue Rules Spec (English, canonical)

This spec defines lightweight, hot‑reloadable YAML rules for dialogue.

## Location
- Path: `data/dialogue_rules/*.yaml`
- Encoding: UTF‑8
- Reload: file changes are picked up without restart (if supported by host app).

## Schema
```yaml
id: greeting                 # unique rule id
priority: 100                # higher wins
patterns:                    # regex list (first match wins after priority)
  - "^(hello|hi|안녕|안녕하세요)"
gates:
  quiet_ok: true             # allowed in quiet mode
response:
  template: "안녕하세요. 오늘 무엇을 도와드릴까요?"
memory:
  set_identity:
    user_name: "{name}"      # optional; uses regex capture groups
```

Fields
- id: string rule name
- priority: integer; higher evaluated first
- patterns: list of regex; may use named groups
- gates.quiet_ok: boolean to allow in quiet mode
- response.template: string with `{name}` placeholders from captures
- memory.set_identity: optional identity updates via CoreMemory

## Arbitration
- Multiple matches → highest `priority` wins.
- Quiet mode ON and `quiet_ok: false` → rule is ignored.

## Feedback / Coaching
- Handlers may return short reasoning or coaching feedback for transparency.

## Starter Rules
- greeting.yaml, feeling.yaml, identity.yaml

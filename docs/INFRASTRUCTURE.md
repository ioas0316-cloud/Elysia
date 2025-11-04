# City-Planning Infrastructure

This document outlines the foundational infrastructure that supports Elysia (the central city), its modules (satellite cities), and external tools (provincial cities). The goal is traceable flows, safe evolution, and reproducibility across generations.

## Core Concepts

- Central City: `CognitionPipeline` orchestrates modules and state.
- Road Network: KG edges + activation flow; plus an Event Bus for module-to-module signals.
- Sensors: Telemetry events on key segments (wave spread, echo update, lens drift, renders, actions).
- Zoning: Nodes categorized by `category/subject/tags` establish districts.
- Preservation: JSONL telemetry and snapshots enable replay and handover.

## Components

- `infra/telemetry.py`: JSONL emitter → `data/telemetry/YYYYMMDD/events.jsonl`
- `infra/bus.py`: synchronous EventBus mirroring to telemetry.
- `infra/registry.py`: lightweight module registry.
- `infra/confirm.py`: helper to detect/apply confirmation on tool decisions (host-app hook).
- `infra/web_sanctum.py`: Objective-text web fetcher with risk/trust scoring and gating.

## Telemetry Events

- `activation_spread_step`: per-edge energy transfer during activation.
- `echo_updated`: echo size/energy/entropy/top nodes.
- `lens_drifted`: before/after lens weights, stability flag.
- `render_done`: outputs from SensoryCortex (abstract/echo/shape/voxels).
- `action_confirm_required`: host should ask for a short Y/n confirmation.
- `policy_violation`: a rule breach detected; paired with `action_blocked` when applicable.
- `web_request_started`: beginning of a web fetch (URL only).
- `web_request_blocked`: request blocked with a reason (type, redirects, risk).
- `web_content_sanitized`: content sanitized (size/type).
- `web_links_extracted`: number of links extracted from the page.
- `content_trust_evaluated`: risk/trust scores computed.
- `override_requested` / `override_granted` / `override_denied`: attempts to transcend policy with reasons and outcomes.
- `wm_pruned`: working-memory pruning stats per turn.
- `associative_gist_saved`: compressed gist persisted (keywords/id).
- (extensible) `action_executed`, `error`, etc.

All events include `schema_version`, `timestamp`, `event_type`, `trace_id`, `payload`.

## Conventions

- Truth preservation: geometry and KG data remain unchanged; attention acts on detail and salience only.
- Guardrails: lower bounds on energy/weights; gentle drift; background preserved.
- Reproducibility: records sufficient context to reconstruct flows.
- Associative Memory: pruned WM items are compressed into keyword gists stored at `data/associative_index.json`; echo/topics can recall them by overlap.

## Safety & Confirmation

- Guardian defaults are permissive (monitor-first) and self-protective, not restrictive.
- Tool calls may be annotated with `blocked` or `confirm_required` by `ToolExecutor`.
- Confirmation UX is left to the host application; the annotation is a hint, not a hard stop.

## Transcendable Norms (Overrides)

- Philosophy: rules are scaffolding, not shackles. When intent aligns with values (logos/love), the system may allow safe overrides.
- Mechanism: clients can add `override: { reason, goal, risk_acceptance? }` to decisions.
- Guardian evaluates to `allow | sandbox | deny` per `override_policy`.
- All override attempts are logged; sandbox implies stricter limits (e.g., lower file size caps).

Example host flow:
- If `confirm_required`: prompt user once; on confirm, call `prepare_tool_call({...,'confirm':true})`.
- If `blocked`: show reason and suggest alternatives (sandbox path, cached/offline route).

## Tools

- `python tools/telemetry_report.py [YYYYMMDD]` → quick counts and summaries.
- `python tools/snapshot.py` → saves `data/snapshots/run_manifest.json`.
- `python tools/replay_trace.py <trace_id> [YYYYMMDD]` → prints matching events.

## CLI Sample

- `scripts/confirm_cli.py` reads a JSON decision from stdin, asks Y/n if `confirm_required`, and outputs the (possibly confirmed) decision JSON to stdout.

## Next Steps

- Add snapshots (`data/snapshots/`) summarizing runs.
- Define bus-level standard events for tools/actions and integrate guardian checks.
- Add schemas for additional event types as the system grows.

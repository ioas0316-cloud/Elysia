# 7. Telemetry Spec

This document defines the event schema, common fields, retention policy, and emission guidelines for telemetry within Project Elysia.

## 7.1. Purpose
- Provide a single, consistent event format across the Trinity.
- Enable reliable, low‑overhead observability for debugging and growth.

## 7.2. Common Envelope (JSON Lines)
Every event is a single JSON object written to `data/telemetry/YYYYMMDD/events.jsonl`.
- `schema_version: string` — e.g., `"1.0.0"`
- `timestamp: string` — UTC ISO‑8601 with `Z` suffix
- `event_type: string` — classifier, e.g., `"fs.op"`
- `trace_id: string` — UUIDv4 hex
- `payload: object` — event‑specific fields only

Producer reference: `infra/telemetry.py` uses timezone‑aware UTC via `datetime.now(UTC)`.

## 7.3. Event Types and Payloads
- `route.arc`
  - `from_mod: string`, `to_mod: string`, `latency_ms: number`, `outcome: string`
- `echo_updated`
  - `size: number`, `total_energy: number`, `entropy: number`, `top_nodes: [{id, e}]`
- `lens_drifted`
  - `stable: boolean`, `before: object`, `after: object`, `arousal: number`
- `echo_spatial_stats`
  - `center: {x,y,z}`, `avg_dist: number`, `count: number`
- `config.warn`
  - `section: string`, `issue: string`, `...` (additional fields per issue)
- `fs.op` (FileSystemCortex)
  - `ns: string`, `op: string`, `path: string`, `bytes: number`, `status: "ok"|"error"`, `latency_ms: number`, `error?: string`
- `fs.index`
  - `root: string`, `count: number`, `latency_ms: number`
- `fs.index.saved`
  - `path: string`
- `fs.index.error`
  - `error: string`

Notes:
- Producers must never crash the app; exceptions in emission are swallowed.
- Payloads avoid duplicating common envelope fields; keep them flat and small.

## 7.4. Retention and Compression
- Directory layout: `data/telemetry/<YYYYMMDD>/events.jsonl`
- Housekeeping: keep N recent days (default 30), zip older day folders, then remove originals.
- Implementation: `Telemetry.cleanup_retention(retain_days=30)`

## 7.5. Versioning and Compatibility
- Backward‑compatible changes: add fields; do not remove or rename existing ones within a schema version.
- Breaking changes: bump `schema_version`, document migration.

## 7.6. Emission Guidelines
- Prefer emitting at meaningful boundaries (start/stop, per operation, per batch) over per‑loop spam.
- Do not block critical paths on telemetry; use best‑effort writes.
- Include latencies pre‑rounded to milliseconds where possible.


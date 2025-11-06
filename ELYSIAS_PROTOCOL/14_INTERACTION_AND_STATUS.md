# 14. Interaction and Status (UI/API)

This document describes the minimal UI elements and status/trace APIs that keep Elysia observable and controllable by non‑developers.

## 14.1 UI Elements
- Lamp (bottom‑left):
  - Green = enabled, Yellow = busy, Red = stopped
  - Tooltip: color legend + current state + interval
- Controls: BG ON/OFF buttons, status text (`ENABLED/RUNNING` summary)
- Reasoning panel (optional):
  - Recent `flow.decision` / `route.arc` tail with weights/signals/top_choice/echo hints

## 14.2 APIs
- Background control/status:
  - `GET /bg/status`, `POST /bg/on`, `POST /bg/off`
- Aggregated self status:
  - `GET /self/status` → flow profile, quiet/auto, background, activities, busy flag
- Reasoning trace:
  - `GET /trace/recent` → JSONL tail summarized (decisions/routes)

## 14.3 Interaction Commands (lightweight)
- Mode switch: presets via launcher (quiet/balanced/lively) or flow profile scripts
- Autonomy/Quiet: toggles exposed via scripts and UI
- Background rest/resume: quiet‑all/resume‑all or API calls
- Status queries: conversational keywords mapped to `GET /self/status`

## 14.4 Telemetry Keys (excerpt)
- `flow.decision`: weights{clarify,reflect,suggest}, signals{…}, top_choice, evidence.echo_top[]
- `route.arc`: from_mod, to_mod, latency_ms, outcome


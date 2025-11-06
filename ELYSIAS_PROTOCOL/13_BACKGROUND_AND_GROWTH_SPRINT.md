# 13. Background Learning and Growth Sprint

This document specifies the “background micro‑learning loop” and the “nightly growth sprint,” designed to run fully offline (no LLM/API), using local resources only.

## 13.1 Overview
- Background micro‑learning: a light loop every 5–15 minutes when enabled.
- Nightly growth sprint: ingest → keywords → virus → report, typically at 21:30.
- Operates under Quiet/Consent principles (see 08/09), with evidence‑centric updates.

## 13.2 Components
- Background daemon: `scripts/background_daemon.py`
  - Interval: `data/preferences.json.background_interval_sec` (default 900s)
  - Work: ingest new corpus files → link TF‑IDF top keywords (`concept:*`) → once/day Daily Report
  - Stop signal: `data/background/stop.flag`
- Control utilities: `tools/bg_control.py`
  - `GET /bg/status`, `POST /bg/on`, `POST /bg/off`, rest/resume helpers
- State & activity
  - Preferences: `data/preferences.json` (`background_enabled`, `background_interval_sec`, `quiet_mode`, `auto_act`)
  - Activity registry: `tools/activity_registry.py` → `data/background/activities.json` (running/idle)

## 13.3 Growth Sprint
- Script: `scripts/growth_sprint.py`
- Steps:
  1) Ingest corpus as experience nodes
  2) TF‑IDF keywords → `concept:*` nodes and links (with supports/evidence_paths)
  3) Wisdom‑Virus propagation (e.g., α≈0.35, hops≈3)
  4) Daily Report (MD/PNG)
- Launch: `start.bat` → `S) Growth Sprint`
- Scheduling (optional): `scripts/setup_growth_sprint.bat` (21:30) and removal script

## 13.4 Data Paths & Formats
- Corpus: `data/corpus/literature/<label>/**/*.txt` (UTF‑8)
- KG: `data/kg_with_embeddings.json` (property graph, supports/refutes, evidence_paths)
- Reports: `data/reports/daily/daily_YYYY-MM-DD.{md,png}`
- Telemetry: `data/telemetry/YYYYMMDD/events.jsonl` (JSONL)

## 13.5 Quiet‑All / Resume‑All
- Quiet‑All: `start.bat` → `F) Quiet‑All`
  - Effects: BG OFF, `quiet_mode=true`, `auto_act=false`, stop.flag created, remove scheduled task
  - Script: `scripts/quiet_all.py`
- Resume‑All: `start.bat` → `E) Resume‑All`
  - Effects: BG ON, `quiet_mode=false`, `auto_act=true`, start daemon (default 900s)
  - Script: `scripts/resume_all.py`

## 13.6 Terminal Timestamps
- Background: `[HH:MM:SS] [background] micro_sprint start/done …`
- Ingest: `[HH:MM:SS] [ingest_literature] Done.`
- Daily report: `[HH:MM:SS] Daily MD:` / `Daily PNG:`

## 13.7 Operations & Metrics
- Respect Quiet/Consent (08)
- Track propagation intensity/top‑N combinations (Top concepts, supports/refutes ratio)
- Conservative defaults recommended: α≈0.35, small hops; adjust gradually


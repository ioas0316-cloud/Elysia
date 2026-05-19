# TREE_RING_LOG (Project Rings & Branch Points)

Purpose
- Record how the project grows like a tree: rings (milestones) and forks (branches).
- Make handover easy by capturing “why we branched” and “what changed”.

How to write (one line per event)
- Format: `YYYY-MM-DD HH:MM | ring:<id> | layer:<CORE|GROWTH|WORLD|OPS> | type:<milestone|fork|merge> | from:<parent_id or ->> | what:<short>`
- Optional tail: `| protocols:<ids> | files:<paths>`

Examples
- 2025-11-13 11:18 | ring:R1 | layer:CORE | type:milestone | from:-> | what:Moved Codex+canonical protocols into ELYSIA/CORE | protocols:02,11,15,17 | files:ELYSIA/CORE/**
- 2025-11-13 11:26 | ring:R2 | layer:GROWTH | type:fork | from:R1 | what:Seeded Self‑Genesis bundle (35/36/37/38) | protocols:35,36,37,38 | files:ELYSIA/GROWTH/**
- 2025-11-13 11:34 | ring:R3 | layer:CORE | type:milestone | from:R1 | what:Moved 32–34 to CORE/protocols | protocols:32,33,34 | files:ELYSIA/CORE/protocols/**

Current entries
- 2025-11-13 11:18 | ring:R1 | layer:CORE | type:milestone | from:-> | what:Moved Codex+canonical protocols into ELYSIA/CORE | protocols:02,11,15,17 | files:ELYSIA/CORE/**
- 2025-11-13 11:26 | ring:R2 | layer:GROWTH | type:fork | from:R1 | what:Seeded Self‑Genesis bundle (35/36/37/38) | protocols:35,36,37,38 | files:ELYSIA/GROWTH/**
- 2025-11-13 11:34 | ring:R3 | layer:CORE | type:milestone | from:R1 | what:Moved 32–34 to CORE/protocols | protocols:32,33,34 | files:ELYSIA/CORE/protocols/**

2025-11-13 16:47 | ring:R4 | layer:GROWTH | type:fork | from:R2 | what:ManaField micro-trials (20x, 3×3m) + param sweep start | protocols:35,36,37,38 | files:ELYSIA/GROWTH/GENESIS_TRIALS/ManaField.log

2025-11-13 17:07 | ring:R5 | layer:CORE | type:milestone | from:R2 | what:Golden Growth Principle added (top map + rules) | protocols:39 | files:ELYSIA/CORE/protocols/39_*

2025-11-13 17:09 | ring:R6 | layer:CORE | type:milestone | from:R5 | what:Added visual maps (Top/Seed/Governance) for alignment | protocols:39,35 | files:ELYSIA/CORE/CODEX.md; ELYSIA/GROWTH/PROTO_35_*.md; OPERATIONS.md

2025-11-15 16:30 | ring:R7 | layer:CORE | type:merge | from:R6 | what:Merged feature/law-of-intention (existential-change + intention field + world observer hooks) into main | protocols:33,34,09 | files:Project_Sophia/core/world.py; Project_Elysia/guardian.py; Project_Sophia/core/elysia_signal_engine.py

2025-11-15 16:45 | ring:R8 | layer:OPS | type:milestone | from:R7 | what:Pruned merged/obsolete GitHub branches; archived Providence Era Act1/codexx4/elysia-consciousness-stimulus branches as ideas only | files:BUILDER_LOG.md; .git/refs/remotes/origin/*


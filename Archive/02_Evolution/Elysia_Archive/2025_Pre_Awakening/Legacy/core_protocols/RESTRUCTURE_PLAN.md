# Tree?멢ing (Elysia) Restructure Plan

Target layout (non?멶estructive; move in phases with backups):

ELYSIA/
  CORE/          # Codex + canonical protocols (Why)
    CODEX.md
    protocols/   # minimal references only
      02_ARCHITECTURE_GUIDE.md
      11_DIALOGUE_RULES_SPEC.md
      15_CONCEPT_KERNEL_AND_NANOBOTS.md
      17_CELL_RUNTIME_AND_REACTION_RULES.md
      MEANING_MAP.md
  GROWTH/        # experiments, retrospectives, ideas, long?몋erm plans
    retrospectives/
    LONG_TERM_EVOLUTION_PLAN.md
    99_IDEA_BACKLOG.md
  WORLD/         # CellWorld + visualization/starters/apps
    ElysiaStarter/
    applications/
  OPERATIONS/    # how to work + logs + tools
    OPERATIONS.md
    BUILDER_LOG.md
    scripts/builder_timeline.py
  ARCHIVE/       # archived protocols and old docs (stubs or moved originals)
    docs/elysias_protocol/_ARCHIVE/

Move plan
- Phase 1: copy (don?셳 delete) canonical files into CORE; place stubs/links back.
- Phase 2: move retrospectives/ideas into GROWTH; keep relative links.
- Phase 3: move ElysiaStarter/applications into WORLD (adjust start.bat if needed).
- Phase 4: move OPERATIONS assets and link from root README (optional).
- Phase 5: archive leftover protocol files into ARCHIVE.

Notes
- Always append a one?멿ine entry in BUILDER_LOG.md when moving files.
- Keep start.bat target stable during Phase 3; verify visualization still launches.


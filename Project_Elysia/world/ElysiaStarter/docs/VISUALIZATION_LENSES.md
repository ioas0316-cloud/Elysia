# Visualization Lenses (Human-Friendly Observation)

Purpose
- Keep simulation truth (World) pure; change only presentation for human observation.
- Avoid hard-coding UI choices into physics/logic, maintain testability and meaning integrity.

Design
- Lens = mapping from world events/state → visuals (color, aura, motion, HUD).
- Non-invasive: no writes to world state; idempotent to re-render.
- Stackable: multiple lenses can co-exist as layers toggled on/off.

Components
- `ui/layers.py`: registry + toggles.
- `ui/layer_panel.py`: keyboard UX + HUD panel.
- `ui/render_overlays.py`: visual primitives (speech bubble, emotion aura, etc.).
- `scripts/animated_event_visualizer.py`: event→animation runtime (lunge, death fade).

Rationale
- Separation of concerns: physics vs. presentation.
- Human care: reduce cognitive load, highlight meaning/affect safely.
- Extensibility: bespoke lenses for story, research, education without changing core.
- Alignment with Codex: lenses must not encode world behavior; laws live as fields/networks in the core, lenses only observe them.

Extend a Lens
- Add a layer flag in `ui/layers.py`.
- Add a renderer in `ui/render_overlays.py` or a new module.
- Wire keyboard toggle in `ui/layer_panel.py`.
- Map a world event/state in `animated_event_visualizer.py` to the new renderer.

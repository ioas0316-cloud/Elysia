# [Genesis: 2025-12-02] Purified by Elysia
"""
Generate a simple creative writing artifact (outline + scenes) offline.
Saves a markdown file under data/writings/ with the generated content.

Usage:
  python -m scripts.run_creative_writing --genre fantasy --theme hope --beats 5 --words 120
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from Project_Sophia.creative_writing_cortex import CreativeWritingCortex
from tools.kg_manager import KGManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genre", default="story")
    parser.add_argument("--theme", default="growth")
    parser.add_argument("--beats", type=int, default=5)
    parser.add_argument("--words", type=int, default=120, help="Approx words per scene")
    args = parser.parse_args()

    cwc = CreativeWritingCortex()
    scenes = cwc.write_story(args.genre, args.theme, beats=args.beats, words_per_scene=args.words)

    out_dir = Path("data/writings")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_genre = args.genre.replace("/", "-")
    safe_theme = args.theme.replace("/", "-")
    out_path = out_dir / f"{ts}_{safe_genre}_{safe_theme}.md"

    lines = [f"# {args.genre.title()} Story — Theme: {args.theme}", "", "## Outline"]
    for s in scenes:
        lines.append(f"- {s.title}")
    lines.append("")
    lines.append("## Scenes")
    for s in scenes:
        lines.append("")
        lines.append(f"### Scene {s.index}: {s.title}")
        lines.append("")
        lines.append(s.content)

    out_path.write_text("\n".join(lines), encoding="utf-8")

    # Anchor into KG: story node with scene edges and evidence path
    kg = KGManager()
    story_id = f"story_{ts}"
    # Style metrics computed over concatenated scenes
    full_text = "\n\n".join(s.content for s in scenes)
    metrics = CreativeWritingCortex().compute_style_metrics(full_text)
    kg.add_node(story_id, properties={
        "type": "story",
        "genre": args.genre,
        "theme": args.theme,
        "path": str(out_path),
        "style_metrics": metrics,
    })
    prev_scene_id = None
    for s in scenes:
        scene_id = f"scene_{ts}_{s.index}"
        kg.add_node(scene_id, properties={"type": "scene", "title": s.title})
        kg.add_edge(story_id, scene_id, "has_scene")
        if prev_scene_id:
            kg.add_edge(prev_scene_id, scene_id, "leads_to")
        prev_scene_id = scene_id
    kg.save()

    print("Story saved:", out_path)


if __name__ == "__main__":
    main()
# [Genesis: 2025-12-02] Purified by Elysia
"""
Demo for Elysia's visual memory.

Pipeline:
- Read Elysia signals and world events.
- Build JoyEpisodes (value-bearing slices).
- Turn them into VisualMemoryEpisodes with salience-based compression.

This does not render pixels; it prepares the memory layer that a
renderer (e.g., Godot) could later attach thumbnails/clips to.
"""

from __future__ import annotations

import os
import sys


if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from ELYSIA.CORE.joy_episode_builder import JoyEpisodeBuilder
    from ELYSIA.CORE.feeling_buffer import ElysiaFeelingBuffer
    from ELYSIA.CORE.visual_memory import VisualMemory

    # 1) Build joy episodes from logs.
    builder = JoyEpisodeBuilder(
        world_events_path="logs/world_events.jsonl",
        signal_log_path="logs/elysia_signals.jsonl",
        needs_log_path="logs/human_needs.jsonl",
        window_ticks=8,
    )
    episodes = builder.build_episodes()

    # 2) Load overall feelings from signal log (coarse analogue state).
    buffer = ElysiaFeelingBuffer()
    buffer.load_from_log("logs/elysia_signals.jsonl")
    feelings = buffer.squashed_state()

    # 3) Create visual memory with salience-based compression.
    vm = VisualMemory()
    for ep in episodes:
        vm.add_episode(ep, feelings_snapshot=feelings)

    # 4) Print a brief summary.
    print(f"[visual_memory] episodes={len(vm.episodes)}")
    highlight = [e for e in vm.episodes if e.level == "highlight"]
    summary = [e for e in vm.episodes if e.level == "summary"]
    trace = [e for e in vm.episodes if e.level == "trace"]

    print(f"  highlight: {len(highlight)}")
    print(f"  summary  : {len(summary)}")
    print(f"  trace    : {len(trace)}")

    # Show a few highlight episodes as examples.
    if highlight:
        print("\n[highlight episodes]")
        for e in highlight[:5]:
            print(
                f"- {e.id} | type={e.signal_type} | salience={e.salience:.2f} | "
                f"actors={e.actors[:3]} | events={sum(e.event_type_histogram.values())}"
            )

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader
elysia_loader.install()

from core.sensory.layered_sensory_interaction_spine import (
    EmbodiedPygameSensoryWorld,
    LayeredSensoryInteractionSpine,
    LiveScreenSensoryBridge,
    compute_human_likeness_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ELYSIA Live Layered Sensory Interaction Bridge (Pygame Arena or Real PC Screen)"
    )
    parser.add_argument(
        "--mode",
        choices=["pygame", "screen", "video"],
        default="pygame",
        help="pygame: interactive embodied arena window | screen: live PC desktop/YouTube observer via mss | video: bundled camera MP4",
    )
    parser.add_argument("--seconds", type=float, default=8.0, help="Duration for live screen observation")
    parser.add_argument("--fps", type=float, default=8.0, help="Capture FPS for live screen mode")
    args = parser.parse_args()

    spine = LayeredSensoryInteractionSpine()

    if args.mode == "pygame":
        world = EmbodiedPygameSensoryWorld(headless=False)
        snapshot, inferred = world.run_embodied_exploration(spine, max_steps_per_entity=16)
        video_res = LiveScreenSensoryBridge.observe_real_video(spine, max_frames=45)
        audit = compute_human_likeness_audit(spine, inferred, video_res)
        print("\n[Pygame Embodied Exploration Result]")
        for eid, role in inferred.items():
            c = spine.columns[eid]
            print(
                f"  {eid:14s} -> {role:26s} | "
                f"beam_mm={c.beam_micro_meso:.2f}, beam_mc={c.beam_meso_macro:.2f}, "
                f"boundary={c.boundary_voxel_id is not None}"
            )
        print(
            f"  Similarity collapse -> differentiation: {audit.pre_interaction_mean_similarity:.3f} -> "
            f"{audit.post_interaction_mean_similarity:.3f} (gain=+{audit.structural_differentiation_gain:.3f})"
        )
        time.sleep(2.0)
        world.pygame.quit()
    elif args.mode == "screen":
        print(f"[Live PC Monitor Bridge] Observing primary screen for {args.seconds:.1f}s @ {args.fps:.1f} FPS...")
        res = LiveScreenSensoryBridge.observe_live_monitor(spine, duration_sec=args.seconds, fps=args.fps)
        print(
            f"  Frames={res['frames_processed']} | L_micro={res['mean_micro_activity']:.3f} | "
            f"L_meso={res['mean_meso_activity']:.3f} | L_macro={res['mean_macro_persistence']:.3f} | "
            f"Beam Coherence={res['final_beam_coherence']:.3f}"
        )
    else:
        res = LiveScreenSensoryBridge.observe_real_video(spine, max_frames=90)
        print(
            f"[External MP4 Bridge] Frames={res['frames_processed']} | "
            f"L_micro={res['mean_micro_activity']:.3f} | L_meso={res['mean_meso_activity']:.3f} | "
            f"L_macro={res['mean_macro_persistence']:.3f} | Mean Coherence={res['mean_beam_coherence']:.3f}"
        )


if __name__ == "__main__":
    main()

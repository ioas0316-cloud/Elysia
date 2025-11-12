# -*- coding: utf-8 -*-
import os
import sys

# Ensure Starter package root is on sys.path (ElysiaStarter)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_THIS_DIR)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

try:
    from core.divine_engine import ElysiaDivineEngineV2  # noqa: F401
except Exception:
    ElysiaDivineEngineV2 = None  # optional


def main():
    # Optional: spin up engine if available (no-op if not used by viewer)
    if ElysiaDivineEngineV2 is not None:
        try:
            eng = ElysiaDivineEngineV2()  # noqa: F841
        except Exception:
            pass

    # Hand over to the timeline viewer
    import scripts.visualize_timeline as viz
    viz.main()


if __name__ == "__main__":
    main()


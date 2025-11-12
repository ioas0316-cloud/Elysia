# -*- coding: utf-8 -*-
import os
import sys
import importlib
import importlib.util

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

    # Prefer package import so PyInstaller can collect it
    try:
        viz = importlib.import_module('ElysiaStarter.scripts.visualize_timeline')
        if hasattr(viz, 'main'):
            viz.main()
            return
    except Exception as e:
        # If running as frozen app, do not attempt filesystem fallback
        if getattr(sys, 'frozen', False):
            print('Failed to import ElysiaStarter.scripts.visualize_timeline in packaged app:', e)
            return

    # Fallback to file-path import for source runs only
    viz_path = os.path.join(_PKG_ROOT, 'scripts', 'visualize_timeline.py')
    spec = importlib.util.spec_from_file_location('starter_visualize_timeline', viz_path)
    if spec and spec.loader:
        viz = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(viz)
        if hasattr(viz, 'main'):
            viz.main()
        else:
            print('visualize_timeline.main() not found')
    else:
        print('Failed to load visualize_timeline.py from Starter package')


if __name__ == "__main__":
    main()

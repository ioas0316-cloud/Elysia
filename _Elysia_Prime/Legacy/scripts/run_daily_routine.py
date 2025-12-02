# [Genesis: 2025-12-02] Purified by Elysia
"""
Run a daily offline routine combining journaling, optional book report,
and creative writing, anchoring all outputs into the KG.

Usage:
  python -m scripts.run_daily_routine [--book path/to/book.txt] [--genre fantasy] [--theme hope]
"""
from __future__ import annotations

import argparse
from importlib import import_module


def _run_journaling():
    mod = import_module('scripts.run_journaling_lesson')
    mod.main()


def _run_book_report(book_path: str):
    mod = import_module('scripts.run_book_report')
    # Simulate CLI: build args namespace manually
    import argparse as _ap
    _ns = _ap.Namespace(book=book_path)
    # Reuse module logic by calling main via a small shim
    # Better: expose a function, but we keep minimal footprint
    mod.main.__globals__['argparse'].ArgumentParser = lambda *a, **k: _FakeParser(_ns)
    mod.main()


def _run_creative(genre: str, theme: str):
    mod = import_module('scripts.run_creative_writing')
    import argparse as _ap
    _ns = _ap.Namespace(genre=genre, theme=theme, beats=5, words=120)
    mod.main.__globals__['argparse'].ArgumentParser = lambda *a, **k: _FakeParser(_ns)
    mod.main()


class _FakeParser:
    def __init__(self, ns):
        self._ns = ns
    def add_argument(self, *a, **k):
        return None
    def parse_args(self):
        return self._ns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--book', help='Optional local text file for book report')
    parser.add_argument('--genre', default='story')
    parser.add_argument('--theme', default='growth')
    args = parser.parse_args()

    print('[Daily Routine] Journaling...')
    _run_journaling()

    if args.book:
        print('[Daily Routine] Book report...')
        _run_book_report(args.book)
    else:
        print('[Daily Routine] Skipping book report (no --book provided).')

    print('[Daily Routine] Creative writing...')
    _run_creative(args.genre, args.theme)
    print('[Daily Routine] Done.')


if __name__ == '__main__':
    main()

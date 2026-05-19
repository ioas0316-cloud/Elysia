"""
Ingest simple dialogue corpus into the KG.

Input format (txt): lines like
  A: 안녕
  B: 안녕하세요
Blank lines separate conversations.

Usage:
  python -m scripts.ingest_dialog_corpus --path data/corpus/dialogues/sample_dialog.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from tools.kg_manager import KGManager


def parse_dialogue(path: Path):
    convs = []
    current = []
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        s = line.strip()
        if not s:
            if current:
                convs.append(current)
                current = []
            continue
        current.append(s)
    if current:
        convs.append(current)
    return convs


def ingest(path: Path, kg: KGManager):
    convs = parse_dialogue(path)
    for i, conv in enumerate(convs, start=1):
        conv_id = f"dialog_{path.stem}_{i}"
        kg.add_node(conv_id, properties={"type": "dialogue", "source": str(path)})
        last_speaker = None
        for turn in conv:
            if ':' in turn:
                speaker, text = turn.split(':', 1)
            else:
                speaker, text = 'Unknown', turn
            speaker = speaker.strip()
            text = text.strip()
            tid = f"{conv_id}_" + str(abs(hash(turn)) % 10**8)
            kg.add_node(tid, properties={"type": "utterance", "speaker": speaker, "text": text})
            kg.add_edge(conv_id, tid, "has_utterance")
            if last_speaker and last_speaker != speaker:
                kg.add_edge(f"speaker_{last_speaker}", f"speaker_{speaker}", "speaks_to")
            kg.add_node(f"speaker_{speaker}", properties={"type": "actor"})
            last_speaker = speaker
    kg.save()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--path', required=True)
    args = p.parse_args()
    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    kg = KGManager()
    ingest(path, kg)
    print('Ingested dialogues from', path)


if __name__ == '__main__':
    main()


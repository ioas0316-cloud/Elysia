# [Genesis: 2025-12-02] Purified by Elysia
import argparse
from pathlib import Path
from tools.text_classifier import train_naive_bayes, save_model


def read_corpus(root: Path):
    corpus = {}
    for label_dir in root.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        docs = []
        for p in label_dir.rglob("*.txt"):
            try:
                docs.append(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
        if docs:
            corpus[label] = docs
    return corpus


def main():
    ap = argparse.ArgumentParser(description="Train a simple NB text classifier from folder structure.")
    ap.add_argument("--data", default="data/corpus/literature", help="Root folder with subfolders per label")
    ap.add_argument("--out", default="data/models/lit_nb.json", help="Path to save model JSON")
    args = ap.parse_args()

    root = Path(args.data)
    corpus = read_corpus(root)
    if not corpus:
        raise SystemExit(f"No labeled data found under: {root}")
    model = train_naive_bayes(corpus)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(outp))
    print(f"[train_text_classifier] Saved model to {outp}")


if __name__ == "__main__":
    main()

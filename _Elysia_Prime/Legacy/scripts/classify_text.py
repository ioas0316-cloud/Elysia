# [Genesis: 2025-12-02] Purified by Elysia
import argparse
from pathlib import Path
from tools.text_classifier import load_model, predict


def main():
    ap = argparse.ArgumentParser(description="Classify a text or file using a trained NB model.")
    ap.add_argument("--model", default="data/models/lit_nb.json")
    ap.add_argument("--file", help=".txt file to classify", default=None)
    ap.add_argument("--text", help="Raw text to classify", default=None)
    args = ap.parse_args()

    if not args.file and not args.text:
        raise SystemExit("Provide --file or --text")

    text = args.text
    if args.file:
        p = Path(args.file)
        if not p.exists():
            raise SystemExit(f"File not found: {p}")
        text = p.read_text(encoding="utf-8", errors="ignore")

    model = load_model(args.model)
    label, probs = predict(model, text or "")
    print("label:", label)
    for k, v in probs:
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()

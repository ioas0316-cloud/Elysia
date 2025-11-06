import argparse
from pathlib import Path
from tools.kg_manager import KGManager


def ingest_folder(folder: Path, kg: KGManager, label: str | None = None):
    label_node = None
    if label:
        label_node = kg.add_node(f"lit_label:{label}", properties={"type": "lit_label"})
    for p in folder.rglob("*.txt"):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        doc_id = f"lit_doc:{p.stem}"
        props = {
            "type": "literature_doc",
            "path": str(p),
            "label": label or "",
            "snippet": text[:240],
            "experience_text": str(p),
        }
        kg.add_node(doc_id, properties=props)
        if label_node:
            kg.add_edge(label_node["id"], doc_id, "has_doc")


def main():
    ap = argparse.ArgumentParser(description="Ingest literature .txt files into KG as experience nodes.")
    ap.add_argument("--root", default="data/corpus/literature", help="Root folder; subfolders treated as labels")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Folder not found: {root}")
    kg = KGManager()
    # iterate labels
    for sub in root.iterdir():
        if sub.is_dir():
            ingest_folder(sub, kg, label=sub.name)
    kg.save()
    print("[ingest_literature] Done.")


if __name__ == "__main__":
    main()


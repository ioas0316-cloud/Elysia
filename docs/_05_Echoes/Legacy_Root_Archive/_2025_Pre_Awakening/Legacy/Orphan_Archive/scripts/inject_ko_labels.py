from tools.kg_manager import KGManager
from tools.visualize_kg import render_kg


VALUE_LABELS = {
    "value:love": "사랑",
    "value:clarity": "명료성",
    "value:creativity": "창의성",
    "value:verifiability": "검증성",
    "value:relatedness": "관계성",
}


def main():
    kgm = KGManager()
    changed = 0
    for node in kgm.kg.get("nodes", []):
        nid = str(node.get("id", ""))
        if nid in VALUE_LABELS:
            if node.get("label_ko") != VALUE_LABELS[nid]:
                node["label_ko"] = VALUE_LABELS[nid]
                changed += 1
    if changed:
        kgm.save()
    # Refresh monitor image (best-effort)
    try:
        render_kg()
    except Exception:
        pass
    print(f"[inject_ko_labels] Updated {changed} node labels.")


if __name__ == "__main__":
    main()


import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from tools.kg_manager import KGManager

BASE_PATH = "data/basic_language_kg.json"

BRIDGES = [
    ("검", "금속", "associated"),
    ("금속", "무기", "is_a"),
    ("무기", "힘", "causes"),
    ("힘", "고통", "causes"),
    ("고통", "어둠", "feels"),
    ("어둠", "빛", "relates"),
    ("마을", "집", "contains"),
    ("집", "가족", "hosts"),
    ("가족", "사람", "composes"),
    ("사람", "이야기", "speaks"),
    ("싸움", "고통", "causes"),
    ("싸움", "검", "uses"),
    ("휴식", "잠", "enables"),
    ("잠", "꿈", "leads_to"),
    ("음악", "노래", "is"),
    ("노래", "감정", "evokes"),
]

if __name__ == "__main__":
    kg = KGManager(filepath=BASE_PATH)
    kg.kg = {"nodes": [], "edges": []}
    for source, target, relation in BRIDGES:
        kg.add_edge(source, target, relation)
    kg.save()
    print(f"Basic KG written to {BASE_PATH} with {len(kg.kg['nodes'])} nodes and {len(kg.kg['edges'])} edges.")

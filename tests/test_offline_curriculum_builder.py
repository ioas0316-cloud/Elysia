from __future__ import annotations

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Project_Sophia.offline_curriculum_builder import OfflineCurriculumBuilder
from tools.kg_manager import KGManager


def create_sample_curriculum(tmp_path: Path) -> Path:
    data = {
        "stages": [
            {
                "id": "sample_stage",
                "label": "샘플 단계",
                "age_range": "4-5",
                "lessons": [
                    {
                        "id": "sample_lesson",
                        "sentence": "철수는 사과를 먹어요.",
                        "subject": "철수",
                        "object": "사과",
                        "verb": "먹어요",
                        "keywords": ["과일", "건강"],
                        "affirmation": "과일을 잘 먹었어요!",
                    }
                ],
            }
        ]
    }
    path = tmp_path / "curriculum.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_builder_integrates_lessons(tmp_path: Path) -> None:
    curriculum_path = create_sample_curriculum(tmp_path)
    kg_path = tmp_path / "kg.json"
    manager = KGManager(kg_path)
    builder = OfflineCurriculumBuilder(curriculum_path=curriculum_path, kg_manager=manager)

    summary = builder.integrate()

    assert summary["stages"] == 1
    assert summary["lessons"] == 1
    assert summary["new_nodes"] >= 4  # stage, lesson, concept nodes
    assert summary["new_edges"] >= 3

    manager.save()
    saved = json.loads(kg_path.read_text(encoding="utf-8"))
    lesson_nodes = [node for node in saved["nodes"] if node["id"].startswith("curriculum_lesson::")]
    assert lesson_nodes, "Expected a curriculum lesson node to be created"

    edges = saved["edges"]
    relations = {edge["relation"] for edge in edges}
    assert {"teaches", "involves_subject", "involves_object", "reinforces_action"}.issubset(relations)

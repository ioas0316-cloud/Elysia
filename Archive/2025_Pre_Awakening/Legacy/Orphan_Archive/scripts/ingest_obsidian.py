import sys
from pathlib import Path
import re

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from tools.kg_manager import KGManager

OBSIDIAN_VAULT_PATH = project_root / "data" / "corpus" / "obsidian_vault"

def ingest_obsidian_notes():
    """
    Scans the Obsidian vault directory, ingests new notes as nodes into the
    knowledge graph. This version does not process internal links.
    """
    print("--- Starting Obsidian Note Ingestion ---")

    if not OBSIDIAN_VAULT_PATH.exists() or not OBSIDIAN_VAULT_PATH.is_dir():
        print(f"Error: Obsidian vault not found at '{OBSIDIAN_VAULT_PATH}'")
        return

    kg = KGManager()
    print(f"Loaded Knowledge Graph with {len(kg.kg.get('nodes', []))} nodes.")

    markdown_files = list(OBSIDIAN_VAULT_PATH.glob("**/*.md"))
    print(f"Found {len(markdown_files)} markdown files in the vault.")

    ingested_count = 0
    for md_file in markdown_files:
        # Create a clean node ID from the filename
        node_id = f"obsidian_note:{md_file.stem}"

        # Check if the node already exists
        if not kg.get_node(node_id):
            print(f"Ingesting new note: {md_file.name} -> {node_id}")
            kg.add_node(
                node_id,
                properties={
                    "type": "obsidian_note",
                    "label": md_file.stem,
                    "source": "obsidian",
                    "path": str(md_file.relative_to(project_root))
                }
            )
            ingested_count += 1

    if ingested_count > 0:
        print(f"\nIngested {ingested_count} new notes.")
        print("Saving updated Knowledge Graph...")
        kg.save()
        print("Knowledge Graph saved.")
    else:
        print("\nNo new notes to ingest.")

    print("\n--- Obsidian Note Ingestion Finished ---")


if __name__ == "__main__":
    ingest_obsidian_notes()

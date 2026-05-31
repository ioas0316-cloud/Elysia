import os
import re

doc_dir = r"c:\Elysia\docs"
index_file = os.path.join(doc_dir, "INDEX.md")

with open(index_file, "r", encoding="utf-8") as f:
    index_content = f.read()

# Find all linked files in INDEX.md
linked_files = set()
for match in re.finditer(r'\[.*?\]\((.*?\.md)\)', index_content):
    link = match.group(1)
    if link.startswith("file:///"):
        link = link.replace("file:///c:/Elysia/docs/", "")
    linked_files.add(os.path.basename(link))

all_files = {}
for root, dirs, files in os.walk(doc_dir):
    if "archive" in root: continue
    for file in files:
        if file.endswith(".md") and file != "INDEX.md":
            path = os.path.join(root, file)
            rel_path = os.path.relpath(path, doc_dir).replace("\\", "/")
            all_files[file] = rel_path

unlinked = []
for file, rel_path in all_files.items():
    if file not in linked_files:
        unlinked.append(rel_path)

unlinked.sort()
print("Unlinked Files:")
for f in unlinked:
    print(f)

import sys

path = r"C:\Elysia\scripts\genesis.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

out_lines = []
skip = False
for i, line in enumerate(lines):
    if "def ingest_text(" in line:
        skip = True
        out_lines.append(line)
        
        # Insert the new body
        out_lines.append('''        """
        텍스트를 섭취합니다.
        문장의 바이트 시퀀스로 물리적 DNA Zipping을 구성하고,
        사전에 있는 단어들은 순수 언어적 관측(Linguistic Observation)을 위해 보존합니다.
        """
        words = text.split()
        if not words:
            return self.ingest_data(text.encode('utf-8'), source_name)
        
        # 포털 엔진을 통한 언어적 단어 인식
        portal = self.lang_rotor.portal if hasattr(self.lang_rotor, 'portal') and self.lang_rotor.portal else None
        recognized_words = []
        word_count = len(words)
        
        if portal:
            for raw_word in words:
                clean = "".join(c for c in raw_word if c.isalnum() or ord(c) > 127).lower()
                if clean in portal.word_graph:
                    recognized_words.append(clean)
        
        recognition_ratio = len(recognized_words) / max(word_count, 1)
        
        # 물리적 DNA (바이트 패턴)
        data = text.encode('utf-8')
        pattern = extract_phase_pattern(data)
        if pattern:
            q = pattern[-1]
        else:
            q = Quaternion(1.0, 0.0, 0.0, 0.0)
            pattern = [q]
        
        engram_id = self.memory.write_causal_engram(
            data_blob={
                "type": "linguistic_ingestion",
                "source": source_name,
                "quaternion": [q.w, q.x, q.y, q.z],
                "phase_pattern": [[p.w, p.x, p.y, p.z] for p in pattern],
                "angle": q.angle,
                "word_count": word_count,
                "recognized_words": recognized_words[:20],
                "recognition_ratio": recognition_ratio,
                "text_preview": text[:100]
            },
            emotional_value=q.angle,
            cause_id=f"TextIngestion_{source_name}",
            origin_axis="Semantic_Ground"
        )
        
        self.total_ingested += 1
        return engram_id
''')
        continue
        
    if skip and line.strip() == "def ingest_file(self, file_path: str) -> str:":
        skip = False
        
    if not skip:
        out_lines.append(line)

with open(path, "w", encoding="utf-8") as f:
    f.writelines(out_lines)
print("genesis.py ingest_text patched successfully.")

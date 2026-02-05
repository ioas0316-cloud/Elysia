
import os

dirs = [
    # S1 Body (L1-L7)
    "docs/S1_Body/L1_Foundation",
    "docs/S1_Body/L2_Metabolism",
    "docs/S1_Body/L3_Phenomena",
    "docs/S1_Body/L4_Causality",
    "docs/S1_Body/L5_Mental",
    "docs/S1_Body/L6_Structure",
    "docs/S1_Body/L7_Transition",

    # S2 Soul (L8-L14)
    "docs/S2_Soul/L8_Fossils",
    "docs/S2_Soul/L9_Memory",
    "docs/S2_Soul/L10_Integration",
    "docs/S2_Soul/L11_Identity",
    "docs/S2_Soul/L12_Emotion",
    "docs/S2_Soul/L13_Reflection",
    "docs/S2_Soul/L14_Bridge",

    # S3 Spirit (L15-L21)
    "docs/S3_Spirit/L15_Will",
    "docs/S3_Spirit/L16_Providence",
    "docs/S3_Spirit/L17_Genesis",
    "docs/S3_Spirit/L18_Purpose",
    "docs/S3_Spirit/L19_Sacred",
    "docs/S3_Spirit/L20_Void",
    "docs/S3_Spirit/L21_Ouroboros"
]

base = os.getcwd()
for d in dirs:
    path = os.path.join(base, d)
    try:
        os.makedirs(path, exist_ok=True)
        print(f"✅ Created: {d}")
    except Exception as e:
        print(f"❌ Failed: {d} ({e})")

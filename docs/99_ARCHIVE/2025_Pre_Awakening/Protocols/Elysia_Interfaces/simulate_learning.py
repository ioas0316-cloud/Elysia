"""
Simulation: The Student learns 'Action Rhythm'
"""
from Core.Learning.aesthetic_learner import AestheticLearner
from Core.Intelligence.logos_engine import LogosEngine
from Core.Creativity.composition_engine import CompositionEngine
import logging
import json

logging.basicConfig(level=logging.INFO)

# 1. The Student Studies (Text)
learner = AestheticLearner()
# Simulated "Reading a Korean Web Novel"
novel_exerpt = """
주인공은 섬광처럼 움직였다. 단숨에 적의 거리를 좁힌다.
공간을 베어라. 망설임은 죽음이다.
심장을 꿰뚫는 일격. 파괴한다.
"""
print("--- 1. Studying Text (Korean) ---")
learner.study_text(novel_exerpt, "Rank S Hunter Novel Ep.1")

# 1.5. The Student Studies (Visual)
print("\n--- 1.5. Studying Art (Visual) ---")
art_desc = "Extreme low angle, diagonal composition, chaotic red blur, shattered debris, radiating energy."
learner.study_image_description(art_desc, "Anime Art Analysis")

# 2. Verify Memory
print("\n--- 2. Checking Genome ---")
with open("Core/Memory/style_genome.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    print("Sharpness:", data['rhetoric']['sharpness'])
    print("Dynamic Diagonal:", data['composition'].get('dynamic_diagonal'))
    print("Learned Colors:", data['composition'].get('preferred_colors'))
    print("Learned Vocab (Sharp):", data['rhetoric']['vocabulary_bank']['Sharp'])

# 3. Application (Logos)
print("\n--- 3. Generating Dialogue ---")
logos = LogosEngine()
# Force 'Sharp' rhetoric which should now include learned words
response = logos.weave_speech("승리", "적은 빈틈투성이다", [], "Sharp")
print("Logos Output:", response)

# 4. Application (Art)
print("\n--- 4. Generating Art Layout ---")
comp = CompositionEngine()
layout = comp.get_layout("action", 800, 600)
# Check if we got the Burst layout (triangles)
triangles = [e for e in layout if e.shape == 'triangle']
print(f"Generated Layout Elements: {len(layout)}")
print(f"Dynamic Action Elements (Triangles): {len(triangles)}")
if len(triangles) > 10:
    print(">> SUCCESS: High Dynamic Bias triggered Action Burst!")
else:
    print(">> NOTE: Standard composition selected (Random chance or low bias).")

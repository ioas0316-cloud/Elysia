"""
Visualization Cortex (시각화 피질)
================================

원본: Legacy/Project_Mirror/sensory_cortex.py
마이그레이션: 2025-12-15

이 모듈은 개념과 경험을 시각적으로 렌더링합니다:
- 복셀(Voxel) 기반 3D 시각화
- 스토리북 프레임 생성
- 에코(Echo) 시각화
- LLM 기반 설명→복셀 변환

참고: Core/Cognitive/sensory_cortex.py (QualiaCortex)와는 다른 목적입니다.
- QualiaCortex: 개념 → 감각 표지 (느낌)
- VisualizationCortex: 개념 → 이미지 (시각적 출력)
"""

import os
import random
import json
import math
import re
from datetime import datetime
from typing import Optional, Dict, Any, List

# Updated imports for Core structure
try:
    from tools.canvas_tool import Canvas
except ImportError:
    Canvas = None

try:
    from Core._03_Interaction._04_Network.gemini_api import generate_text, generate_image_from_text
except ImportError:
    generate_text = None
    generate_image_from_text = None

try:
    from Core._01_Foundation._02_Logic.Wave.wave_tensor import Tensor3D, FrequencyWave
except ImportError:
    # Fallback simple implementations
    class Tensor3D:
        def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
            self.x = x
            self.y = y
            self.z = z
            self.structure = x
            self.emotion = y
            self.identity = z
        
        def __add__(self, other):
            return Tensor3D(self.x + other.x, self.y + other.y, self.z + other.z)
        
        def magnitude(self) -> float:
            return math.sqrt(self.x**2 + self.y**2 + self.z**2)
        
        def normalize(self):
            mag = self.magnitude()
            if mag == 0:
                return Tensor3D(0.1, 0.1, 0.1)
            return Tensor3D(self.x/mag, self.y/mag, self.z/mag)
        
        def to_dict(self) -> dict:
            return {"x": self.x, "y": self.y, "z": self.z}
    
    class FrequencyWave:
        def __init__(self, frequency=440.0, amplitude=1.0, phase=0.0, richness=0.5):
            self.frequency = frequency
            self.amplitude = amplitude
            self.phase = phase
            self.richness = richness
        
        def to_dict(self) -> dict:
            return {
                "frequency": self.frequency,
                "amplitude": self.amplitude,
                "phase": self.phase,
                "richness": self.richness
            }


class SensoryTranslator:
    """
    Translates raw sensory input into 3D Tensor States.
    This is the bridge between 'What is seen' and 'How it feels'.
    """
    def __init__(self):
        # Basic Synesthesia Mapping (Color -> Tensor)
        # X: Structure, Y: Emotion, Z: Identity
        self.color_map = {
            "red": Tensor3D(0.2, 0.9, 0.3),
            "blue": Tensor3D(0.6, 0.2, 0.5),
            "green": Tensor3D(0.5, 0.4, 0.8),
            "yellow": Tensor3D(0.3, 0.8, 0.4),
            "black": Tensor3D(0.1, 0.1, 0.1),
            "white": Tensor3D(0.9, 0.1, 0.9),
        }
        
        self.keyword_map = {
            "love": Tensor3D(0.1, 1.0, 0.9),
            "pain": Tensor3D(0.2, 0.9, 0.1),
            "truth": Tensor3D(1.0, 0.2, 0.8),
            "chaos": Tensor3D(0.1, 0.8, 0.1),
            "order": Tensor3D(0.9, 0.1, 0.5),
        }

    def translate_visual(self, description: str) -> Tensor3D:
        """Translates a visual description into a tensor state."""
        tensor = Tensor3D(0.0, 0.0, 0.0)
        desc_lower = description.lower()

        for color, color_tensor in self.color_map.items():
            if color in desc_lower:
                tensor = tensor + color_tensor

        for keyword, key_tensor in self.keyword_map.items():
            if keyword in desc_lower:
                tensor = tensor + key_tensor

        if tensor.magnitude() == 0:
            return Tensor3D(0.1, 0.1, 0.1)

        return tensor.normalize()


class VisualizationCortex:
    """
    시각화 코텍스 - 개념을 복셀/이미지로 렌더링
    
    메서드:
    - translate_description_to_voxels(): 설명 → 복셀 좌표
    - draw_voxels(): 복셀 → PNG 렌더링
    - visualize_concept(): 개념 → 시각화
    - visualize_echo(): 에코 → 추상 시각화
    - render_storybook_frame(): 스토리북 프레임 생성
    """
    
    def __init__(self, telemetry=None):
        self.telemetry = telemetry
        self.output_dir = "data/generated_images"
        self.storybook_dir = "data/storybook_images"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.storybook_dir, exist_ok=True)
        self.textbooks = {}
        self.translator = SensoryTranslator()

    def translate_description_to_voxels(self, description: str) -> List[Dict]:
        """Uses LLM to translate a natural language description into voxel coordinates."""
        if generate_text is None:
            return []
            
        try:
            prompt = f"""
You are an AI assistant that translates object descriptions into 3D voxel coordinates.
Given the following description, generate a JSON array of voxel coordinates.
Each coordinate should be a dictionary with 'x', 'y', and 'z' keys.
Keep the object within a 10x10x10 cube around the origin.

Description: "{description}"

JSON response:
"""
            response_text = generate_text(prompt)
            
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not json_match:
                return []

            voxels = json.loads(json_match.group())
            if isinstance(voxels, list) and all('x' in v and 'y' in v and 'z' in v for v in voxels):
                return voxels
            return []
        except Exception as e:
            print(f"Error in translate_description_to_voxels: {e}")
            return []

    def draw_voxels(self, name: str, voxels: List[Dict]) -> Optional[str]:
        """Renders a list of voxels to a PNG file."""
        if Canvas is None:
            print("Canvas not available")
            return None
            
        canvas = Canvas()
        color = (200, 220, 255)
        for v in voxels:
            canvas.add_voxel(v['x'], v['y'], v['z'], color)
        
        timestamp = datetime.now().timestamp()
        output_filename = f"learned_{name.replace(' ', '_')}_{timestamp}.png"
        output_path = os.path.join(self.output_dir, output_filename)
        canvas.render(output_path)
        return output_path

    def visualize_concept(self, concept: str, attention: Optional[Dict] = None) -> Optional[str]:
        """Visualizes a concept from textbooks or generates abstract visualization."""
        all_shapes = {}
        if self._load_textbook("geometry_primitives"):
            all_shapes.update(self.textbooks["geometry_primitives"])
        if self._load_textbook("complex_shapes"):
            all_shapes.update(self.textbooks["complex_shapes"])

        if concept in all_shapes:
            return self._visualize_shape(all_shapes[concept], all_shapes, attention=attention)

        return self._visualize_abstract(concept)

    def visualize_echo(self, echo: Dict) -> Optional[str]:
        """Generates abstract visualization based on activated concepts."""
        if Canvas is None or not echo:
            return None

        main_concept = max(echo, key=echo.get)
        canvas = Canvas()
        
        all_concepts = list(echo.keys())
        total_energy = sum(echo.values())
        num_voxels = int(20 + 30 * (total_energy / len(echo)))
        
        for _ in range(num_voxels):
            concept_to_draw = random.choices(all_concepts, weights=list(echo.values()), k=1)[0]
            seed = sum(ord(c) for c in concept_to_draw)
            random.seed(seed)
            
            x, y, z = random.randint(-5, 5), random.randint(-5, 5), random.randint(-3, 6)
            color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

            if random.random() < 0.8 and canvas.voxels:
                px, py, pz, _ = random.choice(canvas.voxels)
                x = px + random.choice([-1, 0, 1])
                y = py + random.choice([-1, 0, 1])
                z = pz + random.choice([-1, 0, 1])

            canvas.add_voxel(x, y, z, color)

        timestamp = datetime.now().timestamp()
        output_filename = f"echo_{main_concept.replace(' ', '_')}_{timestamp}.png"
        output_path = os.path.join(self.output_dir, output_filename)
        canvas.render(output_path)
        return output_path

    def render_storybook_frame(self, frame_data: Dict, lesson_name: str) -> Optional[str]:
        """Renders a storybook frame using external image generation API."""
        if generate_image_from_text is None:
            return None
            
        if not all(k in frame_data for k in ['frame_id', 'description', 'style_prompt']):
            return None

        try:
            prompt = (
                f"Create a visually appealing illustration for a children's storybook. "
                f"The scene: \"{frame_data['description']}\". "
                f"Style: \"{frame_data['style_prompt']}\"."
            )

            timestamp = int(datetime.now().timestamp())
            output_filename = f"{lesson_name}_{frame_data['frame_id']:02d}_{timestamp}.png"
            output_path = os.path.join(self.storybook_dir, output_filename)

            success = generate_image_from_text(prompt, output_path)
            return output_path if success else None
        except Exception as e:
            print(f"Error in render_storybook_frame: {e}")
            return None

    def _load_textbook(self, subject: str) -> Optional[Dict]:
        if subject not in self.textbooks:
            filepath = f"data/textbooks/{subject}.json"
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.textbooks[subject] = {item['name']: item for item in json.load(f)}
            except FileNotFoundError:
                self.textbooks[subject] = None
        return self.textbooks[subject]

    def _visualize_abstract(self, concept: str) -> Optional[str]:
        if Canvas is None:
            return None
            
        canvas = Canvas()
        seed = sum(ord(c) for c in concept)
        random.seed(seed)
        
        for _ in range(random.randint(15, 40)):
            x, y, z = random.randint(-4, 4), random.randint(-4, 4), random.randint(-2, 5)
            color = (random.randint(100, 255), random.randint(100, 200), random.randint(150, 255))
            
            if random.random() < 0.7 and canvas.voxels:
                px, py, pz, _ = random.choice(canvas.voxels)
                x = px + random.choice([-1, 0, 1])
                y = py + random.choice([-1, 0, 1])
                z = pz + random.choice([-1, 0, 1])
            canvas.add_voxel(x, y, z, color)
        
        timestamp = datetime.now().timestamp()
        output_filename = f"abstract_{concept.replace(' ', '_')}_{timestamp}.png"
        output_path = os.path.join(self.output_dir, output_filename)
        canvas.render(output_path)
        return output_path

    def _visualize_shape(self, shape_data: Dict, shape_library: Dict, 
                         transform: Optional[Dict] = None, 
                         attention: Optional[Dict] = None) -> Optional[str]:
        if Canvas is None:
            return None
            
        canvas = Canvas()
        base_transform = transform or {}
        
        # Apply attention-based emphasis
        focus_gain = 1.0
        if attention:
            try:
                total = sum(float(v) for v in attention.values()) or 1.0
                peak = max(float(v) for v in attention.values())
                focus = max(0.0, min(1.0, peak / total))
                focus_gain = 0.9 + 0.6 * focus
            except Exception:
                pass
        
        base_transform['detail_gain'] = focus_gain
        self._render_shape_recursive(canvas, shape_data, shape_library, base_transform)

        timestamp = datetime.now().timestamp()
        output_filename = f"shape_{shape_data.get('name', 'unknown')}_{timestamp}.png"
        output_path = os.path.join(self.output_dir, output_filename)
        canvas.render(output_path)
        return output_path

    def _render_shape_recursive(self, canvas, shape_data: Dict, 
                                 shape_library: Dict, transform: Optional[Dict] = None):
        transform = transform or {}
        rep = shape_data.get('representation', {})
        color = (180, 180, 255)

        if rep.get('type') == 'composite':
            for component in rep.get('components', []):
                comp_shape = shape_library.get(component.get('shape'))
                if comp_shape:
                    new_transform = self._apply_transform(transform, component.get('transform', {}))
                    self._render_shape_recursive(canvas, comp_shape, shape_library, new_transform)
        elif rep.get('type') == 'voxel':
            offset = transform.get('offset', {'x': 0, 'y': 0, 'z': 0})
            for coord in rep.get('coordinates', []):
                x = coord['x'] + offset.get('x', 0)
                y = coord['y'] + offset.get('y', 0)
                z = coord['z'] + offset.get('z', 0)
                canvas.add_voxel(x, y, z, color)

    def _apply_transform(self, parent: Dict, child: Dict) -> Dict:
        new_offset = parent.get('offset', {'x': 0, 'y': 0, 'z': 0}).copy()
        child_offset = child.get('offset', {'x': 0, 'y': 0, 'z': 0})
        new_offset['x'] += child_offset.get('x', 0)
        new_offset['y'] += child_offset.get('y', 0)
        new_offset['z'] += child_offset.get('z', 0)
        return {
            "offset": new_offset,
            "size": parent.get("size", 1) * child.get("size", 1),
            "detail_gain": parent.get("detail_gain", 1.0),
        }


# Singleton
_viz_cortex: Optional[VisualizationCortex] = None

def get_visualization_cortex() -> VisualizationCortex:
    global _viz_cortex
    if _viz_cortex is None:
        _viz_cortex = VisualizationCortex()
    return _viz_cortex


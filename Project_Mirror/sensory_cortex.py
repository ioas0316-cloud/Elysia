import os
import random
import json
import math
import re
from datetime import datetime
from tools.canvas_tool import Canvas
from Project_Sophia.value_cortex import ValueCortex
try:
    from infra.telemetry import Telemetry
except Exception:
    Telemetry = None
from Project_Sophia.gemini_api import generate_text, generate_image_from_text

class SensoryCortex:
    def __init__(self, value_cortex: ValueCortex, telemetry: Telemetry | None = None):
        self.value_cortex = value_cortex
        self.telemetry = telemetry
        self.output_dir = "data/generated_images"
        self.storybook_dir = "data/storybook_images"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.storybook_dir, exist_ok=True)
        self.textbooks = {}

    def translate_description_to_voxels(self, description: str) -> list:
        """
        Uses the LLM to translate a natural language description into a list of voxel coordinates.
        """
        try:
            prompt = f"""
You are an AI assistant that translates object descriptions into 3D voxel coordinates.
Given the following description, generate a JSON array of voxel coordinates.
Each coordinate should be a dictionary with 'x', 'y', and 'z' keys.
The origin (0,0,0) is the center of the object.
Keep the object relatively small, within a 10x10x10 cube around the origin.
The output should be only the raw JSON array.

Description: "{description}"

JSON response:
"""
            response_text = generate_text(prompt)
            
            # Extract the JSON part of the response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not json_match:
                print(f"Warning: Could not find a JSON array in the LLM response for description: {description}")
                return []

            voxels = json.loads(json_match.group())
            # Basic validation
            if isinstance(voxels, list) and all('x' in v and 'y' in v and 'z' in v for v in voxels):
                return voxels
            else:
                print(f"Warning: Invalid voxel data format received for description: {description}")
                return []
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: Could not parse voxel data from LLM response: {response_text}. Error: {e}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred during voxel translation: {e}")
            return []

    def draw_voxels(self, name: str, voxels: list) -> str:
        """
        Renders a list of voxels to a PNG file.
        """
        canvas = Canvas()
        color = (200, 220, 255) # A nice light blue for learned shapes
        for v in voxels:
            canvas.add_voxel(v['x'], v['y'], v['z'], color)
        
        timestamp = datetime.now().timestamp()
        output_filename = f"learned_{name.replace(' ', '_')}_{timestamp}.png"
        output_path = os.path.join(self.output_dir, output_filename)
        canvas.render(output_path)
        if self.telemetry:
            try:
                self.telemetry.emit('render_done', {
                    'kind': 'voxels',
                    'name': name,
                    'num_voxels': len(voxels),
                    'output_path': output_path,
                })
            except Exception:
                pass
        return output_path

    def save_learned_shape(self, name: str, description: str, voxels: list):
        """
        Saves a newly learned shape to the 'learned_shapes.json' textbook.
        """
        textbook_path = "data/textbooks/learned_shapes.json"
        
        new_shape = {
            "name": name,
            "description": description,
            "representation": {
                "type": "voxel",
                "coordinates": voxels
            }
        }

        try:
            if os.path.exists(textbook_path):
                with open(textbook_path, 'r+', encoding='utf-8') as f:
                    shapes = json.load(f)
                    # Avoid duplicates
                    if not any(s['name'] == name for s in shapes):
                        shapes.append(new_shape)
                        f.seek(0)
                        json.dump(shapes, f, indent=2, ensure_ascii=False)
            else:
                with open(textbook_path, 'w', encoding='utf-8') as f:
                    json.dump([new_shape], f, indent=2, ensure_ascii=False)
            
            # Invalidate cache for this textbook if it was loaded
            if "learned_shapes" in self.textbooks:
                del self.textbooks["learned_shapes"]

        except Exception as e:
            print(f"Error saving learned shape '{name}': {e}")

    def _load_textbook(self, subject: str):
        if subject not in self.textbooks:
            filepath = f"data/textbooks/{subject}.json"
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.textbooks[subject] = {item['name']: item for item in json.load(f)}
            except FileNotFoundError:
                self.textbooks[subject] = None
        return self.textbooks[subject]

    def _get_color_palette(self, concept: str) -> list:
        # ... (rest of the method is unchanged)
        connections = self.value_cortex.find_meaning_connection(concept)
        if not connections: return [(100, 100, 150), (150, 100, 100), (100, 150, 100)]
        palette, base_colors = [], {"love": (255, 105, 180), "growth": (50, 205, 50), "creation": (138, 43, 226), "truth-seeking": (0, 191, 255)}
        base_color = base_colors.get(connections[-1].lower(), (200, 200, 200))
        for i, node in enumerate(connections):
            factor = 1.0 - (i / len(connections)) * 0.5
            r = int(base_color[0] * factor + random.randint(-20, 20))
            g = int(base_color[1] * factor + random.randint(-20, 20))
            b = int(base_color[2] * factor + random.randint(-20, 20))
            palette.append((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
        return palette if palette else [(200, 200, 200)]

    def visualize_concept(self, concept: str, attention: dict | None = None) -> str:
        # Check all textbooks for the concept
        all_shapes = {}
        if self._load_textbook("geometry_primitives"):
            all_shapes.update(self.textbooks["geometry_primitives"])
        if self._load_textbook("complex_shapes"):
            all_shapes.update(self.textbooks["complex_shapes"])

        if concept in all_shapes:
            return self._visualize_shape(all_shapes[concept], all_shapes, attention=attention)

        # Fallback to abstract visualization
        return self._visualize_abstract(concept)

    def visualize_echo(self, echo: dict) -> str:
        """
        Generates an abstract visualization based on a dictionary of activated concepts (the "echo").
        """
        if not echo:
            return self._visualize_abstract("emptiness")

        # The main concept is the one with the highest energy
        main_concept = max(echo, key=echo.get)

        canvas = Canvas()
        palette = self._get_color_palette(main_concept)
        
        # Use the echo to influence the generation
        all_concepts = list(echo.keys())
        total_energy = sum(echo.values())
        
        num_voxels = int(20 + 30 * (total_energy / len(echo))) # More energy = more voxels
        
        for _ in range(num_voxels):
            # Pick a concept from the echo, weighted by energy
            concept_to_draw = random.choices(all_concepts, weights=echo.values(), k=1)[0]
            
            # Use concept to seed position and color
            seed = sum(ord(c) for c in concept_to_draw)
            random.seed(seed)
            
            x, y, z = random.randint(-5, 5), random.randint(-5, 5), random.randint(-3, 6)
            color = random.choice(palette)

            # Connect to other voxels to create a structure
            if random.random() < 0.8 and canvas.voxels:
                px, py, pz, _ = random.choice(canvas.voxels)
                x, y, z = px + random.choice([-1, 0, 1]), py + random.choice([-1, 0, 1]), pz + random.choice([-1, 0, 1])

            canvas.add_voxel(x, y, z, color)

        timestamp = datetime.now().timestamp()
        output_filename = f"echo_{main_concept.replace(' ', '_')}_{timestamp}.png"
        output_path = os.path.join(self.output_dir, output_filename)
        canvas.render(output_path)
        if self.telemetry:
            try:
                self.telemetry.emit('render_done', {
                    'kind': 'shape',
                    'name': shape_data.get('name', 'unknown'),
                    'output_path': output_path,
                    'salience_gain': base_transform.get('salience_gain', 1.0),
                    'detail_gain': base_transform.get('detail_gain', 1.0),
                })
            except Exception:
                pass
        return output_path

    def _visualize_abstract(self, concept: str) -> str:
        canvas = Canvas()
        palette = self._get_color_palette(concept)
        # ... (rest of the abstract visualization logic is unchanged)
        seed = sum(ord(c) for c in concept); random.seed(seed)
        for _ in range(random.randint(15, 40)):
            x, y, z = random.randint(-4, 4), random.randint(-4, 4), random.randint(-2, 5)
            color = random.choice(palette)
            if random.random() < 0.7 and canvas.voxels:
                px, py, pz, _ = random.choice(canvas.voxels)
                x, y, z = px + random.choice([-1, 0, 1]), py + random.choice([-1, 0, 1]), pz + random.choice([-1, 0, 1])
            canvas.add_voxel(x, y, z, color)
        timestamp = datetime.now().timestamp()
        output_filename = f"abstract_{concept.replace(' ', '_')}_{timestamp}.png"
        output_path = os.path.join(self.output_dir, output_filename)
        canvas.render(output_path)
        if self.telemetry:
            try:
                self.telemetry.emit('render_done', {
                    'kind': 'echo',
                    'main_concept': main_concept,
                    'output_path': output_path,
                    'total_energy': float(total_energy),
                    'num_concepts': len(echo),
                })
            except Exception:
                pass
        return output_path

    def _visualize_shape(self, shape_data: dict, shape_library: dict, transform: dict = None, attention: dict | None = None) -> str:
        canvas = Canvas()
        # Compute attention-based emphasis (no geometric scaling)
        focus_gain = 1.0   # controls detail density only
        salience_gain = 1.0  # controls brightness only
        if attention and isinstance(attention, dict):
            try:
                total = sum(float(v) for v in attention.values()) or 1.0
                peak = max(float(v) for v in attention.values())
                focus = max(0.0, min(1.0, peak / total))
                # Map focus to perceptual gains with conservative bounds
                focus_gain = 0.9 + 0.6 * focus       # 0.9 ~ 1.5 (detail)
                salience_gain = 0.95 + 0.7 * focus   # 0.95 ~ 1.65 (brightness)
            except Exception:
                pass

        base_transform = transform or {}
        base_transform = {
            'offset': base_transform.get('offset', {'x': 0, 'y': 0, 'z': 0}),
            'size': base_transform.get('size', 1),  # geometry remains unchanged by attention
            'salience_gain': salience_gain,
            'detail_gain': focus_gain,
        }

        self._render_shape_recursive(canvas, shape_data, shape_library, base_transform)

        timestamp = datetime.now().timestamp()
        output_filename = f"shape_{shape_data['name']}_{timestamp}.png"
        output_path = os.path.join(self.output_dir, output_filename)
        canvas.render(output_path)
        if self.telemetry:
            try:
                self.telemetry.emit('render_done', {
                    'kind': 'abstract',
                    'concept': concept,
                    'output_path': output_path,
                })
            except Exception:
                pass
        return output_path

    def _render_shape_recursive(self, canvas: Canvas, shape_data: dict, shape_library: dict, transform: dict = None):
        """Recursively renders a shape, applying transformations."""
        transform = transform or {}
        rep = shape_data['representation']
        # Apply attention salience (brightness) if provided via transform
        base_color = (180, 180, 255)
        cboost = float(transform.get('salience_gain', 1.0))
        color = (
            int(min(255, base_color[0] * cboost)),
            int(min(255, base_color[1] * cboost)),
            int(min(255, base_color[2] * cboost)),
        )

        if rep['type'] == 'composite':
            for component in rep['components']:
                comp_shape_data = shape_library.get(component['shape'])
                if comp_shape_data:
                    # Apply parent transform to component transform
                    new_transform = self._apply_transform(transform, component.get('transform', {}))
                    self._render_shape_recursive(canvas, comp_shape_data, shape_library, new_transform)

        else: # It's a primitive
            self._render_primitive(canvas, rep, color, transform)

    def _render_primitive(self, canvas: Canvas, representation: dict, color: tuple, transform: dict):
        """Renders a single primitive with a given transformation."""
        # This is a simplified transformation application.
        # A real implementation would use matrix multiplication for rotations.
        offset = transform.get('offset', {'x': 0, 'y': 0, 'z': 0})
        size_mult = transform.get('size', 1)
        detail_gain = float(transform.get('detail_gain', 1.0))

        if representation['type'] == 'voxel':
            # Attention influences detail density, not geometry
            # Keep probability increases with detail_gain
            keep_prob = min(1.0, 0.6 + 0.4 * max(0.0, detail_gain))
            for coord in representation['coordinates']:
                if random.random() <= keep_prob:
                    x = (coord['x'] * size_mult) + offset['x']
                    y = (coord['y'] * size_mult) + offset['y']
                    z = (coord['z'] * size_mult) + offset['z']
                    canvas.add_voxel(x, y, z, color)

        elif representation['type'] == 'grid':
            x_range = representation['x_range']
            y_range = representation['y_range']
            z_base = representation['z']

            # Apply size multiplier to the range (from shape, not attention)
            scaled_x_range = range(int(x_range[0] * size_mult), int(x_range[1] * size_mult) + 1)
            scaled_y_range = range(int(y_range[0] * size_mult), int(y_range[1] * size_mult) + 1)
            # Attention controls sampling stride: higher detail => stride 1
            stride = 1 if detail_gain >= 1.2 else 2

            for x in scaled_x_range[::stride]:
                for y in scaled_y_range[::stride]:
                    z = (z_base * size_mult) + offset['z']
                    canvas.add_voxel(x + offset['x'], y + offset['y'], z, color)

    def _apply_transform(self, parent_transform, child_transform):
        # Simplified transform combination (offsets are additive)
        new_offset = parent_transform.get('offset', {'x': 0, 'y': 0, 'z': 0}).copy()
        child_offset = child_transform.get('offset', {'x': 0, 'y': 0, 'z': 0})
        new_offset['x'] += child_offset.get('x', 0)
        new_offset['y'] += child_offset.get('y', 0)
        new_offset['z'] += child_offset.get('z', 0)
        return {
            "offset": new_offset,
            "size": parent_transform.get("size", 1) * child_transform.get("size", 1),
            # propagate attention descriptors without changing geometry
            "salience_gain": parent_transform.get("salience_gain", 1.0),
            "detail_gain": parent_transform.get("detail_gain", 1.0),
        }

    def render_storybook_frame(self, frame_data: dict, lesson_name: str) -> str | None:
        """
        Renders a single frame of a storybook using an external image generation API.
        Acts as a 'curator' by requesting the image and returning its path.
        """
        if not all(k in frame_data for k in ['frame_id', 'description', 'style_prompt']):
            print("Warning: Invalid frame_data provided to render_storybook_frame.")
            return None

        try:
            # Create a detailed prompt for the image generation API
            prompt = (
                f"Create a visually appealing and clear illustration for a children's storybook. "
                f"The scene should depict: \"{frame_data['description']}\". "
                f"The artistic style should be: \"{frame_data['style_prompt']}\"."
            )

            # Define a unique path for the generated image
            timestamp = int(datetime.now().timestamp())
            output_filename = f"{lesson_name}_{frame_data['frame_id']:02d}_{timestamp}.png"
            output_path = os.path.join(self.storybook_dir, output_filename)

            # Request the image generation
            success = generate_image_from_text(prompt, output_path)

            if success:
                if self.telemetry:
                    try:
                        self.telemetry.emit('storybook_frame_rendered', {
                            'lesson': lesson_name,
                            'frame_id': frame_data['frame_id'],
                            'output_path': output_path
                        })
                    except Exception:
                        pass
                return output_path
            else:
                print(f"Warning: Failed to generate image for frame {frame_data['frame_id']} of lesson {lesson_name}.")
                return None

        except Exception as e:
            print(f"An unexpected error occurred in render_storybook_frame: {e}")
            return None

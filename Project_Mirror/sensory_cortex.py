import os
import random
import json
import math
from datetime import datetime
from tools.canvas_tool import Canvas
from Project_Sophia.emotional_cortex import Mood

class SensoryCortex:
    def __init__(self):
        self.output_dir = "data/generated_images"
        os.makedirs(self.output_dir, exist_ok=True)
        self.textbooks = {}

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
        """Returns a deterministic, concept-based but simplified color palette."""
        seed = sum(ord(c) for c in concept)
        random.seed(seed)

        base_color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        palette = []
        for _ in range(5):
            r = int(base_color[0] + random.randint(-50, 50))
            g = int(base_color[1] + random.randint(-50, 50))
            b = int(base_color[2] + random.randint(-50, 50))
            palette.append((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
        return palette

    def visualize_concept(self, concept: str, mood: Mood = None) -> str:
        # Check all textbooks for the concept
        all_shapes = {}
        if self._load_textbook("geometry_primitives"):
            all_shapes.update(self.textbooks["geometry_primitives"])
        if self._load_textbook("complex_shapes"):
            all_shapes.update(self.textbooks["complex_shapes"])

        if concept in all_shapes:
            return self._visualize_shape(all_shapes[concept], all_shapes, mood=mood)

        # Fallback to abstract visualization
        return self._visualize_abstract(concept, mood=mood)

    def _get_color_palette_from_mood(self, mood: Mood) -> list:
        """Generates a color palette based on the current mood."""
        if not mood or mood.primary_mood == "neutral":
            return [(180, 180, 180), (210, 210, 210), (150, 150, 150)]

        # Mood to color mapping
        mood_colors = {
            "sense_of_accomplishment": (255, 215, 0), # Gold
            "curiosity": (0, 191, 255),               # Deep Sky Blue
            "connectedness": (255, 105, 180),         # Hot Pink (Love)
            "warmth": (255, 165, 0),                  # Orange
            "focused": (70, 130, 180),                # Steel Blue
            "internal_conflict": (138, 43, 226),      # Blue Violet
        }
        base_color = mood_colors.get(mood.primary_mood, (200, 200, 200)) # Default grey

        palette = []
        for i in range(5):
            # Intensity affects color variation. Higher intensity = more vibrant and varied.
            variation = int(80 * mood.intensity)
            factor = 1.0 - (i / 5) * (0.5 * mood.intensity)
            r = int(base_color[0] * factor + random.randint(-variation, variation))
            g = int(base_color[1] * factor + random.randint(-variation, variation))
            b = int(base_color[2] * factor + random.randint(-variation, variation))
            palette.append((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
        return palette

    def _visualize_abstract(self, concept: str, mood: Mood = None) -> str:
        canvas = Canvas()
        palette = self._get_color_palette_from_mood(mood) if mood else self._get_color_palette(concept)

        seed = sum(ord(c) for c in concept); random.seed(seed)

        # Mood intensity affects the complexity of the art
        num_voxels = int(15 + 25 * (mood.intensity if mood else 0.5))

        for _ in range(random.randint(num_voxels, num_voxels + 10)):
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
        return output_path

    def _visualize_shape(self, shape_data: dict, shape_library: dict, transform: dict = None, mood: Mood = None) -> str:
        canvas = Canvas()
        palette = self._get_color_palette_from_mood(mood) if mood else [(180, 180, 255)]
        color = random.choice(palette)

        self._render_shape_recursive(canvas, shape_data, shape_library, transform, color)

        timestamp = datetime.now().timestamp()
        output_filename = f"shape_{shape_data['name']}_{timestamp}.png"
        output_path = os.path.join(self.output_dir, output_filename)
        canvas.render(output_path)
        return output_path

    def _render_shape_recursive(self, canvas: Canvas, shape_data: dict, shape_library: dict, transform: dict = None, color: tuple = (180, 180, 255)):
        """Recursively renders a shape, applying transformations."""
        transform = transform or {}
        rep = shape_data['representation']

        if rep['type'] == 'composite':
            for component in rep['components']:
                comp_shape_data = shape_library.get(component['shape'])
                if comp_shape_data:
                    # Apply parent transform to component transform
                    new_transform = self._apply_transform(transform, component.get('transform', {}))
                    self._render_shape_recursive(canvas, comp_shape_data, shape_library, new_transform, color)

        else: # It's a primitive
            self._render_primitive(canvas, rep, color, transform)

    def _render_primitive(self, canvas: Canvas, representation: dict, color: tuple, transform: dict):
        """Renders a single primitive with a given transformation."""
        # This is a simplified transformation application.
        # A real implementation would use matrix multiplication for rotations.
        offset = transform.get('offset', {'x': 0, 'y': 0, 'z': 0})
        size_mult = transform.get('size', 1)

        if representation['type'] == 'voxel':
            for coord in representation['coordinates']:
                x = (coord['x'] * size_mult) + offset['x']
                y = (coord['y'] * size_mult) + offset['y']
                z = (coord['z'] * size_mult) + offset['z']
                canvas.add_voxel(x, y, z, color)

        elif representation['type'] == 'grid':
            x_range = representation['x_range']
            y_range = representation['y_range']
            z_base = representation['z']

            # Apply size multiplier to the range
            scaled_x_range = range(int(x_range[0] * size_mult), int(x_range[1] * size_mult) + 1)
            scaled_y_range = range(int(y_range[0] * size_mult), int(y_range[1] * size_mult) + 1)

            for x in scaled_x_range:
                for y in scaled_y_range:
                    z = (z_base * size_mult) + offset['z']
                    canvas.add_voxel(x + offset['x'], y + offset['y'], z, color)

    def _apply_transform(self, parent_transform, child_transform):
        # Simplified transform combination (offsets are additive)
        new_offset = parent_transform.get('offset', {'x': 0, 'y': 0, 'z': 0}).copy()
        child_offset = child_transform.get('offset', {'x': 0, 'y': 0, 'z': 0})
        new_offset['x'] += child_offset.get('x', 0)
        new_offset['y'] += child_offset.get('y', 0)
        new_offset['z'] += child_offset.get('z', 0)
        return {"offset": new_offset, "size": child_transform.get("size", 1)}

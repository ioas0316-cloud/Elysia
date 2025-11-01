from PIL import Image, ImageDraw

class Canvas:
    """
    A 3D voxel canvas that can be rendered as a 2D PNG image.
    This tool is the foundation of Elysia's sensory expression.
    """

    def __init__(self, width: int = 256, height: int = 256, bg_color=(10, 10, 25)):
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.image = Image.new("RGB", (self.width, self.height), self.bg_color)
        self.draw = ImageDraw.Draw(self.image)
        self.voxels = []  # List to store (x, y, z, color) tuples
        self.z_buffer = [[float('-inf')] * width for _ in range(height)]

    def _project(self, x, y, z):
        """Projects a 3D point to 2D using isometric projection."""
        iso_x = (x - y) * 0.707 + self.width / 2
        iso_y = (x + y) * 0.408 - z * 0.816 + self.height / 2
        return int(iso_x), int(iso_y)

    def add_voxel(self, x: int, y: int, z: int, color: tuple):
        """Adds a voxel to the canvas."""
        self.voxels.append((x, y, z, color))

    def render(self, output_path: str, voxel_size: int = 10, shadow_offset: int = 2):
        """
        Renders the voxels to a 2D PNG image with shadows and depth.
        """
        # Sort voxels from back to front to handle occlusion correctly
        self.voxels.sort(key=lambda v: v[0] + v[1] + v[2], reverse=True)

        for x, y, z, color in self.voxels:
            px, py = self._project(x, y, z)

            # --- Shadow Pass ---
            shadow_color = (
                max(0, self.bg_color[0] - 10),
                max(0, self.bg_color[1] - 10),
                max(0, self.bg_color[2] - 15)
            )
            shadow_x = px + shadow_offset
            shadow_y = py + shadow_offset

            # Draw shadow if it's in front of what's already there
            if 0 <= shadow_x < self.width and 0 <= shadow_y < self.height:
                # A simple approximation for shadow depth
                shadow_z = z - 1
                if shadow_z > self.z_buffer[shadow_y][shadow_x]:
                    self.draw.rectangle(
                        [shadow_x, shadow_y, shadow_x + voxel_size, shadow_y + voxel_size],
                        fill=shadow_color
                    )
                    # We don't update z-buffer for shadows to allow objects to be drawn over them

            # --- Voxel Pass ---
            if 0 <= px < self.width and 0 <= py < self.height:
                if z > self.z_buffer[py][px]:
                    self.z_buffer[py][px] = z

                    # Main voxel face
                    self.draw.rectangle(
                        [px, py, px + voxel_size, py + voxel_size],
                        fill=color,
                        outline=(255, 255, 255, 30) # Subtle outline
                    )

                    # Highlight for a 3D effect
                    highlight_color = (
                        min(255, color[0] + 40),
                        min(255, color[1] + 40),
                        min(255, color[2] + 40)
                    )
                    self.draw.line([(px, py), (px + voxel_size, py)], fill=highlight_color, width=1)
                    self.draw.line([(px, py), (px, py + voxel_size)], fill=highlight_color, width=1)


        self.image.save(output_path, "PNG")

if __name__ == '__main__':
    # Example usage: Create a simple structure and render it
    canvas = Canvas()

    # Create a simple 3x3x3 cube
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                # Simple color based on position
                r = (i + 1) * 80
                g = (j + 1) * 80
                b = (k + 1) * 80
                canvas.add_voxel(i, j, k, (r, g, b))

    canvas.render("test_render.png")
    print("Test render saved to test_render.png")

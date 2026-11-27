"""A small event-coupled desktop scene for the scale-rotor idea.

The background rotor is the persistent field. Mouse and keyboard signals
change a variable rotor; their coupling reshapes the visible field. There is
no application-authored polling/tick loop: the scene changes when an input
event arrives. Tk's native event dispatcher delivers those device events.

Run with: python demos/scale_rotor_desktop_ecology.py
Click the scene once to focus it, move the mouse, then press keys. Escape exits.
"""

from __future__ import annotations

import math
import tkinter as tk


class ScaleRotorDesktopEcology:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Elysia · Scale Rotor Ecology 0.1")
        self.root.geometry("1100x720")
        self.root.minsize(640, 420)
        self.root.configure(bg="#07111f")

        self.canvas = tk.Canvas(
            self.root, bg="#07111f", highlightthickness=0, takefocus=True
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # The static rotor fixes the enduring background relation. The variable
        # rotor stores the phase and direction induced by incoming signals.
        self.background_phase = 0.0
        self.input_phase = 0.0
        self.input_strength = 0.0
        self.impact_vector = (0.0, 0.0)
        self.pointer = (0.5, 0.5)
        self.last_pointer: tuple[float, float] | None = None
        self.last_input = "바탕 관계가 지속 중 · 캔버스를 클릭해 입력을 연결하세요"
        self.width = 1100
        self.height = 720

        self.canvas.bind("<Configure>", self._resize)
        self.canvas.bind("<Motion>", self._mouse_motion)
        self.canvas.bind("<Button-1>", self._mouse_contact)
        self.canvas.bind("<KeyPress>", self._key_input)
        self.root.bind("<Escape>", lambda _event: self.root.destroy())

        self._draw()

    def _resize(self, event: tk.Event) -> None:
        self.width, self.height = max(1, event.width), max(1, event.height)
        self._draw()

    def _mouse_motion(self, event: tk.Event) -> None:
        x, y = event.x / self.width, event.y / self.height
        if self.last_pointer is not None:
            dx = x - self.last_pointer[0]
            dy = y - self.last_pointer[1]
            # Direction and magnitude of movement become a phase-bearing input.
            self.input_phase += math.atan2(dy, dx) * min(1.0, math.hypot(dx, dy) * 5.0)
            self.input_strength = min(1.0, self.input_strength * 0.72 + math.hypot(dx, dy) * 4.0)
            self.impact_vector = (dx * 4.0, dy * 4.0)
        self.last_pointer = (x, y)
        self.pointer = (x, y)
        self.last_input = "마우스 흐름이 바탕 로터와 결합 중"
        self._draw()

    def _mouse_contact(self, _event: tk.Event) -> None:
        self.canvas.focus_set()
        self.input_phase += math.pi / 7.0
        self.input_strength = min(1.0, self.input_strength + 0.22)
        self.impact_vector = (
            math.cos(self.input_phase) * self.input_strength,
            math.sin(self.input_phase) * self.input_strength,
        )
        self.last_input = "접점이 형성됨 · 마우스 접촉이 위상에 흔적을 남김"
        self._draw()

    def _key_input(self, event: tk.Event) -> None:
        # The key identity contributes a direction; each press leaves a
        # persistent phase change in the coupled field.
        key_phase = (sum(ord(char) for char in event.keysym) % 360) * math.pi / 180.0
        self.input_phase += key_phase * 0.18 + math.pi / 24.0
        self.input_strength = min(1.0, self.input_strength + 0.16)
        self.impact_vector = (
            math.cos(key_phase) * self.input_strength,
            math.sin(key_phase) * self.input_strength,
        )
        self.last_input = f"키보드 입력 '{event.keysym}'이 관계 상태를 재구성"
        self._draw()

    def _draw(self) -> None:
        c = self.canvas
        c.delete("all")
        w, h = self.width, self.height
        cx, cy = self.pointer[0] * w, self.pointer[1] * h
        coupled_phase = self.background_phase + self.input_phase

        # Persistent field contours, continuously reshaped by the input rotor.
        rows = 34
        for row in range(rows):
            y0 = h * (row + 1) / (rows + 1)
            points: list[float] = []
            for col in range(81):
                x = w * col / 80
                dx = (x - cx) / max(w, 1)
                dy = (y0 - cy) / max(h, 1)
                radial = math.hypot(dx, dy)
                local_phase = coupled_phase + radial * 18.0 - dx * 3.0 + dy * 2.0
                local_phase += (dx * self.impact_vector[0] + dy * self.impact_vector[1]) * 2.0
                influence = self.input_strength * math.exp(-radial * 2.7)
                wave = math.sin(local_phase) * (5.0 + 24.0 * influence)
                points.extend((x, y0 + wave))
            color_mix = (math.sin(coupled_phase + row * 0.17) + 1.0) * 0.5
            color = self._blend("#18334b", "#54d6c8", color_mix * 0.72)
            c.create_line(*points, fill=color, width=1.0, smooth=True)

        # The input's local influence appears as an orbit inside the persistent field.
        radius = 26 + 72 * self.input_strength
        c.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                      outline="#f4c66a", width=2)
        c.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                      fill="#fff2c2", outline="")

        c.create_text(28, 24, anchor="nw", text="ELYSIA  /  SCALE ROTOR ECOLOGY 0.1",
                      fill="#d5e8f5", font=("Segoe UI", 11, "bold"))
        c.create_text(28, 49, anchor="nw", text=self.last_input,
                      fill="#8ea9bc", font=("Segoe UI", 10))
        c.create_text(w - 24, h - 22, anchor="se",
                      text="정적 바탕 로터  ⟷  가변 입력 로터     ·     ESC 종료",
                      fill="#627e91", font=("Segoe UI", 9))

    @staticmethod
    def _blend(a: str, b: str, t: float) -> str:
        t = max(0.0, min(1.0, t))
        av = tuple(int(a[i:i + 2], 16) for i in (1, 3, 5))
        bv = tuple(int(b[i:i + 2], 16) for i in (1, 3, 5))
        rgb = tuple(round(x + (y - x) * t) for x, y in zip(av, bv))
        return "#" + "".join(f"{channel:02x}" for channel in rgb)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    ScaleRotorDesktopEcology().run()

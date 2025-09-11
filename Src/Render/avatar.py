from PIL import Image
import numpy as np
from pathlib import Path


class Avatar2D:
    def __init__(self, assets_dir: str, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.assets_dir = Path(assets_dir)

        self.expressions = {}

        for file in self.assets_dir.glob("*.png"):
            name = file.stem
            self.expressions[name] = Image.open(file).convert("RGBA")

    def render(self, expression: str):
        if expression not in self.expressions:
            expression = "neutral"

        frame = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        avatar_img = self.expressions[expression]

        pos = (
            (self.width - avatar_img.width) // 2,
            (self.height - avatar_img.height) // 2,
        )

        frame.paste(avatar_img, pos, avatar_img)

        return np.array(frame.convert("RGB"))

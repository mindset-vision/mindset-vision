"""lightness contrast illusion dataset generator."""
import csv
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from mindset.drawing.grayscale_shapes import add_arrow
from mindset.utils.misc import apply_antialiasing
from mindset.generators._base import GeneratorConfig, generator, register


@dataclass
class LightnessContrastConfig(GeneratorConfig):
    """config for lightness contrast illusion dataset."""
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    steps_arrow: int = field(default=30, metadata={"min": 1, "max": 100, "step": 1, "label": "arrow step size"})
    square_color: int = field(default=200, metadata={"min": 0, "max": 255, "step": 1, "label": "square grayscale color"})
    steps_bg_color: int = field(default=20, metadata={"min": 1, "max": 128, "step": 1, "label": "background color step"})
    output_folder: str = field(default="data/visual_illusions/lightness_contrast", metadata={"label": "output folder"})


@register("lightness_contrast", "visual_illusions")
@generator(LightnessContrastConfig)
def generate_all(config: LightnessContrastConfig):
    """generate lightness contrast illusion dataset."""
    output_folder = Path(config.output_folder)
    (output_folder / "all_images").mkdir(parents=True, exist_ok=True)

    canvas_size = config.canvas_size
    coordinates = [
        (x, y)
        for x in range(config.steps_arrow, canvas_size[0], config.steps_arrow)
        for y in range(config.steps_arrow, canvas_size[1], config.steps_arrow)
    ]

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Target Pixel Color", "Target Pixel Location", "Background Color", "Rectangle Color"])

        for coordinate in tqdm(coordinates, colour="green"):
            grayscale_background_all = np.arange(0, 255, config.steps_bg_color)
            for background_c in grayscale_background_all:
                image = Image.new("RGBA", canvas_size, (background_c,) * 4)
                draw = ImageDraw.Draw(image)
                width, height = image.size
                size_rect = 80
                draw.rectangle(
                    (width // 2 - size_rect // 2, height // 2 - size_rect // 2,
                     width // 2 + size_rect // 2, height // 2 + size_rect // 2),
                    fill=(config.square_color,) * 3,
                )
                image = add_arrow(image, coordinate)
                image_path = Path("all_images") / f"{uuid.uuid4().hex[:8]}.png"
                image = apply_antialiasing(image) if config.antialiasing else image
                image = image.convert("L")
                pixel_color = image.getpixel(coordinate)
                image.save(output_folder / image_path)
                writer.writerow([str(image_path), pixel_color, coordinate, background_c, config.square_color])

    return str(output_folder)

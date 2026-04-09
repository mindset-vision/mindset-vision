"""weber law dataset generator."""

import csv
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from PIL import ImageDraw
from tqdm import tqdm

from mindset.drawing.base import DrawStimuli
from mindset.generators._base import GeneratorConfig, generator, register


class DrawWeberLength(DrawStimuli):
    """draws a horizontal line with configurable length, width, and luminance."""

    def gen_stim(self, length, width, lum):
        """generate a single line stimulus image."""
        img = self.create_canvas()
        x0, y0 = self.canvas_size[0] / 2 - (length / 2), self.canvas_size[1] / 2
        x1, y1 = self.canvas_size[0] / 2 + (length / 2), self.canvas_size[1] / 2
        bbox = [(x0, y0), (x1, y1)]
        drawing = ImageDraw.Draw(img)
        drawing.line(bbox, width=width, fill=(lum, lum, lum))
        return img


@dataclass
class WeberLawConfig(GeneratorConfig):
    """config for weber law dataset."""

    num_samples_per_condition: int = field(
        default=50,
        metadata={"min": 1, "max": 5000, "step": 1, "label": "samples per condition"},
    )
    max_line_length: int = field(
        default=50,
        metadata={"min": 5, "max": 200, "step": 1, "label": "max line length (px)"},
    )
    min_line_length: int = field(
        default=5,
        metadata={"min": 1, "max": 200, "step": 1, "label": "min line length (px)"},
    )
    interval_line_length: int = field(
        default=1,
        metadata={"min": 1, "max": 50, "step": 1, "label": "line length interval"},
    )
    min_grayscale: int = field(
        default=50, metadata={"min": 0, "max": 255, "step": 1, "label": "min grayscale"}
    )
    max_grayscale: int = field(
        default=255,
        metadata={"min": 0, "max": 255, "step": 1, "label": "max grayscale"},
    )
    interval_grayscale: int = field(
        default=20,
        metadata={"min": 1, "max": 100, "step": 1, "label": "grayscale interval"},
    )
    width: int = field(
        default=2, metadata={"min": 1, "max": 20, "step": 1, "label": "line width (px)"}
    )
    output_folder: str = field(
        default="data/low_mid_level_vision/weber_law",
        metadata={"label": "output folder"},
    )


@register("weber_law", "low_mid_vision")
@generator(WeberLawConfig)
def generate_all(config: WeberLawConfig):
    """generate weber law dataset."""
    output_folder = Path(config.output_folder)
    (output_folder / "all_images").mkdir(parents=True, exist_ok=True)

    ds = DrawWeberLength(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
    )

    lengths_conditions = range(
        config.min_line_length, config.max_line_length, config.interval_line_length
    )
    grayscale_conditions = range(
        config.min_grayscale, config.max_grayscale, config.interval_grayscale
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            ["Path", "BackgroundColor", "Length", "Width", "Luminance", "IterNum"]
        )

        for ln in tqdm(lengths_conditions):
            for gr in grayscale_conditions:
                for n in range(config.num_samples_per_condition):
                    w = config.width
                    unique_hex = uuid.uuid4().hex[:8]
                    img_path = f"{ln}_{gr}_{unique_hex}.png"
                    img = ds.gen_stim(ln, w, gr)
                    img.save(output_folder / "all_images" / img_path)
                    writer.writerow([img_path, ds.background, ln, w, gr, n])

    return str(output_folder)

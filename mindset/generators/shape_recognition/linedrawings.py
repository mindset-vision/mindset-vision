"""linedrawings dataset generator."""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from PIL import Image, ImageOps
from tqdm import tqdm

from mindset.drawing.base import (
    DrawStimuli,
    paste_linedrawing_onto_canvas,
    resize_image_keep_aspect_ratio,
)
from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils import apply_antialiasing

# ---------------------------------------------------------------------------
# drawing class
# ---------------------------------------------------------------------------


class DrawLinedrawings(DrawStimuli):
    """draws simple linedrawings (white stroke on canvas)."""

    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def get_linedrawings(self, image_path):
        """load and paste a linedrawing onto a canvas."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = resize_image_keep_aspect_ratio(img, self.obj_longest_side)
        img = ImageOps.invert(Image.fromarray(img).convert("L"))

        canvas = paste_linedrawing_onto_canvas(img, self.create_canvas(), self.fill)

        return apply_antialiasing(canvas) if self.antialiasing else canvas


# ---------------------------------------------------------------------------
# generator config and entry point
# ---------------------------------------------------------------------------


@dataclass
class LinedrawingsConfig(GeneratorConfig):
    """config for linedrawings dataset."""

    object_longest_side: int = field(
        default=200,
        metadata={
            "min": 50,
            "max": 500,
            "step": 10,
            "label": "object longest side (px)",
        },
    )
    linedrawing_input_folder: str = field(
        default="mindset/assets/baker_2018_linedrawings/cropped/",
        metadata={"label": "input folder with line drawings"},
    )
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    output_folder: str = field(
        default="data/shape_and_object_recognition/linedrawings",
        metadata={"label": "output folder"},
    )


@register("linedrawings", "shape_recognition")
@generator(LinedrawingsConfig)
def generate_all(config: LinedrawingsConfig):
    """generate linedrawings dataset from source images."""
    output_folder = Path(config.output_folder)
    linedrawing_input_folder = Path(config.linedrawing_input_folder)

    all_categories = [p.stem for p in linedrawing_input_folder.glob("*") if p.is_dir()]
    for cat in all_categories:
        (output_folder / cat).mkdir(exist_ok=True, parents=True)

    ds = DrawLinedrawings(
        background=config.background_color,
        canvas_size=config.canvas_size,
        obj_longest_side=config.object_longest_side,
    )

    all_images = sorted(linedrawing_input_folder.rglob("*.jpg")) + sorted(
        linedrawing_input_folder.rglob("*.png")
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Class", "BackgroundColor", "IterNum"])

        for i, img_path in enumerate(tqdm(all_images)):
            img = ds.get_linedrawings(img_path)
            category = img_path.parent.stem
            path = Path(category) / f"{i}.png"
            img.save(output_folder / path)
            writer.writerow([path, category, ds.background, i])

    return str(output_folder)

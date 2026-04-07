"""dotted linedrawings dataset generator."""
import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils.drawing_utils import (
    DrawStimuli,
    paste_linedrawing_onto_canvas,
    resize_image_keep_aspect_ratio,
)
from mindset.utils.misc import apply_antialiasing


class DrawDottedImage(DrawStimuli):
    """draws dotted versions of linedrawing images."""

    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def dotted_image(self, image_path, dot_distance, dot_size):
        """convert a linedrawing to a dotted contour image."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = resize_image_keep_aspect_ratio(img, self.obj_longest_side)

        _, binary_img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
        contours, b = cv2.findContours(
            binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        dotted_img = np.ones_like(img) * 255

        def draw_dot(image, x, y, size, color):
            half_size = size // 2
            cv2.rectangle(
                image,
                (x - half_size, y - half_size),
                (x + half_size, y + half_size),
                color,
                -1,
            )

        for contour in contours:
            for i, point in enumerate(contour):
                if i % dot_distance == 0:
                    x, y = point[0]
                    draw_dot(dotted_img, x, y, dot_size, color=0)

        dotted_img = Image.fromarray(dotted_img)
        dotted_img = ImageOps.invert(dotted_img.convert("L"))

        canvas = paste_linedrawing_onto_canvas(
            dotted_img, self.create_canvas(), self.fill
        )

        return apply_antialiasing(canvas) if self.antialiasing else canvas


@dataclass
class DottedLinedrawingsConfig(GeneratorConfig):
    """config for dotted linedrawings dataset."""
    object_longest_side: int = field(default=200, metadata={"min": 50, "max": 500, "step": 10, "label": "object longest side (px)"})
    linedrawing_input_folder: str = field(default="mindset/assets/baker_2018_linedrawings/cropped/", metadata={"label": "input folder with line drawings"})
    dot_distance: int = field(default=5, metadata={"min": 1, "max": 50, "step": 1, "label": "distance between dots"})
    dot_size: int = field(default=1, metadata={"min": 1, "max": 20, "step": 1, "label": "dot size"})
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    output_folder: str = field(default="data/shape_and_object_recognition/dotted_linedrawings", metadata={"label": "output folder"})


@register("dotted_linedrawings", "shape_recognition")
@generator(DottedLinedrawingsConfig)
def generate_all(config: DottedLinedrawingsConfig):
    """generate dotted linedrawings dataset."""
    output_folder = Path(config.output_folder)
    linedrawing_input_folder = Path(config.linedrawing_input_folder)

    all_categories = [p.stem for p in linedrawing_input_folder.glob("*") if p.is_dir()]
    for cat in all_categories:
        (output_folder / cat).mkdir(exist_ok=True, parents=True)

    ds = DrawDottedImage(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        obj_longest_side=config.object_longest_side,
    )

    image_files = sorted(linedrawing_input_folder.rglob("*.jpg")) + sorted(linedrawing_input_folder.rglob("*.png"))

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Class", "BackgroundColor", "DotDistance", "DotSize", "IterNum"])

        for n, img_path in enumerate(tqdm(image_files)):
            class_name = img_path.parent.stem
            name_sample = img_path.stem
            img = ds.dotted_image(img_path, dot_distance=config.dot_distance, dot_size=config.dot_size)
            path = Path(class_name) / f"{name_sample}.png"
            img.save(output_folder / path)
            writer.writerow([path, class_name, ds.background, config.dot_distance, config.dot_size, n])

    return str(output_folder)

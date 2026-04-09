"""silhouettes dataset generator."""

import csv
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm.auto import tqdm

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
    """draws silhouettes from linedrawings or silhouette inputs."""

    def __init__(self, obj_longest_side, input_image_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side
        self.input_image_type = input_image_type

    def get_linedrawings(self, image_path):
        """load image and convert to silhouette on canvas."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = resize_image_keep_aspect_ratio(img, self.obj_longest_side)
        _, binary_img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
        if self.input_image_type == "linedrawings":
            contours, _ = cv2.findContours(
                binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            mask = np.ones_like(img) * 255

            cv2.drawContours(mask, contours, -1, (0), thickness=cv2.FILLED)

            [
                cv2.drawContours(mask, [c], -1, (0), thickness=cv2.FILLED)
                for c in contours
            ]
        else:
            mask = cv2.bitwise_not(binary_img)
        mask = ImageOps.invert(Image.fromarray(mask).convert("L"))

        canvas = paste_linedrawing_onto_canvas(mask, self.create_canvas(), self.fill)

        return apply_antialiasing(canvas) if self.antialiasing else canvas


# ---------------------------------------------------------------------------
# generator config and entry point
# ---------------------------------------------------------------------------


@dataclass
class SilhouettesConfig(GeneratorConfig):
    """config for silhouettes dataset."""

    object_longest_side: int = field(
        default=200,
        metadata={
            "min": 50,
            "max": 500,
            "step": 10,
            "label": "object longest side (px)",
        },
    )
    image_input_folder: str = field(
        default="mindset/assets/baker_2018_linedrawings/cropped/",
        metadata={"label": "input folder with images"},
    )
    input_image_type: str = field(
        default="linedrawings",
        metadata={
            "choices": ["linedrawings", "silhouettes"],
            "label": "input image type",
        },
    )
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    output_folder: str = field(
        default="data/shape_and_object_recognition/silhouettes",
        metadata={"label": "output folder"},
    )


@register("silhouettes", "shape_recognition")
@generator(SilhouettesConfig)
def generate_all(config: SilhouettesConfig):
    """generate silhouettes dataset from source images."""
    output_folder = Path(config.output_folder)
    image_input_folder = Path(config.image_input_folder)

    all_categories = [p.stem for p in image_input_folder.glob("*")]
    for cat in all_categories:
        (output_folder / cat).mkdir(exist_ok=True, parents=True)

    ds = DrawLinedrawings(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        obj_longest_side=config.object_longest_side,
        input_image_type=config.input_image_type,
    )

    jpg_files = list(image_input_folder.rglob("*.jpg"))
    png_files = list(image_input_folder.rglob("*.png"))
    image_files = jpg_files + png_files

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Class", "BackgroundColor", "IterNum"])

        for n, img_path in enumerate(tqdm(image_files)):
            class_name = img_path.parent.stem
            image_name = img_path.stem
            img = ds.get_linedrawings(img_path)
            unique_hex = uuid.uuid4().hex[:8]
            path = Path(class_name) / f"{image_name}_{unique_hex}.png"
            img.save(output_folder / path)
            writer.writerow([path, class_name, ds.background, n])

    return str(output_folder)

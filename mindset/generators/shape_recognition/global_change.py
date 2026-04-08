"""global change dataset generator."""
import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from mindset.drawing.base import (
    DrawStimuli,
    resize_image_keep_aspect_ratio,
)
from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils import apply_antialiasing


# ---------------------------------------------------------------------------
# drawing class
# ---------------------------------------------------------------------------

class DrawLinedrawings(DrawStimuli):
    """draws whole/fragmented/frankenstein linedrawings (baker & elder 2022 style)."""

    def __init__(self, obj_longest_side, convert_to_silhouettes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side
        self.convert_to_silhouettes = convert_to_silhouettes

    def get_linedrawings(self, image_path, type):
        """produce whole, fragmented, or frankenstein version of a linedrawing."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        img = resize_image_keep_aspect_ratio(img, self.obj_longest_side)
        _, binary_img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
        if self.convert_to_silhouettes:
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

        silhouette = Image.fromarray(mask)
        width, height = silhouette.size
        top_half = silhouette.crop((0, 0, width, height // 2))
        bottom_half = silhouette.crop((0, height // 2, width, height))

        if type in ["frankenstein", "fragmented"]:
            top_half = top_half.transpose(Image.FLIP_LEFT_RIGHT)

        top_half_np = np.array(top_half)
        bottom_half_np = np.array(bottom_half)
        if type == "frankenstein":
            top = np.min(np.where(top_half_np[-1] == 0))
            bottom = np.min(np.where(bottom_half_np[0] == 0))
        elif type == "fragmented":
            top = np.min(np.where(top_half_np[-1] == 0))
            bottom = np.max(np.where(bottom_half_np[0] == 0))
        else:  # type == "whole":
            bottom = 0
            top = 0
        top_offset = max(0, bottom - top)
        bottom_offset = max(0, top - bottom)

        new_canvas = Image.fromarray(
            np.ones(
                (
                    self.canvas_size[0],
                    max(top_half.size[0], bottom_half.size[0])
                    + max(top_offset, bottom_offset),
                )
            )
            * 255
        ).convert("RGB")

        new_canvas.paste(
            top_half,
            (
                top_offset,
                new_canvas.size[1] // 2 - top_half.size[1],
            ),
        )
        new_canvas.paste(
            bottom_half,
            (
                bottom_offset,
                new_canvas.size[1] // 2,
            ),
        )

        cs = tuple(
            np.array(self.canvas_size)
            * max(np.array(new_canvas.size) / self.canvas_size)
        )
        canvas = self.create_canvas(size=[int(i) for i in cs])
        canvas.paste(
            ImageOps.invert(new_canvas.convert("L")),
            (
                canvas.size[0] // 2 - new_canvas.size[0] // 2,
                canvas.size[1] // 2 - new_canvas.size[1] // 2,
            ),
        )
        canvas = canvas.resize(self.canvas_size)

        return apply_antialiasing(canvas) if self.antialiasing else canvas


# ---------------------------------------------------------------------------
# generator config and entry point
# ---------------------------------------------------------------------------

@dataclass
class GlobalChangeConfig(GeneratorConfig):
    """config for global change dataset."""
    object_longest_side: int = field(default=120, metadata={"min": 50, "max": 500, "step": 10, "label": "object longest side (px)"})
    image_input_folder: str = field(default="mindset/assets/baker_2018_linedrawings/cropped/", metadata={"label": "input folder with images"})
    convert_to_silhouettes: int = field(default=0, metadata={"choices": [0, 1], "label": "convert to silhouettes"})
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    output_folder: str = field(default="data/shape_and_object_recognition/global_change", metadata={"label": "output folder"})


@register("global_change", "shape_recognition")
@generator(GlobalChangeConfig)
def generate_all(config: GlobalChangeConfig):
    """generate global change dataset with whole, fragmented, and frankenstein conditions."""
    output_folder = Path(config.output_folder)
    image_input_folder = Path(config.image_input_folder)

    all_categories = [p.stem for p in image_input_folder.glob("*")]
    conditions = ["whole", "fragmented", "frankenstein"]

    for c in conditions:
        for cat in all_categories:
            (output_folder / c / cat).mkdir(exist_ok=True, parents=True)

    ds = DrawLinedrawings(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        obj_longest_side=config.object_longest_side,
        convert_to_silhouettes=config.convert_to_silhouettes,
    )

    image_files = sorted(image_input_folder.rglob("*.jpg")) + sorted(image_input_folder.rglob("*.png"))

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Class", "BackgroundColor", "IterNum"])
        for n, img_path in enumerate(tqdm(image_files)):
            for t in conditions:
                class_name = img_path.parent.stem
                image_name = img_path.stem
                img = ds.get_linedrawings(img_path, type=t)
                path = Path(t) / class_name / f"{image_name}.png"
                img.save(output_folder / path)
                writer.writerow([path, class_name, ds.background, n])

    return str(output_folder)

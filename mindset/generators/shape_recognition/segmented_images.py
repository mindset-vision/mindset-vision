"""segmented images dataset generator."""
import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from mindset.generators._base import GeneratorConfig, generator, register
from mindset.drawing.base import (
    DrawStimuli,
    paste_linedrawing_onto_canvas,
    resize_image_keep_aspect_ratio,
)
from mindset.utils.misc import apply_antialiasing


class DrawGriddedImages(DrawStimuli):
    """draws linedrawings with grid-based segment deletions."""

    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def apply_grid_mask(
        self,
        image_path,
        grid_size,
        grid_thickness=1,
        grid_shift=0,
        rotation_degrees=0,
        complement=False,
    ):
        """apply a rotated grid mask to delete segments of a linedrawing."""
        opencv_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = resize_image_keep_aspect_ratio(opencv_img, self.obj_longest_side)

        img = ImageOps.invert(Image.fromarray(img).convert("L"))

        img = np.array(
            paste_linedrawing_onto_canvas(
                img, self.create_canvas(), self.line_args["fill"]
            )
        )

        height, width, _ = img.shape

        mask = np.full((height * 2, width * 2), False)

        for i in range(grid_shift, mask.shape[0], grid_size):
            mask[i : i + grid_thickness, :] = 1

        rotated_mask = np.array(
            Image.fromarray(mask).rotate(rotation_degrees, expand=True, fillcolor=(0))
        )
        rotated_mask = rotated_mask[
            rotated_mask.shape[0] // 2
            - height // 2 : rotated_mask.shape[0] // 2
            - height // 2
            + height,
            rotated_mask.shape[1] // 2
            - width // 2 : rotated_mask.shape[1] // 2
            - width // 2
            + width,
        ]

        if complement:
            img[~rotated_mask] = self.background
        else:
            img[rotated_mask] = self.background
        img = Image.fromarray(img)
        return apply_antialiasing(img) if self.antialiasing else img


@dataclass
class SegmentedImagesConfig(GeneratorConfig):
    """config for segmented images dataset."""
    linedrawing_input_folder: str = field(default="mindset/assets/baker_2018_linedrawings/cropped", metadata={"label": "input folder with line drawings"})
    object_longest_side: int = field(default=200, metadata={"min": 50, "max": 500, "step": 10, "label": "object longest side (px)"})
    grid_degree: int = field(default=45, metadata={"min": 0, "max": 360, "step": 5, "label": "grid rotation (degrees)"})
    grid_size: int = field(default=8, metadata={"min": 1, "max": 50, "step": 1, "label": "grid cell size (px)"})
    grid_thickness: int = field(default=4, metadata={"min": 1, "max": 20, "step": 1, "label": "grid thickness (px)"})
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    output_folder: str = field(default="data/shape_and_object_recognition/segmented_images", metadata={"label": "output folder"})


@register("segmented_images", "shape_recognition")
@generator(SegmentedImagesConfig)
def generate_all(config: SegmentedImagesConfig):
    """generate segmented images dataset with complementary grid deletions."""
    output_folder = Path(config.output_folder)
    linedrawing_input_folder = Path(config.linedrawing_input_folder)

    all_categories = [p.stem for p in linedrawing_input_folder.glob("*")]
    for ff in ["del", "del_complement"]:
        for cat in all_categories:
            (output_folder / ff / cat).mkdir(parents=True, exist_ok=True)

    ds = DrawGriddedImages(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        obj_longest_side=config.object_longest_side,
        width=1,
    )

    image_files = sorted(linedrawing_input_folder.rglob("*.jpg")) + sorted(linedrawing_input_folder.rglob("*.png"))

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "ClassName", "IsComplement", "BackgroundColor", "GridShift", "GridThickness", "GridDegree"])
        grid_shift = 0
        for complement in tqdm([True, False]):
            for img_path in image_files:
                class_name = img_path.parent.stem
                image_name = img_path.stem
                img = ds.apply_grid_mask(
                    img_path,
                    config.grid_size,
                    grid_shift=grid_shift,
                    grid_thickness=config.grid_thickness,
                    rotation_degrees=config.grid_degree,
                    complement=complement,
                )
                path = Path("del" + ("_complement" if complement else "")) / class_name / f"{image_name}.png"
                img.save(str(output_folder / path))
                writer.writerow([path, class_name, complement, ds.background, grid_shift, config.grid_thickness, config.grid_degree])

    return str(output_folder)

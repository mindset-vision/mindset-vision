"""nap vs mp 2d lines dataset generator."""
import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import PIL.Image as Image
from tqdm import tqdm

from mindset.drawing.base import (
    DrawStimuli,
    paste_linedrawing_onto_canvas,
    resize_image_keep_aspect_ratio,
)
from mindset.utils.misc import apply_antialiasing
from mindset.generators._base import GeneratorConfig, generator, register


class DrawLinedrawings(DrawStimuli):
    """draw line drawings onto a canvas with resizing."""

    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def get_linedrawings(self, image_path):
        """load a grayscale image, resize, and paste onto canvas."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = resize_image_keep_aspect_ratio(img, self.obj_longest_side)
        canvas = paste_linedrawing_onto_canvas(
            Image.fromarray(img), self.create_canvas(), self.fill
        )
        return apply_antialiasing(canvas) if self.antialiasing else canvas


@dataclass
class NapVsMp2dConfig(GeneratorConfig):
    """config for nap vs mp 2d lines dataset."""
    object_longest_side: int = field(default=200, metadata={"min": 10, "max": 1000, "step": 10, "label": "object longest side"})
    input_folder: str = field(default="mindset/assets/kubilius_2017/pngs", metadata={"label": "input folder"})
    output_folder: str = field(default="data/low_mid_level_vision/NAP_vs_MP_2D_lines", metadata={"label": "output folder"})
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})


@register("nap_vs_mp_2d", "low_mid_vision")
@generator(NapVsMp2dConfig)
def generate_all(config: NapVsMp2dConfig):
    """generate nap vs mp 2d lines dataset."""
    output_folder = Path(config.output_folder)
    input_folder = Path(config.input_folder)

    all_categories = [i.stem for i in input_folder.glob("*")]
    for cat in all_categories:
        (output_folder / cat).mkdir(exist_ok=True, parents=True)

    ds = DrawLinedrawings(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        obj_longest_side=config.object_longest_side,
    )

    image_files = list(input_folder.rglob("*.jpg")) + list(input_folder.rglob("*.png"))

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Class", "BackgroundColor", "IterNum"])
        for n, img_path in enumerate(tqdm(image_files)):
            class_name = img_path.parent.stem
            img = ds.get_linedrawings(img_path)
            path = Path(class_name) / f"{n}.png"
            img.save(output_folder / path)
            writer.writerow([path, class_name, ds.background, n])

    return str(output_folder)

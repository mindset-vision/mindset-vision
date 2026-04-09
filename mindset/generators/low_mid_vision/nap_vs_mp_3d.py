"""nap vs mp 3d geons dataset generator."""
import colorsys
import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import PIL.Image as Image
from tqdm import tqdm

from mindset.drawing.base import (DrawStimuli, paste_linedrawing_onto_canvas,
                                  resize_image_keep_aspect_ratio)
from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils import apply_antialiasing


class DrawShape(DrawStimuli):
    """draw 3d geon shapes with optional color remapping."""

    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def process_image(self, image_path, shape_color_rgb):
        """load, resize, paste, and optionally recolor a geon image."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = resize_image_keep_aspect_ratio(img, self.obj_longest_side)
        img = paste_linedrawing_onto_canvas(
            Image.fromarray(img), self.create_canvas(), (255, 255, 255)
        )

        new_img = self.create_canvas(size=img.size)
        colors = {(0, 0, 0): self.background}
        if shape_color_rgb:
            shape_color_hls = colorsys.rgb_to_hls(
                *tuple(np.array(shape_color_rgb) // 255)
            )

        data = img.load()
        new_data = new_img.load()
        for y in range(img.size[1]):
            for x in range(img.size[0]):
                if data[x, y] in colors:
                    new_data[x, y] = colors[data[x, y]]
                elif shape_color_rgb:
                    pixel_rgb = [v / 255 for v in data[x, y]]
                    pixel_hls = colorsys.rgb_to_hls(*pixel_rgb)
                    new_hls = (shape_color_hls[0], pixel_hls[1], shape_color_hls[2])
                    new_rgb = [int(v * 255) for v in colorsys.hls_to_rgb(*new_hls)]
                    new_data[x, y] = tuple(new_rgb)
                else:
                    new_data[x, y] = data[x, y]

        new_img = new_img.resize(self.canvas_size)
        return apply_antialiasing(new_img) if self.antialiasing else new_img


@dataclass
class NapVsMp3dConfig(GeneratorConfig):
    """config for nap vs mp 3d geons dataset."""
    object_longest_side: int = field(default=200, metadata={"min": 10, "max": 1000, "step": 10, "label": "object longest side"})
    stroke_color: str = field(default="", metadata={"label": "stroke color (rgb as 255_255_255 or empty)"})
    shape_folder: str = field(default="mindset/assets/amir_geons/cropped/NAPvsMP", metadata={"label": "shape folder"})
    output_folder: str = field(default="data/low_mid_level_vision/NAP_vs_MP_3D_geons", metadata={"label": "output folder"})


@register("nap_vs_mp_3d", "low_mid_vision")
@generator(NapVsMp3dConfig)
def generate_all(config: NapVsMp3dConfig):
    """generate nap vs mp 3d geons dataset."""
    output_folder = Path(config.output_folder)
    shape_folder = Path(config.shape_folder)
    all_types = ["reference", "MP", "NAP"]

    for t in all_types:
        (output_folder / t).mkdir(exist_ok=True, parents=True)

    stroke_color = config.stroke_color
    if isinstance(stroke_color, str) and "_" in stroke_color:
        stroke_color = [int(i) for i in stroke_color.split("_")]
    elif isinstance(stroke_color, str) and stroke_color == "":
        stroke_color = ""

    ds = DrawShape(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        obj_longest_side=config.object_longest_side,
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Type", "BackgroundColor", "SampleName"])
        for t in tqdm(all_types):
            for i in tqdm((shape_folder / t).glob("*"), leave=False):
                name_sample = i.stem
                img_path = Path(t) / f"{name_sample}.png"
                img = ds.process_image(shape_folder / img_path, stroke_color)
                img.save(output_folder / img_path)
                writer.writerow([img_path, t, ds.background, name_sample])

    return str(output_folder)

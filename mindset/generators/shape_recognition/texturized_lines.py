"""texturized linedrawings (lines) dataset generator."""
import csv
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms import InterpolationMode, transforms
from tqdm import tqdm

from mindset.generators._base import GeneratorConfig, generator, register
from mindset.drawing.base import (
    DrawStimuli,
    get_mask_from_linedrawing,
    resize_image_keep_aspect_ratio,
)
from mindset.drawing.affine import get_affine_rnd_fun, my_affine
from mindset.utils.misc import apply_antialiasing


class DrawPatternedCanvas(DrawStimuli):
    """draws texturized linedrawings using line-segment patterns."""

    def add_line_pattern(self, canvas, line_length=10, slope_rad=0, density=1):
        """tile a canvas with short line segments at a given slope."""
        width, height = canvas.size

        draw = ImageDraw.Draw(canvas)

        line_segment = [
            (0, 0),
            (line_length * math.cos(slope_rad), line_length * math.sin(slope_rad)),
        ]
        horizontal_spacing = int(np.round(line_length / density))
        vertical_spacing = int(np.round(line_length / density))
        noise = 1
        for y in range(
            0,
            height,
            np.round(
                line_length * np.abs(math.sin(slope_rad)) + vertical_spacing
            ).astype(int),
        ):
            for x in range(
                0,
                width,
                np.round(line_length * math.cos(slope_rad) + horizontal_spacing).astype(
                    int
                ),
            ):
                x_offset, y_offset = random.randint(-noise, noise), random.randint(
                    -noise, noise
                )
                shifted_line_segment = [
                    (point[0] + x + x_offset, point[1] + y + y_offset)
                    for point in line_segment
                ]
                draw.line(shifted_line_segment, **self.line_args)
        return canvas

    def __init__(
        self,
        texturize_background,
        texturize_foreground,
        obj_longest_side,
        transform_code,
        density,
        *args,
        **kwargs,
    ):
        self.obj_longest_side = obj_longest_side
        self.density = density
        self.transform_code = transform_code
        self.texturize_foreground = texturize_foreground
        self.texturize_background = texturize_background
        super().__init__(*args, **kwargs)

    def draw_pattern(self, img_path, slope_rad, line_length):
        """draw a texturized linedrawing with line-segment patterns."""
        expand_factor = 1.5
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = get_mask_from_linedrawing(
            resize_image_keep_aspect_ratio(img, self.obj_longest_side), fill=True
        )

        bg_canvas = self.create_canvas(
            size=tuple((np.array(self.canvas_size) * expand_factor).astype(int))
        )
        if self.texturize_background:
            bg_canvas = self.add_line_pattern(
                canvas=bg_canvas,
                line_length=line_length,
                slope_rad=slope_rad,
                density=self.density,
            )
        perpendicular_radian = (
            (slope_rad + math.pi / 2)
            if abs(slope_rad + math.pi / 2) <= math.pi / 2
            else (slope_rad - math.pi / 2)
        )

        fg_canvas = self.create_canvas(size=mask.size, background=self.background)
        if self.texturize_foreground:
            canvas_foreg_text = self.add_line_pattern(
                fg_canvas,
                line_length,
                perpendicular_radian,
                self.density,
            )

        bg_canvas.paste(
            canvas_foreg_text,
            (
                bg_canvas.size[0] // 2 - mask.size[0] // 2,
                bg_canvas.size[1] // 2 - mask.size[1] // 2,
            ),
            mask=mask,
        )

        af = get_affine_rnd_fun(self.transform_code)()
        img = my_affine(
            bg_canvas,
            translate=list(np.array(af["tr"]) / expand_factor),
            angle=af["rt"],
            scale=af["sc"],
            shear=af["sh"],
            interpolation=InterpolationMode.NEAREST,
            fill=self.background,
        )
        img = transforms.CenterCrop((self.canvas_size[1], self.canvas_size[0]))(img)
        return apply_antialiasing(img) if self.antialiasing else img


@dataclass
class TexturizedLinesConfig(GeneratorConfig):
    """config for texturized linedrawings (lines) dataset."""
    linedrawing_input_folder: str = field(default="mindset/assets/baker_2018_linedrawings/cropped/", metadata={"label": "input folder with line drawings"})
    num_samples: int = field(default=500, metadata={"min": 1, "max": 10000, "step": 10, "label": "samples per line drawing"})
    object_longest_side: int = field(default=200, metadata={"min": 50, "max": 500, "step": 10, "label": "object longest side (px)"})
    density: float = field(default=1.8, metadata={"min": 0.1, "max": 10.0, "step": 0.1, "label": "pattern density"})
    texturize_foreground: bool = field(default=True, metadata={"label": "texturize foreground"})
    texturize_background: bool = field(default=False, metadata={"label": "texturize background"})
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    output_folder: str = field(default="data/shape_and_object_recognition/texturized_linedrawings_lines", metadata={"label": "output folder"})


@register("texturized_lines", "shape_recognition")
@generator(TexturizedLinesConfig)
def generate_all(config: TexturizedLinesConfig):
    """generate texturized linedrawings (lines) dataset."""
    output_folder = Path(config.output_folder)
    linedrawing_input_folder = Path(config.linedrawing_input_folder)

    all_categories = [p.stem for p in linedrawing_input_folder.glob("*")]
    for cat in all_categories:
        (output_folder / cat).mkdir(exist_ok=True, parents=True)

    transf_code = {"translation": [-0.1, 0.1]}

    ds = DrawPatternedCanvas(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        texturize_background=config.texturize_background,
        texturize_foreground=config.texturize_foreground,
        obj_longest_side=config.object_longest_side,
        density=config.density,
        width=1,
        transform_code=transf_code,
    )

    jpg_files = list(linedrawing_input_folder.rglob("*.jpg"))
    png_files = list(linedrawing_input_folder.rglob("*.png"))
    image_files = jpg_files + png_files

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Class", "BackgroundColor", "SlopeLine", "LineLength", "IterNum"])

        for img_path in tqdm(image_files):
            class_name = img_path.parent.stem
            image_name = img_path.stem
            for n in tqdm(range(config.num_samples), leave=False):
                slope_line = np.deg2rad(random.uniform(*random.choice([(-60, 60)])))
                line_length = random.randint(4, 8)
                img = ds.draw_pattern(img_path, slope_line, line_length)
                path = Path(class_name) / f"{image_name}_{n}.png"
                img.save(output_folder / path)
                writer.writerow([path, class_name, ds.background, slope_line, line_length, n])

    return str(output_folder)

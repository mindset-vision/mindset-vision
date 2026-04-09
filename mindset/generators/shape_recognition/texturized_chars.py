"""texturized linedrawings (chars) dataset generator."""
import csv
import math
import random
import string
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import transforms
from tqdm import tqdm

from mindset.drawing.base import (DrawStimuli, get_mask_from_linedrawing,
                                  resize_image_keep_aspect_ratio)
from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils import apply_antialiasing

characters = string.ascii_letters + string.digits + string.punctuation


class DrawPatternedCanvas(DrawStimuli):
    """draws texturized linedrawings using character patterns."""

    def __init__(self, obj_longest_side, transform_code, *args, **kwargs):
        self.transform_code = transform_code
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def get_canvas_char_pattered(
        self,
        size,
        tile_char,
        font_size,
        spacing=0,
        rotation_angle=45,
        font_path="mindset/assets/arial.ttf",
        background=None,
    ):
        """create a canvas tiled with a repeated character pattern."""
        font = ImageFont.truetype(font_path, font_size)
        img = self.create_canvas(
            size=tuple(
                [np.round(np.sqrt(size[0] ** 2 + size[1] ** 2)).astype(int)] * 2
            ),
            background=background,
        )

        bbox = font.getbbox(tile_char + " " * spacing)
        char_width = bbox[2]
        char_heights = bbox[3]

        width, height = img.size
        num_x = width // char_width + 2
        draw = ImageDraw.Draw(img)
        tile_string = (tile_char + " " * spacing) * num_x

        for y in range(-20, height, char_heights):
            draw.text(
                (1, y),
                tile_string,
                fill=self.line_args["fill"],
                font=font,
            )
        if rotation_angle > 0:
            img = img.rotate(rotation_angle, resample=Image.Resampling.NEAREST)

        img = transforms.CenterCrop((size[1], size[0]))(img)
        return img

    def draw_pattern(
        self,
        img_path,
        background_char,
        background_font_size,
        rotation_angle_rad,
        foreground_char,
        foreground_font_size,
    ):
        """draw a texturized linedrawing with foreground and background char patterns."""
        opencv_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        opencv_img = resize_image_keep_aspect_ratio(opencv_img, self.obj_longest_side)

        mask = get_mask_from_linedrawing(opencv_img, fill=True)
        canvas = self.get_canvas_char_pattered(
            size=tuple(np.array(self.canvas_size)),
            tile_char=background_char,
            font_size=background_font_size,
            rotation_angle=np.rad2deg(rotation_angle_rad),
            spacing=0,
        )

        perpendicular_radian = (
            (rotation_angle_rad + math.pi / 2)
            if abs(rotation_angle_rad + math.pi / 2) <= math.pi / 2
            else (rotation_angle_rad - math.pi / 2)
        )
        canvas_foreg_text = self.get_canvas_char_pattered(
            size=mask.size,
            tile_char=foreground_char,
            font_size=foreground_font_size,
            rotation_angle=np.rad2deg(perpendicular_radian),
            spacing=0,
            background=self.background,
        )

        canvas.paste(
            canvas_foreg_text,
            (
                canvas.size[0] // 2 - mask.size[0] // 2,
                canvas.size[1] // 2 - mask.size[1] // 2,
            ),
            mask=mask,
        )
        return apply_antialiasing(canvas) if self.antialiasing else canvas


@dataclass
class TexturizedCharsConfig(GeneratorConfig):
    """config for texturized linedrawings (chars) dataset."""
    linedrawing_input_folder: str = field(default="mindset/assets/baker_2018_linedrawings/cropped/", metadata={"label": "input folder with line drawings"})
    num_samples: int = field(default=500, metadata={"min": 1, "max": 10000, "step": 10, "label": "samples per line drawing"})
    object_longest_side: int = field(default=200, metadata={"min": 50, "max": 500, "step": 10, "label": "object longest side (px)"})
    background_char: str = field(default=" ", metadata={"label": "background character"})
    foreground_char: str = field(default="random", metadata={"label": "foreground character"})
    font_size: list = field(default_factory=lambda: [15, 20], metadata={"label": "font size range"})
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    output_folder: str = field(default="data/shape_and_object_recognition/texturized_linedrawings_chars", metadata={"label": "output folder"})


@register("texturized_chars", "shape_recognition")
@generator(TexturizedCharsConfig)
def generate_all(config: TexturizedCharsConfig):
    """generate texturized linedrawings (chars) dataset."""
    output_folder = Path(config.output_folder)
    linedrawing_input_folder = Path(config.linedrawing_input_folder)

    all_categories = [p.stem for p in linedrawing_input_folder.glob("*")]
    for cat in all_categories:
        (output_folder / cat).mkdir(exist_ok=True, parents=True)

    ds = DrawPatternedCanvas(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        obj_longest_side=config.object_longest_side,
        width=1,
        transform_code=None,
    )

    jpg_files = list(linedrawing_input_folder.rglob("*.jpg"))
    png_files = list(linedrawing_input_folder.rglob("*.png"))
    image_files = jpg_files + png_files

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow([
            "Path", "Class", "BackgroundColor", "BackgroundChar",
            "ForegroundChar", "FontSize", "RotationAngle", "IterNum",
        ])

        for img_path in tqdm(image_files):
            for n in tqdm(range(config.num_samples), leave=False):
                class_name = img_path.parent.stem
                rotation_angle = random.randint(-60, 60)
                font_s = (
                    random.randint(config.font_size[0], config.font_size[1])
                    if isinstance(config.font_size, list)
                    else config.font_size
                )
                background_c = random.choice(characters) if config.background_char == "random" else config.background_char
                foreground_c = random.choice(characters) if config.foreground_char == "random" else config.foreground_char

                img = ds.draw_pattern(
                    img_path=img_path,
                    background_char=background_c,
                    foreground_char=foreground_c,
                    background_font_size=font_s,
                    foreground_font_size=font_s,
                    rotation_angle_rad=np.deg2rad(rotation_angle),
                )
                unique_id = uuid.uuid4().hex[:8]
                image_name = img_path.stem
                path = Path(class_name) / f"{image_name}_{unique_id}.png"
                img.save(output_folder / path)
                writer.writerow([
                    path, class_name, ds.background,
                    background_c, foreground_c, font_s, rotation_angle, n,
                ])

    return str(output_folder)

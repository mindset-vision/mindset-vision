"""texturized blobs dataset generator."""

import csv
import math
import random
import string
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import transforms
from tqdm.auto import tqdm

from mindset.drawing.base import DrawStimuli, get_mask_from_linedrawing
from mindset.drawing.shapes.parent import ParentStimuli
from mindset.drawing.shapes.shapes import Shapes
from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils import apply_antialiasing

characters = string.ascii_letters + string.digits + string.punctuation


class DrawPatternedCanvas(DrawStimuli):
    """draws texturized blob shapes using character patterns."""

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
    ):
        """create a canvas tiled with a repeated character pattern."""
        font = ImageFont.truetype(font_path, font_size)
        img = self.create_canvas(
            size=tuple([np.round(np.sqrt(size[0] ** 2 + size[1] ** 2)).astype(int)] * 2)
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

    def draw_blob(self, blob_id):
        """draw a blob shape on a canvas."""
        parent = ParentStimuli(
            target_image_size=self.canvas_size,
            initial_expansion=1,
        )

        blob = Shapes(parent)
        blob.set_color(self.fill)

        blob.add_puddle(size=0.2, seed=blob_id)

        blob.register()
        self.create_canvas()
        parent.add_background(self.background)

        return apply_antialiasing(parent.canvas) if self.antialiasing else parent.canvas

    def draw_pattern(
        self,
        blob_id,
        background_char,
        background_font_size,
        rotation_angle_rad,
        foreground_char,
        foreground_font_size,
    ):
        """draw a texturized blob with foreground and background character patterns."""
        parent = ParentStimuli(
            target_image_size=self.canvas_size,
            initial_expansion=1,
        )

        blob = Shapes(parent)
        blob.set_color((0, 0, 0, 255))
        blob.add_puddle(size=0.2, seed=blob_id)
        blob.register()
        self.create_canvas()
        parent.add_background((255, 255, 255))

        mask = get_mask_from_linedrawing(
            np.array(parent.canvas.convert("L")), fill=True
        )
        canvas_bg = self.get_canvas_char_pattered(
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
        canvas_fg = self.get_canvas_char_pattered(
            size=mask.size,
            tile_char=foreground_char,
            font_size=foreground_font_size,
            rotation_angle=np.rad2deg(perpendicular_radian),
            spacing=0,
        )

        canvas_bg.paste(
            canvas_fg,
            (
                canvas_bg.size[0] // 2 - mask.size[0] // 2,
                canvas_bg.size[1] // 2 - mask.size[1] // 2,
            ),
            mask=mask,
        )
        return apply_antialiasing(canvas_bg) if self.antialiasing else canvas_bg


@dataclass
class TexturizedBlobsConfig(GeneratorConfig):
    """config for texturized blobs dataset."""

    num_samples_per_blob: int = field(
        default=5,
        metadata={"min": 1, "max": 1000, "step": 1, "label": "samples per blob"},
    )
    num_blobs: int = field(
        default=10,
        metadata={"min": 1, "max": 1000, "step": 1, "label": "number of blobs"},
    )
    object_longest_side: int = field(
        default=200,
        metadata={
            "min": 50,
            "max": 500,
            "step": 10,
            "label": "object longest side (px)",
        },
    )
    background_char: str = field(
        default=" ", metadata={"label": "background character"}
    )
    foreground_char: str = field(
        default="random", metadata={"label": "foreground character"}
    )
    font_size: list = field(
        default_factory=lambda: [15, 20], metadata={"label": "font size range"}
    )
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    output_folder: str = field(
        default="data/shape_and_object_recognition/texturized_blobs",
        metadata={"label": "output folder"},
    )


@register("texturized_blobs", "shape_recognition")
@generator(TexturizedBlobsConfig)
def generate_all(config: TexturizedBlobsConfig):
    """generate texturized blobs dataset."""
    output_folder = Path(config.output_folder)
    (output_folder / "blobs").mkdir(exist_ok=True, parents=True)
    (output_folder / "texturized_blobs").mkdir(exist_ok=True, parents=True)

    ds = DrawPatternedCanvas(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        obj_longest_side=config.object_longest_side,
        width=1,
        transform_code=None,
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "BlobID",
                "IsTexturized",
                "BackgroundColor",
                "BackgroundChar",
                "ForegroundChar",
                "FontSize",
                "RotationAngle",
                "IterNum",
            ]
        )

        for blob_id in tqdm(range(config.num_blobs)):
            img = ds.draw_blob(blob_id)
            path = Path("blobs") / f"{blob_id}.png"
            img.save(str(output_folder / path))
            writer.writerow([path, blob_id, False, ds.background, "", "", "", "", 0])

            (output_folder / "texturized_blobs" / str(blob_id)).mkdir(
                exist_ok=True, parents=True
            )

            for n in tqdm(range(config.num_samples_per_blob), leave=False):
                rotation_angle = random.randint(-60, 60)
                font_s = (
                    random.randint(config.font_size[0], config.font_size[1])
                    if isinstance(config.font_size, list)
                    else config.font_size
                )
                background_c = (
                    random.choice(characters)
                    if config.background_char == "random"
                    else config.background_char
                )
                foreground_c = (
                    random.choice(characters)
                    if config.foreground_char == "random"
                    else config.foreground_char
                )

                img = ds.draw_pattern(
                    blob_id=blob_id,
                    background_char=background_c,
                    foreground_char=foreground_c,
                    background_font_size=font_s,
                    foreground_font_size=font_s,
                    rotation_angle_rad=np.deg2rad(rotation_angle),
                )
                hex_id = uuid.uuid4().hex[:8]
                path = Path("texturized_blobs") / str(blob_id) / f"{hex_id}.png"
                img.save(output_folder / path)
                writer.writerow(
                    [
                        path,
                        blob_id,
                        True,
                        ds.background,
                        background_c,
                        foreground_c,
                        font_s,
                        rotation_angle,
                        n,
                    ]
                )

    return str(output_folder)

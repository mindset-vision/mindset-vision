"""texturized linedrawings (chars) dataset generator."""
import csv
import random
import string
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mindset.drawing.texturized_chars import DrawPatternedCanvas
from mindset.generators._base import GeneratorConfig, generator, register

characters = string.ascii_letters + string.digits + string.punctuation


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

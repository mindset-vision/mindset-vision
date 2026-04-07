"""texturized linedrawings (lines) dataset generator."""
import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mindset.drawing.texturized_lines import DrawPatternedCanvas
from mindset.generators._base import GeneratorConfig, generator, register


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

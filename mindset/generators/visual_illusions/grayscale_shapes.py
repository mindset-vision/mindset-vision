"""grayscale shapes dataset generator."""
import csv
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mindset.drawing.grayscale_shapes import (
    ShapeConfigs,
    ColorPickerStimuli,
    add_arrow,
)
from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils.misc import apply_antialiasing


@dataclass
class GrayscaleShapesConfig(GeneratorConfig):
    """config for grayscale shapes dataset."""
    num_samples: int = field(default=5000, metadata={"min": 1, "max": 50000, "step": 10, "label": "number of samples"})
    arrow_size: float = field(default=1, metadata={"min": 0.1, "max": 10, "step": 0.1, "label": "arrow size"})
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    output_folder: str = field(default="data/visual_illusions/grayscale_shapes", metadata={"label": "output folder"})


@register("grayscale_shapes", "visual_illusions")
@generator(GrayscaleShapesConfig)
def generate_all(config: GrayscaleShapesConfig):
    """generate grayscale shapes dataset."""
    output_folder = Path(config.output_folder)
    (output_folder / "all_images").mkdir(exist_ok=True, parents=True)
    shape_configs = ShapeConfigs()

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Target Pixel Color", "Target Pixel Location"])

        for i in tqdm(range(config.num_samples)):
            img = ColorPickerStimuli(config.canvas_size)

            for step in range(20):
                shape_configs._refresh()
                random_shape_config = shape_configs._return_shape_config()
                getattr(img, "add_" + random_shape_config["shape"])(
                    **random_shape_config["parameters"]
                )

            propose_coordinates = img._propose_arrow_coord()
            counter = 0
            while img._count_colors_withing_circle(propose_coordinates) > 1 and counter < 100:
                propose_coordinates = img._propose_arrow_coord()
                counter += 1

            if counter == 100:
                continue

            pixel_color = img._get_pixel_color(propose_coordinates)
            coord = tuple(map(lambda x: int(x * img.canvas.size[0]), propose_coordinates))
            img.canvas = add_arrow(img.canvas, coord, arrow_size=config.arrow_size)
            img.canvas = apply_antialiasing(img) if config.antialiasing else img.canvas
            image_name = str(uuid.uuid4().hex[:8]) + ".png"
            img._shrink_and_save(save_as=output_folder / "all_images" / image_name)
            writer.writerow([str(Path("all_images") / image_name), pixel_color, coord])

    return str(output_folder)

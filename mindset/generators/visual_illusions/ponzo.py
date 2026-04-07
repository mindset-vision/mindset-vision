"""ponzo illusion dataset generator."""
import csv
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mindset.drawing.ponzo import DrawPonzo
from mindset.generators._base import GeneratorConfig, generator, register


@dataclass
class PonzoConfig(GeneratorConfig):
    """config for ponzo illusion dataset."""
    num_samples_scrambled: int = field(default=5000, metadata={"min": 1, "max": 50000, "step": 10, "label": "scrambled samples"})
    num_samples_illusory: int = field(default=500, metadata={"min": 1, "max": 5000, "step": 10, "label": "illusory samples"})
    num_rail_lines: int = field(default=5, metadata={"min": 0, "max": 20, "step": 1, "label": "number of rail lines"})
    rnd_target_lines: bool = field(default=False, metadata={"label": "random target lines"})
    output_folder: str = field(default="data/visual_illusions/ponzo", metadata={"label": "output folder"})


@register("ponzo", "visual_illusions")
@generator(PonzoConfig)
def generate_all(config: PonzoConfig):
    """generate ponzo illusion dataset."""
    output_folder = Path(config.output_folder)
    for d in ["scrambled_lines", "ponzo_same_length", "ponzo_diff_length"]:
        (output_folder / d).mkdir(parents=True, exist_ok=True)

    ds = DrawPonzo(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
        num_rail_lines=config.num_rail_lines,
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Type", "BackgroundColor", "RedLength", "BlueLength", "UpperLineColor", "NumRailLines", "IterNum"])

        for i in tqdm(range(config.num_samples_scrambled)):
            img, red_l, blue_l, _ = ds.generate_rnd_lines_images(
                colored_line_always_horizontal=not config.rnd_target_lines,
                antialias=config.antialiasing,
            )
            unique_hex = uuid.uuid4().hex[:8]
            path = Path("scrambled_lines") / f"{unique_hex}.png"
            img.save(output_folder / path)
            writer.writerow([path, "scrambled_lines", ds.background, red_l, blue_l, "", config.num_rail_lines, i])

        for i in tqdm(range(config.num_samples_illusory)):
            for c in ["ponzo_same_length", "ponzo_diff_length"]:
                img, red_l, blue_l, upper_line_color = ds.generate_illusory_images(
                    same_length=(c == "ponzo_same_length")
                )
                unique_hex = uuid.uuid4().hex[:8]
                path = Path(c) / f"{upper_line_color}_{unique_hex}.png"
                img.save(output_folder / path)
                writer.writerow([path, c, ds.background, red_l, blue_l, upper_line_color, config.num_rail_lines, i])

    return str(output_folder)

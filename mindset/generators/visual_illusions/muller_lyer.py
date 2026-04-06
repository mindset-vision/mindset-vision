"""muller-lyer illusion dataset generator."""
import csv
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mindset.generate_datasets.visual_illusions.muller_lyer_illusion.generate_dataset import DrawMullerLyer
from mindset.utils.misc import apply_antialiasing
from mindset.generators._base import GeneratorConfig, generator, register


@dataclass
class MullerLyerConfig(GeneratorConfig):
    """config for muller-lyer illusion dataset."""
    num_samples_scrambled: int = field(default=5000, metadata={"min": 1, "max": 50000, "step": 10, "label": "scrambled samples"})
    num_samples_illusory: int = field(default=500, metadata={"min": 1, "max": 5000, "step": 10, "label": "illusory samples"})
    output_folder: str = field(default="data/visual_illusions/muller_lyer", metadata={"label": "output folder"})


@register("muller_lyer", "visual_illusions")
@generator(MullerLyerConfig)
def generate_all(config: MullerLyerConfig):
    """generate muller-lyer illusion dataset."""
    output_folder = Path(config.output_folder)
    canvas_size = config.canvas_size
    conditions = ["scrambled", "inward", "outward"]
    for d in conditions:
        (output_folder / d).mkdir(exist_ok=True, parents=True)

    ds = DrawMullerLyer(
        background=config.background_color,
        canvas_size=canvas_size,
        width=1,
    )

    def get_random_params():
        """generate random parameters for one stimulus."""
        line_length = random.randint(int(canvas_size[0] * 0.25), int(canvas_size[0] * 0.67))
        arrow_length = random.randint(int(canvas_size[0] * 0.07), int(canvas_size[0] * 0.134))
        line_position = tuple(
            np.array([
                random.randint(arrow_length + line_length // 2, canvas_size[0] - arrow_length - line_length // 2),
                random.randint(arrow_length + line_length // 2, canvas_size[1] - arrow_length - line_length // 2),
            ]) / canvas_size
        )
        cap_arrows_angle = random.randint(int(canvas_size[0] * 0.045), int(canvas_size[1] * 0.2))
        angle_arrow = random.randint(0, 360)
        return line_length, line_position, arrow_length, cap_arrows_angle, angle_arrow

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Type", "BackgroundColor", "LineLength", "LinePosition", "ArrowLength", "CapArrowAngle", "ArrowAngle", "IterNum"])

        for c in conditions:
            n = config.num_samples_scrambled if c == "scrambled" else config.num_samples_illusory
            for i in tqdm(range(n), desc=c):
                line_length, line_position, arrow_length, cap_arrow_angle, arrow_angle = get_random_params()
                img = ds.generate_illusion(
                    line_position_rel=line_position,
                    line_length=line_length,
                    arrow_angle=arrow_angle,
                    arrow_cap_angle=cap_arrow_angle,
                    arrow_length=arrow_length,
                    type=c,
                )
                if config.antialiasing:
                    img = apply_antialiasing(img)
                unique_hex = uuid.uuid4().hex[:8]
                path = Path(c) / f"{unique_hex}.png"
                img.save(output_folder / path)
                writer.writerow([path, c, ds.background, line_length, line_position, arrow_length, cap_arrow_angle, arrow_angle, i])

    return str(output_folder)
